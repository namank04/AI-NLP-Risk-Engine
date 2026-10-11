"""Scenario construction: engine signal -> concrete factor shocks.

A Scenario holds per-position shocks already resolved (rate move in the
position's currency, spread move for its rating, equity move for its sector...),
so valuation is pure vector arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from riskengine.config import CONFIG_DIR
from riskengine.modules.stress.portfolio import EM


@lru_cache
def scenario_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "scenarios.yaml").read_text(encoding="utf-8"))


def severity_from_impact(impact: float) -> float:
    return float(np.clip((impact - 4) / 6, 0.2, 1.0))


def z_from_severity(severity: float) -> float:
    """Quantile of the systematic factor implied by a severity (for display only)."""
    return float(norm.ppf(0.5 + 0.49 * severity))


@dataclass
class Scenario:
    key: str
    name: str
    severity: float
    kind: str                                     # "systemic" | "idiosyncratic" | "custom"
    template: dict
    scope_countries: list[str] = field(default_factory=list)
    target_counterparty: str | None = None        # idiosyncratic only
    target_sector: str | None = None
    force_default: bool = False
    trigger: dict = field(default_factory=dict)   # signal that caused it (for the dashboard)

    def describe(self) -> dict:
        return {"key": self.key, "name": self.name, "severity": round(self.severity, 3), "kind": self.kind,
                "scope_countries": self.scope_countries, "target_counterparty": self.target_counterparty,
                "force_default": self.force_default, "trigger": self.trigger}


def systemic(event_class: str, impact: float, scope_countries: list[str] | None = None,
             severity: float | None = None, trigger: dict | None = None) -> Scenario:
    tpl = scenario_config()["systemic"][event_class]
    return Scenario(key=event_class, name=tpl["name"], kind="systemic", template=tpl,
                    severity=severity if severity is not None else severity_from_impact(impact),
                    scope_countries=[c for c in (scope_countries or []) if c != "US"], trigger=trigger or {})


def idiosyncratic(counterparty: str, sector: str, impact: float, subtype: str | None = None,
                  severity: float | None = None, trigger: dict | None = None) -> Scenario:
    tpl = scenario_config()["idiosyncratic"]
    return Scenario(key="idiosyncratic", name=f"{tpl['name']}: {counterparty}", kind="idiosyncratic",
                    template=tpl, severity=severity if severity is not None else severity_from_impact(impact),
                    target_counterparty=counterparty, target_sector=sector,
                    force_default=subtype in tpl["default_subtypes"], trigger=trigger or {})


def _lookup(table: dict, key: str, default_key: str = "_all", default: float = 0.0) -> float:
    return float(table.get(key, table.get(default_key, default)))


@dataclass
class ResolvedShocks:
    """Per-position shock vectors (aligned with the portfolio rows)."""
    d_rate: np.ndarray          # decimal (0.01 = 100bp), in the position's currency curve
    d_spread: np.ndarray        # decimal, credit or sovereign spread
    equity: np.ndarray          # fractional price change
    fx: np.ndarray              # fractional change of position currency vs USD
    credit_z: np.ndarray        # systematic factor in std devs (adverse > 0)
    drawdown: np.ndarray        # fraction of undrawn revolver drawn
    default: np.ndarray         # bool, forced default


def resolve(scn: Scenario, pf: pd.DataFrame) -> ResolvedShocks:
    n = len(pf)
    s = scn.severity
    tpl = scn.template
    ccy_em = pf["currency"].isin(["CNY", "INR", "BRL", "MXN", "TWD"])
    in_scope = pf["country"].isin(scn.scope_countries).to_numpy()

    if scn.kind == "idiosyncratic":
        target = (pf["counterparty"] == scn.target_counterparty).to_numpy()
        same_sector = (pf["sector"] == scn.target_sector).to_numpy() & ~target
        weight = np.where(target, 1.0, np.where(same_sector, tpl["sector_contagion"], 0.0))
        return ResolvedShocks(
            d_rate=np.zeros(n),
            d_spread=weight * tpl["credit_spread_bp"] * s / 1e4,
            equity=weight * tpl["equity_pct"] * s / 100,
            fx=np.zeros(n),
            credit_z=weight * tpl["credit_z"] * s,
            drawdown=np.where(target, 0.5 * s, 0.0),
            default=target & scn.force_default,
        )

    cal = tpl.get("calibrated", {})
    rates = dict(tpl["rates_bp"])
    if "USD" in cal.get("rates_bp", {}):
        rates["USD"] = cal["rates_bp"]["USD"]
    d_rate = np.array([rates.get(c, rates.get("EM", 0.0)) if em else rates.get(c, 0.0)
                       for c, em in zip(pf["currency"], ccy_em, strict=True)], dtype=float) * s / 1e4

    spreads = cal.get("credit_spread_bp") or tpl["credit_spread_bp"]
    is_sov = (pf["instrument"] == "sovereign_bond").to_numpy()
    sov_em = is_sov & pf["country"].isin(EM).to_numpy()
    d_spread = np.where(is_sov, 0.0, pf["rating"].map(spreads).fillna(0).to_numpy()) * s
    d_spread = d_spread + np.where(sov_em, tpl.get("sovereign_spread_bp", {}).get("EM", 0.0) * s, 0.0)

    eq_tab = tpl["equity_pct"] | cal.get("equity_pct", {})
    equity = pf["sector"].map(lambda x: _lookup(eq_tab, x)).to_numpy() * s / 100
    fx_tab = tpl["fx_pct"] | cal.get("fx_pct", {})
    fx = pf["currency"].map(lambda c: fx_tab.get(c, 0.0)).to_numpy() * s / 100
    z_tab = tpl["credit_z"]
    credit_z = pf["sector"].map(lambda x: _lookup(z_tab, x)).to_numpy()
    credit_z = credit_z * s      # template z is the full-severity factor shock in std devs

    add = tpl.get("scope_addon", {})
    if in_scope.any():
        d_spread = d_spread + np.where(in_scope & is_sov, add.get("sovereign_spread_bp", 0) * s, 0.0)
        d_spread = d_spread + np.where(in_scope & ~is_sov, add.get("sovereign_spread_bp", 0) * s * 0.5, 0.0)
        fx = fx + np.where(in_scope & (pf["currency"] != "USD").to_numpy(), add.get("fx_pct", 0) * s / 100, 0.0)
        credit_z = credit_z + np.where(in_scope, add.get("credit_z", 0) * s, 0.0)
        equity = equity + np.where(in_scope, add.get("equity_pct", 0) * s / 100, 0.0)

    return ResolvedShocks(
        d_rate=d_rate, d_spread=d_spread / 1e4, equity=equity, fx=fx, credit_z=credit_z,
        drawdown=np.full(n, tpl.get("revolver_drawdown", 0.0) * s), default=np.zeros(n, dtype=bool),
    )
