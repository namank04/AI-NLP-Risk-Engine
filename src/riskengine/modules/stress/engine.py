"""Stress engine: apply a scenario to the book, report losses and capital."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from riskengine.modules.stress.credit import (
    ifrs9_ecl,
    ifrs9_stage,
    implied_rating,
    irb_correlation,
    irb_rwa,
    vasicek_conditional_pd,
)
from riskengine.modules.stress.scenarios import ResolvedShocks, Scenario, resolve

TAX_RATE = 0.21
INITIAL_CET1_RATIO = 0.135        # bank calibrated to start at 13.5% CET1

OP_RWA_SHARE = 0.12               # operational RWA ~ 12% of credit + market RWA
STRESSED_MKT_RWA = 0.25           # market RWA uplift at full severity (stressed VaR proxy)
CET1_MIN = 0.045
CET1_BUFFERS = 0.025 + 0.01       # capital conservation buffer + an illustrative G-SIB/D-SIB surcharge

FACTOR_BLOCKS = ["rates", "spreads", "equity", "fx", "credit"]


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN -> None) for the API and the event stream."""
    return json.loads(df.to_json(orient="records"))


@dataclass
class BookState:
    pd_1y: np.ndarray
    rating: np.ndarray
    ead: np.ndarray
    ecl: np.ndarray
    rwa_credit: np.ndarray


CCF_UNDRAWN = 0.75                # foundation-IRB credit conversion factor for committed lines
SPREAD_RISK_WEIGHT_BP = {"AAA": 50, "AA": 60, "A": 75, "BBB": 100, "BB": 200, "B": 300, "CCC": 500}


def market_risk_capital(pf: pd.DataFrame) -> dict[str, float]:
    """Simplified sensitivity-based market-risk capital (in the spirit of the Basel standardised approach).

    rates   |net DV01| per currency x 100 bp, plus 10% of gross DV01 for curve/basis risk
    spread  |spread DV01| x rating-bucket risk weight (bonds and CDS, no diversification)
    equity  25% of gross equity exposure
    fx      8% of the larger of total long / total short net open currency positions
    """
    inst = pf["instrument"]
    bond = inst.isin(["corp_bond", "sovereign_bond"])
    dv01 = np.where(bond, -pf["market_value"] * pf["mod_duration"] * 1e-4, 0.0) + pf["dv01"].fillna(0).to_numpy()
    by_ccy = pd.Series(dv01).groupby(pf["currency"].to_numpy()).sum()
    k_rates = float(by_ccy.abs().sum() * 100 + 0.10 * np.abs(dv01).sum() * 100)
    sdv01 = np.where(inst == "corp_bond", pf["market_value"] * pf["mod_duration"] * 1e-4, 0.0) \
        + np.where(inst == "cds", -pf["cs01"].fillna(0), 0.0)
    rw = pf["rating"].map(SPREAD_RISK_WEIGHT_BP).fillna(100).to_numpy()
    k_spread = float((np.abs(sdv01) * rw).sum())
    k_equity = float(0.25 * pf.loc[inst == "equity", "market_value"].abs().sum())
    fx_pos = np.where(inst == "fx_forward", pf["notional"] * pf["direction"], 0.0) \
        + np.where(bond & (pf["currency"] != "USD"), pf["market_value"], 0.0)
    nop = pd.Series(fx_pos).groupby(pf["currency"].to_numpy()).sum().drop("USD", errors="ignore")
    k_fx = float(0.08 * max(nop[nop > 0].sum(), -nop[nop < 0].sum(), 0.0))
    return {"rates": k_rates, "spread": k_spread, "equity": k_equity, "fx": k_fx,
            "total": k_rates + k_spread + k_equity + k_fx}


def _ead(pf: pd.DataFrame, drawdown: np.ndarray | None = None) -> np.ndarray:
    """EAD = drawn + stressed drawdown + CCF x remaining undrawn; bonds at par."""
    drawn = pf["drawn"].fillna(0).to_numpy()
    undrawn = pf["undrawn"].fillna(0).to_numpy()
    extra = undrawn * (0.0 if drawdown is None else drawdown)
    ead = np.where(pf["book"] == "banking", drawn + extra + CCF_UNDRAWN * (undrawn - extra), 0.0)
    bonds = (pf["instrument"] == "corp_bond").to_numpy()
    return np.where(bonds, pf["notional"].to_numpy(), ead)


def book_state(pf: pd.DataFrame, shocks: ResolvedShocks | None = None) -> BookState:
    pd0 = pf["pd_1y"].to_numpy()
    lgd = pf["lgd"].fillna(0.45).to_numpy()
    mat = pf["maturity"].fillna(1.0).to_numpy()
    if shocks is None:
        pd_s, rating, ead = pd0, pf["rating"].to_numpy(), _ead(pf)
    else:
        pd_s = vasicek_conditional_pd(pd0, irb_correlation(pd0), shocks.credit_z)
        pd_s = np.maximum(pd_s, pd0 * (shocks.credit_z >= 0))   # favourable z never improves PD in a stress
        pd_s = np.where(shocks.default, 1.0, pd_s)
        rating = implied_rating(pd_s)
        ead = _ead(pf, shocks.drawdown)
    credit_exposure = pf["instrument"].isin(["term_loan", "revolver", "corp_bond"]).to_numpy()
    stage = ifrs9_stage(pf["rating_orig"].to_numpy(), rating)
    banking = (pf["book"] == "banking").to_numpy()
    ecl = np.where(banking, ifrs9_ecl(pd_s, lgd, ead, mat, stage, pd_ttc=pd0), 0.0)
    rwa = np.where(credit_exposure, irb_rwa(np.minimum(pd_s, 0.999), lgd, ead, mat), 0.0)
    rwa = np.where(np.asarray(pd_s) >= 1.0, 0.0, rwa)            # defaulted: capital via ECL instead
    return BookState(pd_s, rating, ead, ecl, rwa)


def mtm_pnl(pf: pd.DataFrame, sh: ResolvedShocks) -> np.ndarray:
    inst = pf["instrument"].to_numpy()
    mv = pf["market_value"].to_numpy()
    notional = pf["notional"].to_numpy()
    direction = pf["direction"].to_numpy()
    md = pf["mod_duration"].to_numpy()
    cx = pf["convexity"].to_numpy()
    usd = (pf["currency"] == "USD").to_numpy()

    dy = sh.d_rate + sh.d_spread
    bond = np.isin(inst, ["corp_bond", "sovereign_bond"])
    bond_pnl = mv * (-md * dy + 0.5 * cx * dy ** 2)
    # bonds in default recover (1 - LGD) of par
    bond_pnl = np.where(bond & sh.default, -mv * pf["lgd"].fillna(0.45).to_numpy(), bond_pnl)
    pnl = np.where(bond, bond_pnl, 0.0)
    pnl += np.where(inst == "irs", pf["dv01"].to_numpy() * sh.d_rate * 1e4, 0.0)
    cds_default = notional * pf["lgd"].fillna(0.6).to_numpy() * direction
    pnl += np.where(inst == "cds", np.where(sh.default, cds_default, pf["cs01"].to_numpy() * sh.d_spread * 1e4), 0.0)
    pnl += np.where(inst == "equity", mv * sh.equity, 0.0)
    pnl += np.where(inst == "fx_forward", notional * sh.fx * direction, 0.0)
    # translation of non-USD bond holdings into USD (after the price move)
    pnl += np.where(bond & ~usd, (mv + bond_pnl) * sh.fx, 0.0)
    return pnl


@dataclass
class StressResult:
    scenario: dict
    positions: pd.DataFrame
    summary: dict
    by_asset_class: list[dict]
    by_sector: list[dict]
    by_country: list[dict]
    by_rating_migration: list[dict]
    attribution: dict
    top_losses: list[dict]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "positions"}


def _block_only(sh: ResolvedShocks, block: str) -> ResolvedShocks:
    zero = np.zeros_like(sh.d_rate)
    keep = {
        "rates": dict(d_rate=sh.d_rate),
        "spreads": dict(d_spread=sh.d_spread),
        "equity": dict(equity=sh.equity),
        "fx": dict(fx=sh.fx),
        "credit": dict(credit_z=sh.credit_z, drawdown=sh.drawdown, default=sh.default),
    }[block]
    base = ResolvedShocks(zero, zero, zero, zero, zero, zero, np.zeros_like(sh.default))
    return replace(base, **keep)


def _loss(pf: pd.DataFrame, base: BookState, sh: ResolvedShocks) -> float:
    st = book_state(pf, sh)
    return float(-mtm_pnl(pf, sh).sum() + (st.ecl - base.ecl).sum())


def run_stress(pf: pd.DataFrame, scn: Scenario, attribution: bool = True) -> StressResult:
    sh = resolve(scn, pf)
    base = book_state(pf)
    stressed = book_state(pf, sh)
    pnl = mtm_pnl(pf, sh)
    d_ecl = stressed.ecl - base.ecl
    loss = -pnl + d_ecl                                   # positive = loss

    mkt_k = market_risk_capital(pf)
    mkt_rwa = 12.5 * mkt_k["total"]
    rwa0 = base.rwa_credit.sum() + mkt_rwa
    rwa0 *= 1 + OP_RWA_SHARE
    cet1_0 = INITIAL_CET1_RATIO * rwa0
    # systemic stress inflates market RWA (stressed VaR); a single-name event does not
    mkt_mult = 1 + STRESSED_MKT_RWA * scn.severity if scn.kind == "systemic" else 1.0
    rwa1 = (stressed.rwa_credit.sum() + mkt_rwa * mkt_mult) * (1 + OP_RWA_SHARE)
    after_tax_loss = loss.sum() * (1 - TAX_RATE) if loss.sum() > 0 else loss.sum()
    cet1_1 = cet1_0 - after_tax_loss

    pos = pf[["position_id", "instrument", "book", "counterparty", "ticker", "sector", "country", "currency",
              "rating_orig", "notional", "market_value"]].copy()
    pos["rating_stressed"] = stressed.rating
    pos["pd_base"], pos["pd_stressed"] = base.pd_1y, stressed.pd_1y
    pos["ead_base"], pos["ead_stressed"] = base.ead, stressed.ead
    pos["mtm_pnl"], pos["ecl_increase"], pos["loss"] = pnl, d_ecl, loss

    def group(col: str) -> list[dict]:
        g = pos.groupby(col).agg(exposure=("notional", "sum"), mtm_pnl=("mtm_pnl", "sum"),
                                 ecl_increase=("ecl_increase", "sum"), loss=("loss", "sum")).reset_index()
        return _records(g.sort_values("loss", ascending=False).round(3))

    migr = (pos[pos["instrument"].isin(["term_loan", "revolver", "corp_bond"])]
            .groupby(["rating_orig", "rating_stressed"]).agg(exposure=("notional", "sum"), n=("position_id", "count"))
            .reset_index().round(2))
    migr = _records(migr)
    attr: dict = {}
    if attribution:
        attr = {b: round(_loss(pf, base, _block_only(sh, b)), 3) for b in FACTOR_BLOCKS}
        attr["interaction"] = round(float(loss.sum()) - sum(attr.values()), 3)

    summary = {
        "total_loss": round(float(loss.sum()), 3),
        "mtm_loss": round(float(-pnl.sum()), 3),
        "ecl_increase": round(float(d_ecl.sum()), 3),
        "portfolio_value": round(float(pf["market_value"].sum()), 3),
        "portfolio_value_stressed": round(float(pf["market_value"].sum() + pnl.sum() - d_ecl.sum()), 3),
        "ecl_base": round(float(base.ecl.sum()), 3),
        "ecl_stressed": round(float(stressed.ecl.sum()), 3),
        "rwa_base": round(float(rwa0), 2),
        "rwa_stressed": round(float(rwa1), 2),
        "cet1_base": round(float(cet1_0), 2),
        "cet1_stressed": round(float(cet1_1), 2),
        "cet1_ratio_base": round(float(cet1_0 / rwa0), 5),
        "cet1_ratio_stressed": round(float(cet1_1 / rwa1), 5),
        "cet1_requirement": CET1_MIN + CET1_BUFFERS,
        "cet1_minimum": CET1_MIN,
        "breaches_buffer": bool(cet1_1 / rwa1 < CET1_MIN + CET1_BUFFERS),
        "breaches_minimum": bool(cet1_1 / rwa1 < CET1_MIN),
        "stage2_exposure": round(float(pos.loc[(ifrs9_stage(pf["rating_orig"].to_numpy(), stressed.rating) == 2)
                                               & (pos["book"] == "banking"), "notional"].sum()), 2),
        "units": "USD millions",
    }
    top = pos.sort_values("loss", ascending=False).head(10)
    return StressResult(
        scenario=scn.describe(), positions=pos, summary=summary, by_asset_class=group("instrument"),
        by_sector=group("sector"), by_country=group("country"), by_rating_migration=migr,
        attribution=attr, top_losses=_records(top.round(3)),
    )


def severity_curve(pf: pd.DataFrame, make_scenario, steps: np.ndarray | None = None) -> dict:
    """Reverse stress test: CET1 ratio as severity rises, and where it breaches.

    `make_scenario(severity)` builds the scenario. Breach points are linearly
    interpolated between grid steps and also expressed as the engine impact score
    that would produce that severity (severity = (impact - 4) / 6).
    """
    steps = np.round(np.arange(0.0, 1.55, 0.05), 2) if steps is None else steps
    pts = []
    for s in steps:
        r = run_stress(pf, make_scenario(float(s)), attribution=False).summary
        pts.append({"severity": float(s), "cet1_ratio": r["cet1_ratio_stressed"], "loss": r["total_loss"]})
    req = CET1_MIN + CET1_BUFFERS

    def breach(level: float) -> dict | None:
        for a, b in zip(pts, pts[1:], strict=False):
            if a["cet1_ratio"] >= level > b["cet1_ratio"]:
                f = (a["cet1_ratio"] - level) / (a["cet1_ratio"] - b["cet1_ratio"])
                sev = a["severity"] + f * (b["severity"] - a["severity"])
                return {"severity": round(sev, 3), "impact_equivalent": round(4 + 6 * sev, 1)}
        return None

    return {"points": pts, "requirement": req, "minimum": CET1_MIN,
            "breach_buffer": breach(req), "breach_minimum": breach(CET1_MIN)}
