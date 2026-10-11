"""Module B trigger: subscribes to Event Classification + Impact Score."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

import pandas as pd

from riskengine.modules.stress.engine import StressResult, run_stress
from riskengine.modules.stress.portfolio import load_portfolio
from riskengine.modules.stress.scenarios import Scenario, idiosyncratic, scenario_config, systemic
from riskengine.schema import Signal

log = logging.getLogger(__name__)
ALWAYS_SYSTEMIC = {"geopolitical", "macroeconomic"}
# 8-K items that are material credit/operational news on their own (no corroboration needed)
MATERIAL_8K = {"1.03", "1.05", "2.04", "2.06", "3.01", "4.02"}
# a systemic credit scenario needs evidence of an actual credit event, not a topic match
SYSTEMIC_CREDIT_SUBTYPES = {"bankruptcy", "failure_to_pay", "restructuring", "liquidity_stress", "rating_action"}


class StressTrigger:
    def __init__(self, portfolio: pd.DataFrame | None = None):
        self.pf = portfolio if portfolio is not None else load_portfolio()[0]
        self.cfg = scenario_config()["trigger"]
        self.runs: list[dict] = []
        self._cooldown: dict[tuple[str, str], tuple[datetime, float, set[str]]] = {}
        self._lock = threading.Lock()
        self.listeners: list = []
        self.baseline = run_stress(self.pf, systemic("macroeconomic", impact=0, severity=0.0)).summary

    def route(self, sig: Signal) -> Scenario | None:
        cls = sig.event.primary
        trig = {"signal_id": sig.signal_id, "entity": sig.entity, "event": cls, "subtype": sig.event.subtype,
                "impact": sig.impact.score, "sentiment": sig.sentiment.score, "headline": sig.evidence.headline,
                "url": sig.evidence.url, "published_at": sig.published_at.isoformat(), "n_sources": sig.n_sources}
        if cls in ALWAYS_SYSTEMIC or (sig.entity_type == "market" and cls in self.cfg["systemic_classes"]):
            if cls not in self.cfg["systemic_classes"]:
                return None
            if cls == "credit_event" and sig.event.subtype not in SYSTEMIC_CREDIT_SUBTYPES:
                return None
            return systemic(cls, sig.impact.score, sig.scope.countries, trigger=trig)
        if sig.entity_type == "company" and cls in self.cfg["idiosyncratic_classes"]:
            hit = self.pf[self.pf["ticker"] == sig.entity]
            if hit.empty:
                return None
            return idiosyncratic(hit["counterparty"].iloc[0], hit["sector"].iloc[0], sig.impact.score,
                                 sig.event.subtype, trigger=trig)
        return None

    def qualifies(self, sig: Signal) -> bool:
        if sig.impact.score < self.cfg["min_impact"]:
            return False
        # stress tests are for adverse news: an upgrade or a ceasefire is not a stress event
        if sig.sentiment.score > self.cfg.get("max_sentiment", -0.2):
            return False
        if sig.source_type == "filing":
            return any(t.removeprefix("8k:") in MATERIAL_8K for t in sig.tags)
        return sig.n_sources >= self.cfg["min_sources"]

    @staticmethod
    def _scenario_key(scn: Scenario) -> tuple[str, str]:
        return ("systemic", scn.key) if scn.kind == "systemic" else ("idiosyncratic", scn.target_counterparty or "")

    def escalates(self, scn: Scenario, at: datetime) -> bool:
        """Within the cooldown, re-run the same scenario only if the situation escalates.

        Re-worded copies of one story land in different clusters, so cooldown is keyed by
        the scenario (class, or counterparty), not by the story.
        """
        last = self._cooldown.get(self._scenario_key(scn))
        if last is None or at - last[0] >= timedelta(hours=self.cfg["cooldown_hours"]):
            return True
        new_scope = set(scn.scope_countries) - last[2]
        return scn.severity >= last[1] + self.cfg.get("escalation_step", 0.15) or bool(new_scope)

    def on_signal(self, sig: Signal) -> StressResult | None:
        with self._lock:
            if not self.qualifies(sig):
                return None
            scn = self.route(sig)
            if scn is None or not self.escalates(scn, sig.published_at):
                return None
            key = self._scenario_key(scn)
            prev_scope = self._cooldown.get(key, (None, 0.0, set()))[2]
            self._cooldown[key] = (sig.published_at, scn.severity, prev_scope | set(scn.scope_countries))
        return self.run(scn, at=sig.published_at)

    def run(self, scn: Scenario, at: datetime | None = None) -> StressResult:
        res = run_stress(self.pf, scn)
        record = {"run_id": f"R{len(self.runs) + 1:04d}", "at": (at or datetime.now()).isoformat()} | res.to_dict()
        with self._lock:
            self.runs.append(record)
        log.info("stress run %s: %s loss=%.1f CET1 %.2f%% -> %.2f%%", record["run_id"], scn.name,
                 res.summary["total_loss"], 100 * res.summary["cet1_ratio_base"],
                 100 * res.summary["cet1_ratio_stressed"])
        for fn in self.listeners:
            try:
                fn(record)
            except Exception:
                log.exception("stress listener failed")
        return res
