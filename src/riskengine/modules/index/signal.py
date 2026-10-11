"""Per-ticker sentiment state built from the engine's signal stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from riskengine.schema import Signal


@dataclass
class _State:
    num: float = 0.0
    den: float = 0.0
    t: datetime | None = None
    n: int = 0


@dataclass
class SentimentBook:
    half_life: timedelta = timedelta(hours=36)
    prior_mass: float = 1.5
    states: dict[str, _State] = field(default_factory=dict)

    def _decay(self, st: _State, t: datetime) -> None:
        if st.t is not None and t > st.t:
            f = 0.5 ** ((t - st.t) / self.half_life)
            st.num *= f
            st.den *= f
        st.t = max(st.t, t) if st.t else t

    @staticmethod
    def weight(sig: Signal) -> float:
        fresh = 0.4 + 0.6 * sig.novelty
        return sig.source_weight * max(sig.sentiment.confidence, 0.34) * fresh * (sig.impact.score / 5.0)

    def update(self, sig: Signal) -> None:
        if sig.entity_type != "company":
            return
        st = self.states.setdefault(sig.entity, _State())
        self._decay(st, sig.published_at)
        w = self.weight(sig)
        st.num += w * sig.sentiment.score
        st.den += w
        st.n += 1

    def snapshot(self, t: datetime, tickers: list[str]) -> dict[str, dict]:
        out = {}
        for tk in tickers:
            st = self.states.get(tk)
            if st is None:
                out[tk] = {"tilt": 0.0, "sentiment": 0.0, "evidence": 0.0, "n": 0}
                continue
            self._decay(st, t)
            mean = st.num / st.den if st.den > 1e-9 else 0.0
            out[tk] = {"tilt": st.num / (st.den + self.prior_mass), "sentiment": mean, "evidence": st.den, "n": st.n}
        return out
