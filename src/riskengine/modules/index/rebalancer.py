"""Live sentiment-tilted index (Module A)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from riskengine.config import universe
from riskengine.modules.index.optimizer import BLConfig, black_litterman, ledoit_wolf_cov, optimise
from riskengine.modules.index.signal import SentimentBook
from riskengine.schema import Signal

log = logging.getLogger(__name__)


def market_state(tickers: list[str], when: pd.Timestamp | None = None,
                 lookback: int = 252) -> tuple[pd.Series, pd.DataFrame | None]:
    """(cap weights, Ledoit-Wolf covariance) as of `when`; equal weights and no covariance if offline."""
    try:
        from riskengine.market.prices import daily_close, shares_outstanding

        px = daily_close(tickers, start="2020-01-01").ffill()
        if when is not None:
            px = px.loc[: pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)]
        sh = shares_outstanding(tickers)
        cap = (px.iloc[-1] * sh).reindex(tickers)
        if cap.isna().any() or (cap <= 0).any():
            raise ValueError("incomplete market caps")
        cov = ledoit_wolf_cov(px.pct_change(fill_method=None).tail(lookback)[tickers])
        return cap / cap.sum(), cov
    except Exception as exc:  # offline or rate-limited: fall back, but say so
        log.warning("market data unavailable (%s); equal weights, exp-tilt only", exc)
        return pd.Series(1.0 / len(tickers), index=tickers), None


def clip_normalise(w: pd.Series, lo: float, hi: float, iters: int = 50) -> pd.Series:
    """Project onto {sum w = 1, lo <= w <= hi} by iterative clipping."""
    w = w / w.sum()
    for _ in range(iters):
        clipped = w.clip(lo, hi)
        free = (clipped > lo) & (clipped < hi)
        excess = 1.0 - clipped.sum()
        if abs(excess) < 1e-10 or not free.any():
            return clipped / clipped.sum()
        w = clipped.copy()
        w[free] += excess * clipped[free] / clipped[free].sum()
    return w / w.sum()


@dataclass
class RebalancerConfig:
    method: str = "black_litterman"
    kappa: float = 1.2               # exp_tilt strength
    min_weight: float = 0.005        # exp_tilt floor
    max_weight: float = 0.20
    band: float = 0.01               # one-way turnover threshold
    check_interval_s: float = 900.0  # re-evaluate targets at most every 15 minutes of stream time
    bl: BLConfig = field(default_factory=BLConfig)


@dataclass
class Snapshot:
    ts: datetime
    weights: dict[str, float]
    benchmark: dict[str, float]
    signals: dict[str, dict]
    turnover: float
    reason: str
    method: str
    ex_ante_te: float | None = None
    alpha_bps: dict[str, float] = field(default_factory=dict)
    trigger_signal_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ts": self.ts.isoformat(), "weights": self.weights, "benchmark": self.benchmark,
                "signals": self.signals, "turnover": self.turnover, "reason": self.reason, "method": self.method,
                "ex_ante_te": self.ex_ante_te, "alpha_bps": self.alpha_bps,
                "trigger_signal_ids": self.trigger_signal_ids}


class IndexRebalancer:
    def __init__(self, cfg: RebalancerConfig | None = None, benchmark: pd.Series | None = None,
                 book: SentimentBook | None = None):
        self.cfg = cfg or RebalancerConfig()
        self.tickers = universe().tickers
        self.book = book or SentimentBook()
        self.history: list[Snapshot] = []
        self._pending: list[str] = []
        self._last_check: datetime | None = None
        self._lock = threading.Lock()
        self.listeners: list = []
        if benchmark is not None:
            self.benchmark, self.cov = benchmark, None
        else:
            self.benchmark, self.cov = market_state(self.tickers)
        self.weights = self.benchmark.copy()
        self.as_of_date: str | None = None

    # ---- setup ------------------------------------------------------------------
    def as_of(self, when: datetime) -> None:
        """Re-anchor benchmark and covariance to `when` (start of a replay) and reset state."""
        self.benchmark, self.cov = market_state(self.tickers, pd.Timestamp(when))
        self.as_of_date = pd.Timestamp(when).date().isoformat()
        self.reset()

    def reset(self) -> None:
        self.book = SentimentBook(half_life=self.book.half_life, prior_mass=self.book.prior_mass)
        self.weights = self.benchmark.copy()
        self.history.clear()
        self._pending = []
        self._last_check = None

    @property
    def method(self) -> str:
        return self.cfg.method if (self.cfg.method != "black_litterman" or self.cov is not None) else "exp_tilt"

    # ---- construction -------------------------------------------------------------
    def target_weights(self, sig_state: dict[str, dict]) -> tuple[pd.Series, dict[str, float]]:
        x = pd.Series({t: sig_state[t]["tilt"] for t in self.tickers}).clip(-1, 1)
        if self.method == "black_litterman":
            ev = pd.Series({t: sig_state[t]["evidence"] for t in self.tickers})
            mu, pi = black_litterman(self.cov, self.benchmark, x, ev, self.cfg.bl)
            alpha = ((mu - pi) * 252 * 1e4).round(1).to_dict()
            return optimise(mu, self.cov, self.benchmark, self.cfg.bl), alpha
        raw = self.benchmark * np.exp(self.cfg.kappa * x)
        hi = max(self.cfg.max_weight, float(self.benchmark.max()) + 0.05)
        return clip_normalise(raw, self.cfg.min_weight, hi), {}

    def ex_ante_te(self, w: pd.Series) -> float | None:
        if self.cov is None:
            return None
        a = (w - self.benchmark).reindex(self.cov.index).to_numpy()
        return float(np.sqrt(max(a @ self.cov.to_numpy() @ a, 0.0) * 252))

    # ---- streaming ------------------------------------------------------------------
    def on_signal(self, sig: Signal) -> None:
        """Bus callback: subscribe this to the engine's signal topic."""
        if sig.entity_type != "company" or sig.entity not in self.benchmark.index:
            return
        with self._lock:
            self.book.update(sig)
            self._pending.append(sig.signal_id)
            self.maybe_rebalance(sig.published_at)

    def maybe_rebalance(self, now: datetime, force: bool = False) -> Snapshot | None:
        if not self.history:
            empty = {t: {"tilt": 0.0, "sentiment": 0.0, "evidence": 0.0, "n": 0} for t in self.tickers}
            self._record(now, self.weights, empty, 0.0, "inception", {})
        if not force and self._last_check and (now - self._last_check).total_seconds() < self.cfg.check_interval_s:
            return None
        self._last_check = now
        state = self.book.snapshot(now, self.tickers)
        target, alpha = self.target_weights(state)
        turnover = float((target - self.weights).abs().sum() / 2)
        if not force and turnover < self.cfg.band:
            return None
        self.weights = target
        return self._record(now, target, state, turnover, "forced" if force else "band_exceeded", alpha)

    def _record(self, now, weights, state, turnover, reason, alpha) -> Snapshot:
        te = self.ex_ante_te(weights)
        snap = Snapshot(ts=now, weights={k: round(float(v), 6) for k, v in weights.items()},
                        benchmark={k: round(float(v), 6) for k, v in self.benchmark.items()},
                        signals={k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in state.items()},
                        turnover=round(turnover, 6), reason=reason, method=self.method,
                        ex_ante_te=None if te is None else round(te, 5), alpha_bps=alpha,
                        trigger_signal_ids=self._pending[-20:])
        self._pending = []
        self.history.append(snap)
        for fn in self.listeners:
            try:
                fn(snap)
            except Exception:
                log.exception("index listener failed")
        return snap
