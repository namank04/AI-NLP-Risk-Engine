"""Learned impact score: predicts the market's reaction from signal features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.config import ROOT, taxonomy, universe

MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "impact_lgbm.txt"
META_FILE = MODEL_DIR / "impact_lgbm.json"
EVENT_CLASSES = [c for c in taxonomy() if c != "none"]
SIGNAL_FEATURES = ["abs_sentiment", "sentiment", "sentiment_conf", "novelty", "log_n_sources", "source_weight"] + \
    [f"p_{c}" for c in EVENT_CLASSES]
CONTEXT_FEATURES = ["stories_today", "vol20", "beta", "log_mcap", "vix"]
FEATURES = SIGNAL_FEATURES + CONTEXT_FEATURES


class MarketContext:
    """As-of market context per (ticker, date), from cached prices and FRED."""

    def __init__(self, prices: pd.DataFrame, vix: pd.Series | None, shares: pd.Series | None = None):
        from riskengine.market.event_study import fit_market_model

        px = prices.sort_index()
        rets = px.pct_change(fill_method=None)
        # every series is shifted one day: on date t we only know the close of t-1
        self.vol20 = rets.rolling(20, min_periods=10).std().shift(1)
        mm = fit_market_model(px[[c for c in px.columns if c in universe().tickers] + ["SPY"]])
        self.beta = mm.beta
        self.log_mcap = np.log((px * shares).shift(1)) if shares is not None else None
        self.vix = vix.reindex(px.index).ffill().shift(1) if vix is not None else None
        self.days = px.index

    def asof(self, ticker: str, when: pd.Timestamp) -> dict[str, float]:
        when = pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)
        i = min(self.days.searchsorted(when.normalize(), side="right") - 1, len(self.days) - 1)
        if i < 0:
            return {k: np.nan for k in ("vol20", "beta", "log_mcap", "vix")}
        d = self.days[i]

        def get(frame, col=None):
            if frame is None:
                return np.nan
            v = frame.at[d, col] if col is not None else frame.at[d]
            return float(v) if pd.notna(v) else np.nan

        has = ticker in self.vol20.columns
        return {"vol20": get(self.vol20, ticker) if has else np.nan,
                "beta": get(self.beta, ticker) if ticker in self.beta.columns else np.nan,
                "log_mcap": get(self.log_mcap, ticker) if self.log_mcap is not None and has else np.nan,
                "vix": get(self.vix)}


def load_context() -> MarketContext:
    from riskengine.market.fred import fred_series
    from riskengine.market.prices import daily_close, shares_outstanding

    u = universe()
    px = daily_close(u.tickers + [u.market_proxy], start="2020-01-01")
    try:
        vix = fred_series(["VIXCLS"])["VIXCLS"]
    except Exception:
        vix = None
    try:
        sh = shares_outstanding(u.tickers)
    except Exception:
        sh = None
    return MarketContext(px, vix, sh)
