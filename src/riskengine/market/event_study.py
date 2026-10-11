"""Event-study machinery: abnormal returns under the market model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EST_WINDOW = 220   # trading days of estimation data
EST_GAP = 30       # trading days between estimation window end and t0


@dataclass
class MarketModel:
    alpha: pd.DataFrame     # date x ticker, parameters usable on that date (estimated up to t-EST_GAP)
    beta: pd.DataFrame
    sigma: pd.DataFrame
    ar: pd.DataFrame        # abnormal returns


def fit_market_model(prices: pd.DataFrame, market: str = "SPY") -> MarketModel:
    rets = prices.sort_index().pct_change(fill_method=None)
    rm = rets[market]
    stocks = rets.drop(columns=[market])
    win = EST_WINDOW
    mean_m = rm.rolling(win, min_periods=win // 2).mean()
    var_m = rm.rolling(win, min_periods=win // 2).var()
    alpha, beta, sigma = {}, {}, {}
    for t in stocks:
        ri = stocks[t]
        cov = ri.rolling(win, min_periods=win // 2).cov(rm)
        b = cov / var_m
        a = ri.rolling(win, min_periods=win // 2).mean() - b * mean_m
        resid_var = (ri.rolling(win, min_periods=win // 2).var() - b ** 2 * var_m).clip(lower=1e-10)
        # shift so parameters used on day t come from data ending at t - EST_GAP
        alpha[t], beta[t], sigma[t] = a.shift(EST_GAP), b.shift(EST_GAP), np.sqrt(resid_var).shift(EST_GAP)
    alpha, beta, sigma = pd.DataFrame(alpha), pd.DataFrame(beta), pd.DataFrame(sigma)
    ar = stocks - (alpha + beta.mul(rm, axis=0))
    return MarketModel(alpha, beta, sigma, ar)


def event_day(dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day on or after each (date-only) publication date."""
    d = pd.to_datetime(dates).dt.tz_localize(None).dt.normalize() if getattr(dates.dt, "tz", None) else \
        pd.to_datetime(dates).dt.normalize()
    idx = trading_days.searchsorted(d.to_numpy())
    idx = np.clip(idx, 0, len(trading_days) - 1)
    out = pd.Series(trading_days[idx], index=dates.index)
    return out.where(d <= trading_days[-1])


def scar(mm: MarketModel, tickers: pd.Series, t0: pd.Series, k: int = 1) -> pd.DataFrame:
    """Standardised CAR[0,k] (and raw CAR) for each (ticker, t0) pair."""
    days = mm.ar.index
    pos = days.searchsorted(pd.to_datetime(t0).to_numpy())
    car = np.full(len(t0), np.nan)
    sig = np.full(len(t0), np.nan)
    ar_np = mm.ar
    for j, (tk, p) in enumerate(zip(tickers.to_numpy(), pos, strict=True)):
        if tk not in ar_np.columns or p + k >= len(days):
            continue
        window = ar_np[tk].iloc[p:p + k + 1]
        if window.isna().any():
            continue
        car[j] = window.sum()
        sig[j] = mm.sigma[tk].iloc[p]
    return pd.DataFrame({"car": car, "sigma": sig, "scar": car / (sig * np.sqrt(k + 1))}, index=t0.index)
