"""Daily adjusted close prices and shares outstanding, cached as parquet."""

from __future__ import annotations

from datetime import date

import pandas as pd

from riskengine.config import RAW_DIR

PRICE_CACHE = RAW_DIR / "prices_daily.parquet"


def _download(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True, progress=False,
                     group_by="column", threads=True)
    if df.empty:
        return pd.DataFrame()
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def daily_close(tickers: list[str], start: str = "2010-01-01", end: str | None = None,
                refresh: bool = False) -> pd.DataFrame:
    """Adjusted close, one column per ticker. Cached; only missing tickers/dates are fetched."""
    cached = pd.read_parquet(PRICE_CACHE) if PRICE_CACHE.exists() and not refresh else pd.DataFrame()
    missing = [t for t in tickers if t not in cached.columns]
    stale = not cached.empty and cached.index.max().date() < (pd.Timestamp(end).date() if end else date.today()) \
        - pd.Timedelta(days=3)
    # the cache only grows forward; a request for an earlier history must refetch
    too_short = not cached.empty and bool(start) and cached.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7)
    if missing or stale or too_short or cached.empty:
        fetch = tickers if (stale or too_short or cached.empty) else missing
        fresh = _download(sorted(set(fetch)), start, end)
        cached = fresh.combine_first(cached) if not cached.empty else fresh
        PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(PRICE_CACHE)
    out = cached.loc[:, [t for t in tickers if t in cached.columns]]
    out = out.loc[pd.Timestamp(start):] if start else out
    return out.loc[:pd.Timestamp(end)] if end else out


def shares_outstanding(tickers: list[str]) -> pd.Series:
    """Current shares outstanding (used to approximate cap weights; see README assumptions)."""
    import yfinance as yf

    cache = RAW_DIR / "shares_outstanding.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)["shares"]
        if set(tickers) <= set(s.index):
            return s.loc[tickers]
    out = {}
    for t in tickers:
        try:
            out[t] = float(yf.Ticker(t).fast_info["shares"])
        except Exception:
            out[t] = float("nan")
    s = pd.Series(out, name="shares")
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cache)
    return s
