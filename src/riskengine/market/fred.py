"""FRED macro series (no API key needed via the fredgraph CSV endpoint)."""

from __future__ import annotations

import io

import httpx
import pandas as pd

from riskengine.config import RAW_DIR

# Risk factors used to calibrate Module B scenarios and to measure macro impact.
SERIES = {
    "DGS2": "2Y Treasury yield (%)",
    "DGS5": "5Y Treasury yield (%)",
    "DGS10": "10Y Treasury yield (%)",
    "DGS30": "30Y Treasury yield (%)",
    "BAMLC0A0CM": "US IG corporate OAS (%)",
    "BAMLH0A0HYM2": "US HY corporate OAS (%)",
    "BAMLC0A1CAAA": "AAA OAS (%)",
    "BAMLC0A2CAA": "AA OAS (%)",
    "BAMLC0A3CA": "A OAS (%)",
    "BAMLC0A4CBBB": "BBB OAS (%)",
    "BAMLH0A1HYBB": "BB OAS (%)",
    "BAMLH0A2HYB": "B OAS (%)",
    "BAMLH0A3HYC": "CCC & lower OAS (%)",
    # Moody's seasoned yields: long history (back to 1980s), a credit-spread proxy
    # for episodes the ICE OAS series may not cover on FRED
    "DAAA": "Moody's Aaa corporate yield (%)",
    "DBAA": "Moody's Baa corporate yield (%)",
    "VIXCLS": "VIX",
    "DCOILBRENTEU": "Brent crude ($/bbl)",
    "DEXUSEU": "USD per EUR",
    "DTWEXBGS": "Broad USD index",
}


def fred_series(ids: list[str] | None = None, start: str = "2000-01-01", refresh: bool = False) -> pd.DataFrame:
    ids = ids or list(SERIES)
    cache = RAW_DIR / "fred.parquet"
    cached = pd.read_parquet(cache) if cache.exists() and not refresh else pd.DataFrame()
    missing = [i for i in ids if i not in cached.columns]
    if missing or refresh:
        frames = []
        for sid in (ids if refresh else missing):
            resp = httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid},
                             timeout=90, follow_redirects=True)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
            df.columns = ["date", sid]
            frames.append(df.set_index(pd.to_datetime(df.pop("date"))))
        fresh = pd.concat(frames, axis=1)
        cached = fresh.combine_first(cached) if not cached.empty else fresh
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.sort_index().to_parquet(cache)
    return cached.loc[pd.Timestamp(start):, [i for i in ids if i in cached.columns]]
