"""Stage 0 data audit: how much timestamped, ticker-tagged text do we have?

Usage: python scripts/data_audit.py [--skip-tweets]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, ROOT, universe  # noqa: E402
from riskengine.ingest.historical import download_fnspid_headlines, download_stock_tweets  # noqa: E402


def coverage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    d = df.assign(year=df["published_at"].dt.year)
    t = d.pivot_table(index="ticker", columns="year", values="published_at", aggfunc="count", fill_value=0)
    t.insert(0, "source", source)
    return t.reindex(universe().tickers).fillna(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tweets", action="store_true")
    args = ap.parse_args()

    print("FNSPID headlines (streaming ~5.7 GB, keeping universe tickers only)...")
    news = download_fnspid_headlines()
    print(f"  {len(news):,} headlines, {news['published_at'].min()} .. {news['published_at'].max()}")
    tables = [coverage(news, "fnspid_headlines")]

    tweets = None
    if not args.skip_tweets:
        print("Stock tweets (~300 MB)...")
        tweets = download_stock_tweets()
        print(f"  {len(tweets):,} tweets, {tweets['published_at'].min()} .. {tweets['published_at'].max()}")
        tables.append(coverage(tweets, "stock_tweets"))

    cov = pd.concat(tables)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cov.to_csv(PROCESSED_DIR / "coverage_by_year.csv")

    # intraday timestamps matter for event timing (look-ahead): share with a non-midnight time
    def intraday_share(df):
        ts = df["published_at"]
        return float(((ts.dt.hour != 0) | (ts.dt.minute != 0)).mean())

    lines = ["# Data audit", "", "Items per ticker per year (universe only).", ""]
    for src, t in cov.groupby("source"):
        t = t.drop(columns="source")
        t = t.loc[:, t.sum() > 0].astype(int)
        lines += [f"## {src}", "", t.to_markdown(), ""]
    lines += ["## Timestamp precision", "",
              f"- fnspid_headlines: {intraday_share(news):.1%} of items carry an intraday time"]
    if tweets is not None:
        lines.append(f"- stock_tweets: {intraday_share(tweets):.1%} of items carry an intraday time")
    news_pub = news["Publisher"].fillna("unknown").value_counts().head(15)
    lines += ["", "## Top FNSPID publishers", "", news_pub.to_frame("items").to_markdown(), ""]
    out = ROOT / "docs" / "data_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(cov.to_string())


if __name__ == "__main__":
    main()
