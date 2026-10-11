"""Run the Module A walk-forward backtest and write results.

Output: data/processed/backtest.json (served at /api/index/backtest), docs/backtest.md

Usage: python scripts/run_backtest.py [--signals fnspid_2022_2023] [--impact impact_rule]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, ROOT, universe  # noqa: E402
from riskengine.market.prices import daily_close, shares_outstanding  # noqa: E402
from riskengine.modules.index.backtest import BacktestConfig, run_backtest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="fnspid_2022_2023")
    ap.add_argument("--impact", default="impact_rule", help="column used as impact weight (impact_rule or impact_learned)")
    args = ap.parse_args()
    u = universe()
    sig = pd.read_parquet(PROCESSED_DIR / f"signals_{args.signals}.parquet")
    sig = sig[sig["entity_type"] == "company"]
    px = daily_close(u.tickers + [u.market_proxy], start="2020-01-01")
    res = run_backtest(sig, px, shares_outstanding(u.tickers), cfg=BacktestConfig(), impact_col=args.impact)
    (PROCESSED_DIR / "backtest.json").write_text(json.dumps(res), encoding="utf-8")

    m = pd.DataFrame(res["metrics"]).T
    lines = ["# Module A backtest (walk-forward, 2022-01 to 2023-12)", "",
             "Portfolio formed at the close of D from signals dated D-1 or earlier; earns D+1. "
             f"Transaction costs {res['config']['cost_bps']} bp per unit of one-way turnover; "
             f"no-trade band {res['config']['band']:.0%}.", "",
             m.to_markdown(), "",
             "## Signal quality: daily cross-sectional IC vs next-day abnormal return", "",
             pd.DataFrame(res["ic"]).T.to_markdown(), ""]
    (ROOT / "docs" / "backtest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
