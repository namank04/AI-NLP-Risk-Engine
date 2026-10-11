"""Calibrate Module B scenarios to the market moves actually observed in each episode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import CONFIG_DIR, ROOT, universe  # noqa: E402
from riskengine.market.fred import fred_series  # noqa: E402
from riskengine.market.prices import daily_close  # noqa: E402

# spread change of each rating bucket relative to BBB in stress (stylised from
# historical OAS behaviour: lower ratings widen proportionally more)
RATING_MULT = {"AAA": 0.35, "AA": 0.45, "A": 0.65, "BBB": 1.0, "BB": 2.2, "B": 3.3, "CCC": 6.5}
FX_TICKERS = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "CNY": "CNY=X", "INR": "INR=X",
              "BRL": "BRL=X", "MXN": "MXN=X", "TWD": "TWD=X"}
INVERTED = {"JPY", "CNY", "INR", "BRL", "MXN", "TWD"}     # quoted as units per USD


def window_change(s: pd.Series, start: str, end: str) -> float:
    w = s.loc[start:end].dropna()
    return float(w.iloc[-1] - w.iloc[0]) if len(w) > 1 else float("nan")


def peak_to_trough(s: pd.Series, start: str, end: str) -> float:
    """Largest move from the window start, signed: drawdown if the asset fell, rally otherwise."""
    w = s.loc[start:end].dropna()
    if len(w) < 2:
        return float("nan")
    rel = w / w.iloc[0] - 1
    lo, hi = rel.min(), rel.max()
    return float(lo if abs(lo) >= abs(hi) else hi)


def main() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "scenarios.yaml").read_text(encoding="utf-8"))
    fred = fred_series(start="2007-01-01")
    u = universe()
    etfs = dict(u.sector_etfs)
    px = daily_close([u.market_proxy, *etfs.values(), *FX_TICKERS.values()], start="2007-01-01")
    report = ["# Scenario calibration", "",
              "Market-factor shocks measured over each anchor episode (template value → measured).", ""]
    for tpl in cfg["systemic"].values():
        s, e = tpl["episode"]["start"], tpl["episode"]["end"]
        cal: dict = {}
        cal["rates_bp"] = {
            "USD": round(100 * window_change(fred["DGS10"], s, e), 0),
            "USD_2Y": round(100 * window_change(fred["DGS2"], s, e), 0),
        }
        baa = (fred["DBAA"] - fred["DGS10"]) * 100
        bbb_bp = window_change(baa, s, e)
        if "BAMLC0A4CBBB" in fred and fred["BAMLC0A4CBBB"].loc[s:e].notna().sum() > 5:
            bbb_bp = 100 * window_change(fred["BAMLC0A4CBBB"], s, e)           # prefer ICE OAS when available
        cal["credit_spread_bp"] = {r: round(max(0.0, bbb_bp) * m, 0) for r, m in RATING_MULT.items()}
        eq = {"_all": round(100 * peak_to_trough(px[u.market_proxy], s, e), 1)}
        for sector, etf in etfs.items():
            v = peak_to_trough(px[etf], s, e) if etf in px else float("nan")
            if np.isfinite(v):
                eq[sector] = round(100 * v, 1)
        cal["equity_pct"] = eq
        fx = {}
        for ccy, t in FX_TICKERS.items():
            if t in px and px[t].loc[s:e].notna().sum() > 5:
                w = px[t].loc[s:e].dropna()
                ch = float(w.iloc[-1] / w.iloc[0] - 1)          # start-to-end: brief spikes are not the shock
                fx[ccy] = round(100 * (-ch / (1 + ch) if ccy in INVERTED else ch), 1)
        cal["fx_pct"] = fx
        brent = fred["DCOILBRENTEU"].loc[s:e].dropna()
        cal["commodity_pct"] = {"crude_oil": round(100 * float(brent.iloc[-1] / brent.iloc[0] - 1), 1)}
        cal["vix_change"] = round(window_change(fred["VIXCLS"], s, e), 1)
        tpl["calibrated"] = cal

        report += [f"## {tpl['name']} ({s} to {e})", "", "| factor | template | measured |", "|---|---|---|",
                   f"| 10Y UST (bp) | {tpl['rates_bp'].get('USD')} | {cal['rates_bp']['USD']} |",
                   f"| 2Y UST (bp) | — | {cal['rates_bp']['USD_2Y']} |",
                   f"| BBB spread (bp) | {tpl['credit_spread_bp']['BBB']} | {cal['credit_spread_bp']['BBB']} |",
                   f"| HY (B) spread (bp) | {tpl['credit_spread_bp']['B']} | {cal['credit_spread_bp']['B']} |",
                   f"| S&P 500 (%) | {tpl['equity_pct']['_all']} | {cal['equity_pct']['_all']} |",
                   f"| Financials (%) | {tpl['equity_pct'].get('Financials', '—')} | {cal['equity_pct'].get('Financials', '—')} |",
                   f"| Energy (%) | {tpl['equity_pct'].get('Energy', '—')} | {cal['equity_pct'].get('Energy', '—')} |",
                   f"| EUR vs USD (%) | {tpl['fx_pct'].get('EUR')} | {cal['fx_pct'].get('EUR', '—')} |",
                   f"| BRL vs USD (%) | {tpl['fx_pct'].get('BRL')} | {cal['fx_pct'].get('BRL', '—')} |",
                   f"| Brent (%) | {tpl['commodity_pct']['crude_oil']} | {cal['commodity_pct']['crude_oil']} |",
                   f"| VIX (points) | — | {cal['vix_change']} |", ""]
    # keep the hand-written header and trigger block; re-dump the scenario sections
    head = (CONFIG_DIR / "scenarios.yaml").read_text(encoding="utf-8").split("\nsystemic:")[0]
    body = yaml.safe_dump({"systemic": cfg["systemic"], "idiosyncratic": cfg["idiosyncratic"]},
                          sort_keys=False, width=140)
    (CONFIG_DIR / "scenarios.yaml").write_text(head + "\n" + body, encoding="utf-8")
    (ROOT / "docs" / "scenario_calibration.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
