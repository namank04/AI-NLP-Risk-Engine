"""Synthetic wholesale banking book (seeded, reproducible)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.config import SYNTHETIC_DIR, universe
from riskengine.modules.stress.credit import RATINGS, rating_pd

SECTORS = ["Information Technology", "Communication Services", "Consumer Discretionary", "Financials", "Energy",
           "Health Care", "Industrials", "Consumer Staples", "Utilities", "Materials", "Real Estate"]
# country -> (weight, local currency, sovereign rating, sovereign 10y yield %)
COUNTRIES = {
    "US": (0.50, "USD", "AA", 4.2), "EU": (0.14, "EUR", "AA", 2.6), "GB": (0.07, "GBP", "AA", 4.3),
    "JP": (0.05, "JPY", "A", 1.2), "CN": (0.06, "CNY", "A", 2.0), "IN": (0.06, "INR", "BBB", 6.8),
    "BR": (0.04, "BRL", "BB", 12.0), "MX": (0.03, "MXN", "BBB", 9.5), "SA": (0.03, "USD", "A", 5.0),
    "TW": (0.02, "TWD", "AA", 1.6),
}
EM = {"CN", "IN", "BR", "MX", "SA", "TW"}
RATING_WEIGHTS = {"AAA": 0.02, "AA": 0.06, "A": 0.18, "BBB": 0.34, "BB": 0.22, "B": 0.14, "CCC": 0.04}
SPREAD_BPS = {"AAA": 40, "AA": 55, "A": 80, "BBB": 130, "BB": 250, "B": 400, "CCC": 900}
SECTOR_LGD = {"Financials": 0.45, "Real Estate": 0.35, "Utilities": 0.35, "Energy": 0.40}
INSTRUMENT_MIX = {"term_loan": 0.34, "revolver": 0.15, "corp_bond": 0.21, "sovereign_bond": 0.08, "irs": 0.08,
                  "fx_forward": 0.06, "cds": 0.04, "equity": 0.04}

_PREFIX = ["Northwind", "Bluecrest", "Ironbridge", "Silverline", "Redwood", "Harbor", "Summit", "Cobalt",
           "Granite", "Meridian", "Aurora", "Pioneer Ridge", "Falcon", "Evergreen", "Crescent", "Atlas",
           "Sterling", "Orion", "Vertex", "Keystone", "Lakeshore", "Highgate", "Copperleaf", "Sable"]
_SUFFIX = {
    "Information Technology": "Systems", "Communication Services": "Media", "Consumer Discretionary": "Retail",
    "Financials": "Capital", "Energy": "Energy", "Health Care": "Health", "Industrials": "Industries",
    "Consumer Staples": "Foods", "Utilities": "Power", "Materials": "Materials", "Real Estate": "Properties",
}


def bond_risk(coupon: np.ndarray, ytm: np.ndarray, maturity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration and convexity of annual-coupon bullet bonds (per unit price)."""
    md, cx = np.zeros(len(coupon)), np.zeros(len(coupon))
    for i, (c, y, m) in enumerate(zip(coupon, ytm, maturity, strict=True)):
        t = np.arange(1, int(np.ceil(m)) + 1, dtype=float)
        t[-1] = m
        cf = np.full(len(t), c)
        cf[-1] += 1.0
        df = (1 + y) ** -t
        price = (cf * df).sum()
        mac = (t * cf * df).sum() / price
        md[i] = mac / (1 + y)
        cx[i] = (cf * df * t * (t + 1)).sum() / (price * (1 + y) ** 2)
    return md, cx


def _counterparties(rng: np.random.Generator, n_fictitious: int) -> pd.DataFrame:
    rows = []
    for c in universe().companies:
        # large listed names: skew towards investment grade (synthetic internal grades)
        r = rng.choice(RATINGS[:5], p=[0.05, 0.20, 0.40, 0.30, 0.05])
        rows.append({"counterparty": c.name, "ticker": c.ticker, "sector": c.sector, "country": "US",
                     "rating": r, "listed_in_universe": True})
    names = set()
    while len(rows) < 20 + n_fictitious:
        sector = rng.choice(SECTORS)
        name = f"{rng.choice(_PREFIX)} {_SUFFIX[sector]}"
        if name in names:
            continue
        names.add(name)
        country = rng.choice(list(COUNTRIES), p=[v[0] for v in COUNTRIES.values()])
        rating = rng.choice(list(RATING_WEIGHTS), p=list(RATING_WEIGHTS.values()))
        rows.append({"counterparty": name, "ticker": None, "sector": sector, "country": country,
                     "rating": rating, "listed_in_universe": False})
    df = pd.DataFrame(rows)
    df["pd_1y"] = rating_pd(df["rating"])
    return df


def generate_portfolio(n_positions: int = 400, n_fictitious: int = 60, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cps = _counterparties(rng, n_fictitious)
    corp = cps.reset_index(drop=True)
    kinds = rng.choice(list(INSTRUMENT_MIX), size=n_positions, p=list(INSTRUMENT_MIX.values()))
    rows = []
    for i, kind in enumerate(kinds):
        cp = corp.iloc[rng.integers(len(corp))]
        country = cp["country"]
        ccy = COUNTRIES[country][1] if (country not in EM or rng.random() < 0.4) else "USD"
        base = {"position_id": f"P{i:04d}", "instrument": kind, "counterparty": cp["counterparty"],
                "ticker": cp["ticker"], "sector": cp["sector"], "country": country, "currency": ccy,
                "rating_orig": cp["rating"], "rating": cp["rating"], "direction": 1}
        seniority = rng.choice(["senior_secured", "senior_unsecured", "subordinated"], p=[0.45, 0.45, 0.10])
        lgd = {"senior_secured": 0.25, "senior_unsecured": 0.45, "subordinated": 0.75}[seniority]
        lgd = min(0.9, lgd * (SECTOR_LGD.get(cp["sector"], 0.40) / 0.40))
        mat = float(np.round(rng.uniform(1, 7), 2))
        if kind == "term_loan":
            amt = float(rng.uniform(20, 300))
            base |= {"book": "banking", "notional": amt, "drawn": amt, "undrawn": 0.0, "maturity": mat,
                     "seniority": seniority, "lgd": lgd, "spread_bps": SPREAD_BPS[cp["rating"]] * 0.8}
        elif kind == "revolver":
            commit = float(rng.uniform(50, 500))
            drawn = commit * float(rng.uniform(0.1, 0.6))
            base |= {"book": "banking", "notional": commit, "drawn": drawn, "undrawn": commit - drawn,
                     "maturity": min(mat, 5.0), "seniority": "senior_unsecured", "lgd": 0.45,
                     "spread_bps": SPREAD_BPS[cp["rating"]] * 0.7}
        elif kind == "corp_bond":
            base |= {"book": "trading", "notional": float(rng.uniform(10, 150)), "maturity": float(np.round(rng.uniform(2, 15), 2)),
                     "seniority": seniority, "lgd": lgd, "spread_bps": float(SPREAD_BPS[cp["rating"]])}
        elif kind == "sovereign_bond":
            sov = rng.choice(list(COUNTRIES), p=[v[0] for v in COUNTRIES.values()])
            _, sccy, srating, _ = COUNTRIES[sov]
            base |= {"book": "trading", "counterparty": f"Government of {sov}", "ticker": None, "sector": "Sovereign",
                     "country": sov, "currency": sccy, "rating_orig": srating, "rating": srating,
                     "notional": float(rng.uniform(100, 1000)), "maturity": float(np.round(rng.uniform(1, 30), 2)),
                     "seniority": "senior_unsecured", "lgd": 0.40,
                     "spread_bps": 0.0 if sov in ("US", "EU", "GB", "JP") else float(SPREAD_BPS[srating] * 0.8)}
        elif kind == "irs":
            # client-flow swaps (the ALM hedge book is added after durations are known)
            base |= {"book": "trading", "notional": float(rng.uniform(50, 300)), "maturity": float(np.round(rng.uniform(2, 20), 2)),
                     "direction": int(rng.choice([1, -1])),       # +1 pay fixed (gains when rates rise)
                     "lgd": 0.45, "spread_bps": 0.0}
        elif kind == "fx_forward":
            fccy = ccy if ccy != "USD" else rng.choice(["EUR", "GBP", "JPY", "CNY", "INR", "BRL", "MXN"])
            base |= {"book": "trading", "currency": fccy, "notional": float(rng.uniform(50, 500)),
                     "maturity": float(np.round(rng.uniform(0.25, 2), 2)),
                     "direction": int(rng.choice([1, -1])),       # +1 long foreign currency vs USD
                     "lgd": 0.45, "spread_bps": 0.0}
        elif kind == "cds":
            base |= {"book": "trading", "notional": float(rng.uniform(25, 200)), "maturity": 5.0,
                     "direction": int(rng.choice([1, -1], p=[0.6, 0.4])),  # +1 protection bought
                     "lgd": 0.60, "spread_bps": float(SPREAD_BPS[cp["rating"]])}
        else:  # equity: positions only in the 20 universe names
            u = cps[cps["listed_in_universe"]].iloc[rng.integers(20)]
            base |= {"book": "trading", "counterparty": u["counterparty"], "ticker": u["ticker"], "sector": u["sector"],
                     "country": "US", "currency": "USD", "rating_orig": u["rating"], "rating": u["rating"],
                     "notional": float(rng.uniform(10, 80)), "maturity": 0.0, "lgd": 0.0, "spread_bps": 0.0}
        rows.append(base)

    df = pd.DataFrame(rows)
    for col, default in [("drawn", np.nan), ("undrawn", 0.0), ("seniority", None)]:
        if col not in df:
            df[col] = default
    df["pd_1y"] = rating_pd(df["rating"])
    # base curve = the bond's currency curve; local sovereign yields already embed
    # sovereign credit risk, so only corporates add a credit spread on top
    ccy_yield = {v[1]: v[3] for v in COUNTRIES.values() if v[1] != "USD"} | {"USD": COUNTRIES["US"][3]}
    base_yield = df["currency"].map(ccy_yield).fillna(COUNTRIES["US"][3]) / 100
    corp_spread = np.where(df["instrument"] == "sovereign_bond", 0.0, df["spread_bps"].fillna(0) / 1e4)
    ytm = base_yield + corp_spread
    coupon = np.round(ytm * 2, 2) / 2                       # priced near par
    is_rate = df["instrument"].isin(["corp_bond", "sovereign_bond", "irs"])
    md, cx = bond_risk(coupon[is_rate].to_numpy(), ytm[is_rate].to_numpy(), df.loc[is_rate, "maturity"].to_numpy())
    df["coupon"] = np.where(is_rate, coupon, np.nan)
    df["mod_duration"], df["convexity"] = 0.0, 0.0
    df.loc[is_rate, "mod_duration"], df.loc[is_rate, "convexity"] = md, cx
    df["market_value"] = np.select(
        [df["instrument"].isin(["corp_bond", "sovereign_bond", "equity"]),
         df["instrument"].isin(["term_loan", "revolver"])],
        [df["notional"], df["drawn"]], default=0.0)          # derivatives start at zero MTM (at-market)
    df = pd.concat([df, _alm_hedges(df, rng, ccy_yield)], ignore_index=True)
    df["dv01"] = np.where(df["instrument"] == "irs", df["notional"] * df["mod_duration"] * 1e-4 * df["direction"], 0.0)
    df["cs01"] = np.where(df["instrument"] == "cds", df["notional"] * 4.3 * 1e-4 * df["direction"], 0.0)
    df[["notional", "drawn", "undrawn", "market_value"]] = df[["notional", "drawn", "undrawn", "market_value"]].round(2)
    return df, cps


HEDGED_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]     # deep swap markets; EM rate risk stays open


def _alm_hedges(df: pd.DataFrame, rng: np.random.Generator, ccy_yield: dict[str, float]) -> pd.DataFrame:
    """Pay-fixed swaps hedging 60-85% of the fixed-rate bond book's DV01 per currency.

    Banks manage interest-rate risk in the banking book (IRRBB) this way; without it
    a +150 bp shock would wipe out a realistic bank's capital on its own.
    """
    rows = []
    bonds = df[df["instrument"].isin(["corp_bond", "sovereign_bond"])]
    for ccy in HEDGED_CURRENCIES:
        b = bonds[bonds["currency"] == ccy]
        bond_dv01 = float((b["market_value"] * b["mod_duration"]).sum() * 1e-4)
        if bond_dv01 <= 0:
            continue
        ratio = float(rng.uniform(0.60, 0.85))
        for tenor, share in ((5.0, 0.4), (10.0, 0.6)):
            y = ccy_yield.get(ccy, 4.0) / 100
            md, cx = bond_risk(np.array([y]), np.array([y]), np.array([tenor]))
            notional = ratio * share * bond_dv01 / (md[0] * 1e-4)
            rows.append({"position_id": f"H-{ccy}-{int(tenor)}Y", "instrument": "irs", "book": "trading",
                         "counterparty": "ALM hedge (cleared)", "ticker": None, "sector": "Treasury/ALM",
                         "country": {"USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP"}[ccy], "currency": ccy,
                         "rating_orig": "AA", "rating": "AA", "pd_1y": rating_pd(["AA"])[0], "direction": 1,
                         "notional": round(notional, 2), "drawn": np.nan, "undrawn": 0.0, "maturity": tenor,
                         "seniority": None, "lgd": 0.45, "spread_bps": 0.0, "coupon": y, "mod_duration": md[0],
                         "convexity": cx[0], "market_value": 0.0, "hedge_ratio": round(ratio, 3)})
    return pd.DataFrame(rows)


def load_portfolio(regenerate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    p, c = SYNTHETIC_DIR / "portfolio.csv", SYNTHETIC_DIR / "counterparties.csv"
    if regenerate or not p.exists():
        df, cps = generate_portfolio()
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        cps.to_csv(c, index=False)
    return pd.read_csv(p), pd.read_csv(c)
