"""Train and evaluate the learned impact model on event-study labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, ROOT, universe  # noqa: E402
from riskengine.gold import gold_item_ids  # noqa: E402
from riskengine.market.event_study import event_day, fit_market_model, scar  # noqa: E402
from riskengine.market.prices import daily_close  # noqa: E402
from riskengine.nlp.impact_model import (  # noqa: E402
    EVENT_CLASSES,
    META_FILE,
    MODEL_DIR,
    MODEL_FILE,
    load_context,
)

SPLIT = pd.Timestamp("2023-07-01")
VALID = pd.Timestamp("2023-04-01")


def build_dataset(name: str = "fnspid_2022_2023") -> pd.DataFrame:
    sig = pd.read_parquet(PROCESSED_DIR / f"signals_{name}.parquet")
    # gold items are evaluation-only; use the current pool, not the flag frozen at batch time
    sig = sig[(sig["entity_type"] == "company") & ~sig["item_id"].isin(gold_item_ids())].copy()
    sig["published_at"] = pd.to_datetime(sig["published_at"], utc=True)
    sig = sig.sort_values("published_at").reset_index(drop=True)
    u = universe()
    px = daily_close(u.tickers + [u.market_proxy], start="2020-01-01")
    mm = fit_market_model(px[u.tickers + [u.market_proxy]])
    days = mm.ar.index
    sig["t0"] = event_day(sig["published_at"], days)
    sig = sig.dropna(subset=["t0"])
    lab = scar(mm, sig["entity"], sig["t0"], k=1)
    sig = sig.join(lab)
    sig = sig.dropna(subset=["scar"])
    # stories on this name so far that day (causal: counts clusters first seen up to this signal)
    first = sig.drop_duplicates(["entity", "cluster_id"])[["entity", "cluster_id", "t0", "published_at"]]
    first = first.assign(k=first.groupby(["entity", "t0"]).cumcount() + 1)
    sig = sig.merge(first[["entity", "cluster_id", "k"]], on=["entity", "cluster_id"], how="left")
    sig = sig.rename(columns={"k": "stories_today"})
    ctx = load_context()
    rows = [ctx.asof(t, w) for t, w in zip(sig["entity"], sig["published_at"], strict=True)]
    sig = pd.concat([sig, pd.DataFrame(rows, index=sig.index)], axis=1)
    sig["abs_sentiment"] = sig["sentiment"].abs()
    sig["log_n_sources"] = np.log1p(sig["n_sources"])
    for c in EVENT_CLASSES:
        if f"p_{c}" not in sig:
            sig[f"p_{c}"] = 0.0
    sig["y"] = np.log1p(sig["scar"].abs())
    sig["w"] = 1.0 / sig.groupby(["entity", "cluster_id"])["signal_id"].transform("count")
    return sig


def decile_table(pred: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(pred.rank(method="first"), 10, labels=list(range(1, 11)))
    g = df.groupby(q, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "median_abs_car_pct": g["car"].apply(lambda s: 100 * s.abs().median()),
        "mean_abs_scar": g["scar"].apply(lambda s: s.abs().mean()),
        "share_2sigma": g["scar"].apply(lambda s: (s.abs() > 2).mean()),
    }).round(4)


DAY_CONTEXT = ["vol20", "beta", "log_mcap", "vix"]


def to_stock_days(df: pd.DataFrame) -> pd.DataFrame:
    """A stock reacts once per day, not once per article: aggregate that day's news."""
    df = df.assign(abs_s=df["sentiment"].abs())
    agg = {"n": ("signal_id", "count"), "rule_max": ("impact_rule_raw", "max"), "rule_mean": ("impact_rule_raw", "mean"),
           "abs_s_max": ("abs_s", "max"), "sent_mean": ("sentiment", "mean"), "sent_min": ("sentiment", "min"),
           "sw_max": ("source_weight", "max"), "scar": ("scar", "first"), "car": ("car", "first")}
    agg |= {c: (c, "first") for c in DAY_CONTEXT}
    agg |= {f"p_{c}_max": (f"p_{c}", "max") for c in EVENT_CLASSES}
    day = df.groupby(["entity", "t0"]).agg(**agg).reset_index()
    day["log_n"] = np.log1p(day["n"])
    day["y"] = np.log1p(day["scar"].abs())
    return day


def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 7) -> dict:
    """Bootstrap distribution of Spearman(a, y) - Spearman(b, y) over resampled stock-days."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(spearmanr(a[i], y[i]).statistic - spearmanr(b[i], y[i]).statistic)
    d = np.array(diffs)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_learned_not_better": float((d <= 0).mean())}


def main() -> None:
    import lightgbm as lgb

    sig = build_dataset()
    day = to_stock_days(sig)
    feats = [c for c in day.columns if c not in ("entity", "t0", "scar", "car", "y", "n")]
    t0 = pd.to_datetime(day["t0"])
    train, valid, test = day[t0 < VALID], day[(t0 >= VALID) & (t0 < SPLIT)], day[t0 >= SPLIT]
    print(f"stock-days: train {len(train):,}  valid {len(valid):,}  test {len(test):,}")
    params = dict(objective="huber", learning_rate=0.03, num_leaves=7, min_data_in_leaf=60, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
    b = lgb.train(params, lgb.Dataset(train[feats], train["y"]), 3000,
                  valid_sets=[lgb.Dataset(valid[feats], valid["y"])], callbacks=[lgb.early_stopping(150, verbose=False)])
    tv = pd.concat([train, valid])
    booster = lgb.train(params, lgb.Dataset(tv[feats], tv["y"]), max(50, b.best_iteration))

    y_abs = test["scar"].abs().to_numpy()
    pred = booster.predict(test[feats])
    rule = test["rule_max"].to_numpy()

    def rho(x):
        return float(spearmanr(x, y_abs).statistic)

    boot = paired_bootstrap(pred, rule, y_abs)
    adopt = boot["ci95"][0] > 0
    sig_test = sig[pd.to_datetime(sig["t0"]) >= SPLIT]
    results = {
        "unit": "stock-day", "n_train": int(len(tv)), "n_test": int(len(test)), "test_start": str(SPLIT.date()),
        "spearman": {"learned": rho(pred), "rule_v0": rho(rule), "news_count": rho(test["log_n"]),
                     "max_abs_sentiment": rho(test["abs_s_max"]), "p_earnings": rho(test["p_earnings_max"])},
        "spearman_se": float(1 / np.sqrt(len(test))),
        "bootstrap_learned_minus_rule": boot, "adopted": "learned" if adopt else "rule_v0",
        "sentiment_direction": {
            "spearman_sentiment_vs_signed_scar": float(spearmanr(test["sent_mean"], test["scar"]).statistic),
            "hit_rate_strong": float(((np.sign(sig_test["sentiment"]) == np.sign(sig_test["scar"]))
                                      [sig_test["sentiment"].abs() > 0.5]).mean()),
        },
        "deciles_learned": decile_table(pd.Series(pred, index=test.index), test).reset_index(names="decile").to_dict("records"),
        "deciles_rule_v0": decile_table(test["rule_max"], test).reset_index(names="decile").to_dict("records"),
        "feature_importance": dict(sorted(zip(feats, booster.feature_importance("gain").round(1).tolist(), strict=True),
                                          key=lambda kv: -kv[1])),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    booster.save_model(str(MODEL_DIR / "impact_day_lgbm.txt"))
    (MODEL_DIR / "impact_day_lgbm.json").write_text(json.dumps(
        {"features": feats, "label": "log1p(|SCAR[0,+1]|) per stock-day", "version": "lgbm_event_study_day_v1",
         "adopted": results["adopted"], "metrics": results["spearman"]}, indent=2), encoding="utf-8")
    for stale in (MODEL_FILE, META_FILE):            # the earlier signal-level model is superseded
        stale.unlink(missing_ok=True)
    (PROCESSED_DIR / "impact_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# Impact score validation: event study", "",
             "Does the engine's impact score anticipate how strongly a stock actually reacts?", "",
             f"- Unit: stock-day (a stock reacts once per day, however many articles). Train 2022-01 to 2023-06, "
             f"test {results['n_test']:,} stock-days from {results['test_start']}, never seen in training.",
             "- Label: |SCAR[0,+1]|, the 2-day abnormal return vs a SPY market model (220-day estimation window ending",
             "  30 trading days before the event), in units of the stock's own residual volatility.", "",
             "## Rank correlation with the realised reaction (test set)", "",
             f"Standard error of each ρ ≈ {results['spearman_se']:.3f}.", "",
             "| predictor | Spearman ρ |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in results["spearman"].items()]
    lines += ["", f"Paired bootstrap, learned minus rule: mean {boot['mean_diff']:+.3f}, 95% CI "
                  f"[{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}] → production uses **{results['adopted']}**.", "",
              "## Realised reaction by predicted decile (learned, test set)", "",
              pd.DataFrame(results["deciles_learned"]).to_markdown(index=False), "",
              "## Same for the rule baseline (production impact score)", "",
              pd.DataFrame(results["deciles_rule_v0"]).to_markdown(index=False), "",
              "## Does sentiment predict direction?", "",
              f"- Spearman(day mean sentiment, signed SCAR): {results['sentiment_direction']['spearman_sentiment_vs_signed_scar']:.3f}",
              f"- per-signal sign hit rate when |sentiment| > 0.5: {results['sentiment_direction']['hit_rate_strong']:.3f}", "",
              "## Feature importance (gain)", ""]
    lines += [f"- {k}: {v}" for k, v in list(results["feature_importance"].items())[:12]]
    (ROOT / "docs" / "impact_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
