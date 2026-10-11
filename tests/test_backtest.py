import numpy as np
import pandas as pd

from riskengine.modules.index.backtest import BacktestConfig, run_backtest


def test_backtest_rewards_a_genuinely_predictive_signal():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2021-01-04", "2022-06-30")
    tick = [f"T{i}" for i in range(8)]
    rm = rng.normal(0.0003, 0.01, len(days))
    rets = {t: 0.8 * rm + rng.normal(0, 0.015, len(days)) for t in tick}
    rows = []
    start = days.searchsorted(pd.Timestamp("2022-01-03"))
    for k in range(start + 2, len(days)):
        # news on day k-1 (date-stamped) predicts the stock's return on day k+1
        for t in rng.choice(tick, 3, replace=False):
            s = rng.uniform(-1, 1)
            rets[t][k + 1 if k + 1 < len(days) else k] += 0.01 * s
            rows.append({"signal_id": f"{t}{k}", "item_id": f"{t}{k}", "cluster_id": f"{t}{k}", "entity": t,
                         "source": "x", "published_at": pd.Timestamp(days[k - 1], tz="UTC"), "sentiment": s,
                         "sentiment_conf": 0.9, "event_primary": "earnings", "impact_rule": 6, "novelty": 1.0,
                         "n_sources": 2, "source_weight": 1.0})
    px = pd.DataFrame({t: 100 * np.cumprod(1 + r) for t, r in rets.items()}, index=days)
    px["SPY"] = 100 * np.cumprod(1 + rm)
    shares = pd.Series(1.0, index=tick)
    cfg = BacktestConfig(start="2022-01-03", end="2022-06-30", cov_lookback=120)
    res = run_backtest(pd.DataFrame(rows), px, shares, cfg=cfg)
    assert res["ic"]["engine"]["mean_ic"] > 0.05
    assert res["metrics"]["engine_tilt"]["active_return"] > 0
    assert set(res["metrics"]) == {"cap_weight", "equal_weight", "naive_tilt", "engine_tilt", "black_litterman"}
