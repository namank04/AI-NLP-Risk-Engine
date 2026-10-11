import numpy as np
import pandas as pd
import pytest

from riskengine.modules.index.optimizer import BLConfig, black_litterman, optimise


@pytest.fixture
def setup():
    tick = ["A", "B", "C", "D"]
    vol = np.array([0.01, 0.015, 0.02, 0.012])
    corr = np.full((4, 4), 0.4) + 0.6 * np.eye(4)
    S = pd.DataFrame(np.outer(vol, vol) * corr, index=tick, columns=tick)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=tick)
    return S, w


def test_no_views_returns_benchmark(setup):
    S, w = setup
    cfg = BLConfig(max_weight=1.0)
    mu, pi = black_litterman(S, w, pd.Series(0.0, index=w.index), pd.Series(0.0, index=w.index), cfg)
    assert np.allclose(mu, pi)
    out = optimise(mu, S, w, cfg)
    assert np.allclose(out, w, atol=1e-3)


def test_positive_view_overweights_and_confidence_scales_it(setup):
    S, w = setup
    cfg = BLConfig(max_weight=1.0)
    x = pd.Series([0.0, 0.0, 0.8, 0.0], index=w.index)
    lo = optimise(*black_litterman(S, w, x, pd.Series([0, 0, 0.5, 0], index=w.index), cfg)[:1], S, w, cfg)
    hi = optimise(*black_litterman(S, w, x, pd.Series([0, 0, 20.0, 0], index=w.index), cfg)[:1], S, w, cfg)
    assert hi["C"] > lo["C"] > w["C"]
    assert hi.sum() == pytest.approx(1.0)


def test_caps_respected(setup):
    S, w = setup
    cfg = BLConfig(max_weight=0.35)
    x = pd.Series([1.0, 0.0, 0.0, 0.0], index=w.index)
    out = optimise(black_litterman(S, w, x, pd.Series([50.0, 0, 0, 0], index=w.index), cfg)[0], S, w, cfg)
    # cap is max(flat cap, benchmark + 5pp) so a mega-cap is never forced underweight
    upper = np.maximum(0.35, w + 0.05)
    assert (out <= upper + 1e-6).all()
    assert out["A"] > w["A"]


def test_live_rebalancer_black_litterman_tilts_with_te_budget():
    from datetime import UTC, datetime, timedelta

    from riskengine.modules.index.rebalancer import IndexRebalancer
    from riskengine.schema import EventClassification, Evidence, Impact, Sentiment, Signal

    tick = ["AAPL", "MSFT", "XOM", "KO"]
    vol = np.array([0.018, 0.016, 0.02, 0.01])
    corr = np.full((4, 4), 0.35) + 0.65 * np.eye(4)
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=tick, columns=tick)
    reb = IndexRebalancer(benchmark=pd.Series([0.4, 0.3, 0.2, 0.1], index=tick))
    reb.tickers, reb.cov = tick, cov
    reb.weights = reb.benchmark.copy()
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    for i in range(12):
        reb.on_signal(Signal(signal_id=f"s{i}", item_id=f"i{i}", cluster_id=f"c{i}", entity="XOM", entity_type="company",
                             source="t", source_type="news", published_at=t0 + timedelta(minutes=20 * i),
                             sentiment=Sentiment(score=0.9, confidence=0.9, label="positive"),
                             event=EventClassification(primary="earnings", labels=["earnings"], probs={}),
                             impact=Impact(score=8, raw=8.0, confidence=0.3, method="t"), novelty=1.0, n_sources=3,
                             evidence=Evidence(headline="h"), model_version="t"))
    last = reb.history[-1]
    assert last.method == "black_litterman" and last.reason == "band_exceeded"
    assert last.weights["XOM"] > 0.2 and last.ex_ante_te <= 0.02 + 1e-3
    assert last.alpha_bps["XOM"] > 0
