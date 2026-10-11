import numpy as np
import pandas as pd
import pytest

from riskengine.market.event_study import event_day, fit_market_model, scar


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2021-01-01", periods=600)
    rm = rng.normal(0.0004, 0.01, len(days))
    eps = rng.normal(0, 0.012, len(days))
    ri = 0.0001 + 1.5 * rm + eps
    jump_day = 500
    ri[jump_day] += 0.08                      # an 8% idiosyncratic shock
    px = pd.DataFrame({"SPY": 100 * np.cumprod(1 + rm), "XYZ": 50 * np.cumprod(1 + ri)}, index=days)
    return px, days[jump_day]


def test_market_model_recovers_beta(synthetic_prices):
    px, _ = synthetic_prices
    mm = fit_market_model(px)
    assert mm.beta["XYZ"].iloc[-1] == pytest.approx(1.5, abs=0.15)
    assert mm.sigma["XYZ"].iloc[-1] == pytest.approx(0.012, abs=0.002)


def test_scar_flags_the_shock_and_not_quiet_days(synthetic_prices):
    px, jump = synthetic_prices
    mm = fit_market_model(px)
    t0 = pd.Series([jump, px.index[400]])
    res = scar(mm, pd.Series(["XYZ", "XYZ"]), t0, k=1)
    assert res["car"].iloc[0] == pytest.approx(0.08, abs=0.03)
    assert res["scar"].iloc[0] > 3 and abs(res["scar"].iloc[1]) < 3


def test_event_day_rolls_weekends_forward():
    days = pd.bdate_range("2024-01-01", periods=10)
    d = pd.Series(pd.to_datetime(["2024-01-06", "2024-01-08"]).tz_localize("UTC"))   # Saturday, Monday
    out = event_day(d, days)
    assert list(out.dt.strftime("%a")) == ["Mon", "Mon"]
