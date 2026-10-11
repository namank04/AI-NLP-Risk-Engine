import numpy as np
import pytest

from riskengine.modules.stress.credit import (
    implied_rating,
    irb_correlation,
    irb_rwa,
    lifetime_pd,
    rating_pd,
    vasicek_conditional_pd,
)
from riskengine.modules.stress.engine import run_stress
from riskengine.modules.stress.portfolio import bond_risk, generate_portfolio
from riskengine.modules.stress.scenarios import idiosyncratic, severity_from_impact, systemic


def test_irb_risk_weight_matches_basel_reference():
    # Basel corporate curve: PD 1%, LGD 45%, M 2.5y -> risk weight ~ 92.3%
    rw = irb_rwa(np.array([0.01]), np.array([0.45]), np.array([1.0]), np.array([2.5]))[0]
    assert rw == pytest.approx(0.923, abs=0.01)


def test_irb_correlation_bounds():
    assert irb_correlation(np.array([1e-4]))[0] == pytest.approx(0.24, abs=0.01)
    assert irb_correlation(np.array([0.5]))[0] == pytest.approx(0.12, abs=0.001)


def test_vasicek_monotone_in_z_and_pd():
    pd_ = np.array([0.002, 0.02])
    rho = irb_correlation(pd_)
    lo, hi = vasicek_conditional_pd(pd_, rho, 0.5), vasicek_conditional_pd(pd_, rho, 2.0)
    assert (hi > lo).all() and hi[1] > hi[0]


def test_rating_round_trip_and_lifetime_pd():
    assert list(implied_rating(rating_pd(["AAA", "BBB", "CCC"]))) == ["AAA", "BBB", "CCC"]
    assert lifetime_pd(np.array([0.02]), np.array([5.0]))[0] == pytest.approx(1 - 0.98 ** 5)


def test_bond_duration_par_bond():
    # 5y 5% annual par bond: modified duration ~4.33
    md, cx = bond_risk(np.array([0.05]), np.array([0.05]), np.array([5.0]))
    assert md[0] == pytest.approx(4.33, abs=0.01) and cx[0] > 0


def test_portfolio_is_reproducible_and_mixed():
    a, _ = generate_portfolio(seed=1)
    b, _ = generate_portfolio(seed=1)
    assert a.equals(b)
    assert {"term_loan", "revolver", "corp_bond", "sovereign_bond", "irs", "fx_forward", "cds", "equity"} <= set(a.instrument)


@pytest.fixture(scope="module")
def pf():
    return generate_portfolio()[0]


def test_zero_severity_is_no_op(pf):
    r = run_stress(pf, systemic("macroeconomic", impact=0, severity=0.0))
    assert abs(r.summary["total_loss"]) < 1e-6 * pf["notional"].sum()
    assert r.summary["cet1_ratio_base"] == pytest.approx(0.135, abs=1e-4)


@pytest.mark.parametrize("cls", ["geopolitical", "macroeconomic", "credit_event", "operational"])
def test_systemic_scenarios_lose_money_and_capital(pf, cls):
    r = run_stress(pf, systemic(cls, impact=10))
    s = r.summary
    assert s["total_loss"] > 0
    assert s["cet1_ratio_stressed"] < s["cet1_ratio_base"]
    assert s["ecl_stressed"] > s["ecl_base"]
    total = sum(r.attribution.values())
    assert total == pytest.approx(s["total_loss"], rel=1e-6)


def test_severity_scales_losses(pf):
    mild = run_stress(pf, systemic("credit_event", impact=7)).summary["total_loss"]
    severe = run_stress(pf, systemic("credit_event", impact=10)).summary["total_loss"]
    assert severe > mild > 0
    assert severity_from_impact(7) == pytest.approx(0.5)


def test_scope_countries_add_loss(pf):
    base = run_stress(pf, systemic("geopolitical", impact=9)).summary["total_loss"]
    scoped = run_stress(pf, systemic("geopolitical", impact=9, scope_countries=["CN", "TW"])).summary["total_loss"]
    assert scoped > base


def test_idiosyncratic_default_hits_only_target_and_sector(pf):
    name = pf.loc[pf["instrument"] == "term_loan", "counterparty"].iloc[0]
    sector = pf.loc[pf["counterparty"] == name, "sector"].iloc[0]
    r = run_stress(pf, idiosyncratic(name, sector, impact=10, subtype="bankruptcy"))
    pos = r.positions
    hit = pos[pos["loss"].abs() > 1e-9]
    assert set(hit["sector"]) <= {sector}
    tgt = pos[(pos["counterparty"] == name) & (pos["instrument"] == "term_loan")]
    assert (tgt["pd_stressed"] == 1.0).all()


@pytest.mark.parametrize("cls,lo,hi", [("geopolitical", 1.5, 5.0), ("macroeconomic", 3.0, 7.0),
                                       ("credit_event", 3.5, 8.0), ("operational", 3.0, 8.0)])
def test_full_severity_cet1_drawdown_is_plausible(pf, cls, lo, hi):
    # instantaneous shocks, no pre-provision revenue offset; regulatory severely adverse
    # scenarios cost large banks roughly 2.5-4.5pp peak-to-trough including that offset
    s = run_stress(pf, systemic(cls, impact=10)).summary
    drawdown = 100 * (s["cet1_ratio_base"] - s["cet1_ratio_stressed"])
    assert lo <= drawdown <= hi, drawdown


def test_rate_risk_is_hedged(pf):
    from riskengine.modules.stress.engine import market_risk_capital

    hedges = pf[pf["position_id"].str.startswith("H-")]
    assert set(hedges["currency"]) == {"USD", "EUR", "GBP", "JPY"}
    assert market_risk_capital(pf)["rates"] > 0


def _signal(i, cls="geopolitical", impact=8, n_sources=5, minutes=0, countries=()):
    from datetime import UTC, datetime, timedelta

    from riskengine.schema import EventClassification, Evidence, Impact, Scope, Sentiment, Signal

    return Signal(signal_id=f"s{i}", item_id=f"i{i}", cluster_id=f"c{i}", entity="MARKET", entity_type="market",
                  source="gdelt", source_type="news",
                  published_at=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
                  sentiment=Sentiment(score=-0.8, confidence=0.9, label="negative"),
                  event=EventClassification(primary=cls, labels=[cls], probs={cls: 0.9}),
                  impact=Impact(score=impact, raw=float(impact), confidence=0.3, method="rule_v0"),
                  scope=Scope(countries=list(countries)), novelty=1.0, n_sources=n_sources,
                  evidence=Evidence(headline="h"), model_version="t")


def test_trigger_needs_impact_and_corroboration(pf):
    from riskengine.modules.stress.trigger import StressTrigger

    tr = StressTrigger(pf)
    assert tr.on_signal(_signal(1, impact=6)) is None
    assert tr.on_signal(_signal(2, n_sources=1)) is None
    assert tr.on_signal(_signal(3)) is not None


def test_trigger_reruns_only_on_escalation(pf):
    from riskengine.modules.stress.trigger import StressTrigger

    tr = StressTrigger(pf)
    assert tr.on_signal(_signal(1, impact=7)) is not None                    # first run
    assert tr.on_signal(_signal(2, impact=7, minutes=30)) is None            # same story re-worded
    assert tr.on_signal(_signal(3, impact=9, minutes=60)) is not None        # escalation
    assert tr.on_signal(_signal(4, impact=9, minutes=90, countries=["TW"])) is not None  # new country
    assert tr.on_signal(_signal(5, impact=7, minutes=90 + 60 * 6)) is not None   # cooldown over
    assert len(tr.runs) == 4


def test_routine_filing_does_not_trigger_but_material_one_does(pf):
    from riskengine.modules.stress.trigger import StressTrigger

    tr = StressTrigger(pf)
    s = _signal(1, cls="credit_event", n_sources=1).model_copy(update={
        "entity": "BA", "entity_type": "company", "source_type": "filing", "tags": ["8k:8.01", "8k:9.01"]})
    assert tr.on_signal(s) is None
    s2 = s.model_copy(update={"signal_id": "s2", "tags": ["8k:2.04"]})
    assert tr.on_signal(s2) is not None


def test_systemic_credit_needs_credit_evidence(pf):
    from riskengine.modules.stress.trigger import StressTrigger

    tr = StressTrigger(pf)
    vague = _signal(1, cls="credit_event", impact=8)
    assert tr.on_signal(vague) is None
    real = vague.model_copy(update={"signal_id": "s2", "event": vague.event.model_copy(update={"subtype": "liquidity_stress"})})
    assert tr.on_signal(real) is not None


def test_positive_news_never_triggers(pf):
    from riskengine.modules.stress.trigger import StressTrigger

    tr = StressTrigger(pf)
    up = _signal(1, cls="credit_event", impact=8)
    up = up.model_copy(update={"sentiment": up.sentiment.model_copy(update={"score": 0.7, "label": "positive"}),
                               "event": up.event.model_copy(update={"subtype": "rating_action"})})
    assert tr.on_signal(up) is None


def test_reverse_stress_curve_is_monotone_and_breach_consistent(pf):
    import numpy as np

    from riskengine.modules.stress.engine import severity_curve

    cur = severity_curve(pf, lambda s: systemic("credit_event", impact=0, severity=s), steps=np.round(np.arange(0, 1.55, 0.1), 2))
    ratios = [p["cet1_ratio"] for p in cur["points"]]
    assert all(b <= a + 1e-9 for a, b in zip(ratios, ratios[1:], strict=False))
    br = cur["breach_buffer"]
    assert br is not None and 0 < br["severity"] < 1.5
    assert br["impact_equivalent"] == pytest.approx(4 + 6 * br["severity"], abs=0.1)
    below = run_stress(pf, systemic("credit_event", impact=0, severity=min(1.5, br["severity"] + 0.05))).summary
    assert below["breaches_buffer"]
