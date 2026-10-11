from riskengine.nlp.impact import rule_impact
from riskengine.schema import EventClassification, Sentiment


def ev(primary, labels=None):
    return EventClassification(primary=primary, labels=labels or [primary], probs={primary: 0.9})


def s(score):
    return Sentiment(score=score, confidence=0.9, label="negative" if score < 0 else "positive")


def test_impact_is_bounded():
    for cls in ["credit_event", "none", "product_launch"]:
        for sc in [-1, 0, 1]:
            imp = rule_impact(ev(cls), s(sc), novelty=1.0, n_sources=500, source_weight=1.0)
            assert 1 <= imp.score <= 10


def test_impact_orders_drivers_sensibly():
    big = rule_impact(ev("credit_event"), s(-0.9), novelty=1.0, n_sources=20, source_weight=1.0)
    stale = rule_impact(ev("credit_event"), s(-0.9), novelty=0.05, n_sources=20, source_weight=1.0)
    small = rule_impact(ev("product_launch"), s(0.1), novelty=1.0, n_sources=1, source_weight=0.5)
    assert big.raw > stale.raw > small.raw
    assert big.score >= 8 and small.score <= 3
