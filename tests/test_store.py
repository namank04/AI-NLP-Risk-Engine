from datetime import UTC, datetime

from riskengine.schema import (
    EventClassification,
    Evidence,
    Impact,
    Scope,
    Sentiment,
    Signal,
)
from riskengine.store import SignalStore


def sig(i, entity="AAPL", subtype=None, url=None, channels=("equity",)):
    return Signal(signal_id=f"s{i}", item_id=f"i{i}", cluster_id="c", entity=entity, entity_type="company",
                  source="test", source_type="news", published_at=datetime(2024, 1, 1, i, tzinfo=UTC),
                  sentiment=Sentiment(score=-0.5, confidence=0.9, label="negative"),
                  event=EventClassification(primary="earnings", labels=["earnings", "credit_event"],
                                            probs={"earnings": 0.9}, subtype=subtype),
                  impact=Impact(score=7, raw=7.2, confidence=0.3, method="rule_v0"),
                  scope=Scope(countries=["US"]), channels=list(channels), novelty=1.0,
                  evidence=Evidence(headline="h", url=url), model_version="t")


def test_roundtrip_with_lists_and_nulls():
    st = SignalStore(":memory:")
    st.add_signals([sig(1), sig(2, subtype="miss", url="http://x", channels=())])
    out = {s.signal_id: s for s in st.signals(limit=10)}
    assert out["s1"].event.labels == ["earnings", "credit_event"] and out["s1"].event.subtype is None
    assert out["s1"].evidence.url is None and out["s1"].channels == ["equity"]
    assert out["s2"].channels == [] and out["s2"].scope.countries == ["US"]
    assert [s.signal_id for s in st.signals(event="credit_event", min_impact=7)] == ["s2", "s1"]
