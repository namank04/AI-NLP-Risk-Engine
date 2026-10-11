from datetime import UTC, datetime

import pandas as pd

from riskengine.config import taxonomy, universe
from riskengine.modules.index.rebalancer import IndexRebalancer
from riskengine.nlp.dedup import StoryClusterer
from riskengine.nlp.entities import default_linker
from riskengine.nlp.events import EventClassifier
from riskengine.nlp.pipeline import EngineConfig, RiskEngine
from riskengine.schema import RawItem, Sentiment


class FixedSentiment:
    def entity_sentiment(self, groups):
        return [Sentiment(score=-0.5, confidence=0.9, label="negative") for _ in groups]


def _engine() -> RiskEngine:
    eng = RiskEngine.__new__(RiskEngine)
    eng.cfg, eng.linker, eng.clusterer = EngineConfig(), default_linker(), StoryClusterer()
    eng.events = EventClassifier(use_zero_shot=False, student=False)
    eng.news_sent = eng.social_sent = FixedSentiment()
    eng.tax = taxonomy()
    return eng


def test_separate_story_memory_leaves_shared_clusterer_untouched():
    eng = _engine()
    item = RawItem(item_id="x", source="manual", source_type="news", title="Boeing shares slide after Moody's downgrade",
                   published_at=datetime.now(UTC))
    shared = eng.clusterer
    assert eng.process([item], clusterer=StoryClusterer())
    assert eng.clusterer is shared
    assert shared.assign(item).novelty == 1.0      # the shared memory never saw the analysed item


def test_index_reports_degraded_mode_without_market_data():
    t = universe().tickers
    idx = IndexRebalancer(benchmark=pd.Series(1 / len(t), index=t))
    snap = idx.maybe_rebalance(datetime.now(UTC), force=True)
    assert idx.method == "exp_tilt" and snap.degraded and "Market data unavailable" in snap.to_dict()["degraded"]
