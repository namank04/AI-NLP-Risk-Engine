"""The risk engine: RawItem batches in, Signal lists out."""

from __future__ import annotations

from dataclasses import dataclass

from riskengine.config import taxonomy
from riskengine.nlp.credibility import source_weight
from riskengine.nlp.dedup import StoryClusterer
from riskengine.nlp.entities import EntityLinker, default_linker
from riskengine.nlp.events import EventClassifier
from riskengine.nlp.impact import rule_impact
from riskengine.nlp.sentiment import DEFAULT_NEWS_MODEL, SentimentModel
from riskengine.schema import MARKET_ENTITY, Evidence, RawItem, Signal, stable_id

MODEL_VERSION = "v0-baseline"
GENERAL_NEWS_SOURCES = {"gdelt", "google_news"}


@dataclass
class EngineConfig:
    news_sentiment_model: str = DEFAULT_NEWS_MODEL
    social_sentiment_model: str | None = None     # None: use the news model for social posts too
    use_zero_shot: bool = True
    event_threshold: float = 0.5
    min_entity_confidence: float = 0.5
    drop_duplicates: bool = False                 # keep copies (they raise n_sources) but mark novelty


class RiskEngine:
    def __init__(self, cfg: EngineConfig | None = None, linker: EntityLinker | None = None):
        self.cfg = cfg or EngineConfig()
        self.linker = linker or default_linker()
        self.clusterer = StoryClusterer()
        self.news_sent = SentimentModel(self.cfg.news_sentiment_model)
        self.social_sent = (SentimentModel(self.cfg.social_sentiment_model)
                            if self.cfg.social_sentiment_model else self.news_sent)
        self.events = EventClassifier(use_zero_shot=self.cfg.use_zero_shot, threshold=self.cfg.event_threshold)
        self.tax = taxonomy()

    def process(self, items: list[RawItem]) -> list[Signal]:
        items = sorted(items, key=lambda i: i.published_at)
        prepared = []
        for item in items:
            cl = self.clusterer.assign(item)
            if cl.is_duplicate and self.cfg.drop_duplicates:
                continue
            text = item.text
            # general-news firehoses need financial context; finance-specific feeds do not
            general = item.source in GENERAL_NEWS_SOURCES
            econ_tag = "ECON_" in item.labels_hint.get("gdelt_themes", "")
            mentions = [m for m in self.linker.link(text, item.tickers_hint, require_context=general,
                                                    context_hint=econ_tag)
                        if m.confidence >= self.cfg.min_entity_confidence]
            if not mentions:
                continue
            prepared.append((item, cl, text, mentions))
        if not prepared:
            return []

        events = self.events.classify([p[2][:1500] for p in prepared], [p[0].labels_hint for p in prepared])

        # sentiment per (item, entity), grouped by model so each model runs batched
        groups: dict[int, list[tuple[int, int, list[str]]]] = {}
        for i, (item, _, text, mentions) in enumerate(prepared):
            model_key = 1 if item.source_type == "social" else 0
            for j, m in enumerate(mentions):
                # item.text starts with the headline, so a headline that names the
                # entity is already among m.sentences
                groups.setdefault(model_key, []).append((i, j, m.sentences[:6] or [text[:500]]))
        sentiments: dict[tuple[int, int], object] = {}
        for model_key, entries in groups.items():
            model = self.social_sent if model_key == 1 else self.news_sent
            results = model.entity_sentiment([e[2] for e in entries])
            for (i, j, _), s in zip(entries, results, strict=True):
                sentiments[(i, j)] = s

        signals = []
        for i, (item, cl, text, mentions) in enumerate(prepared):
            ev = events[i]
            scope = self.linker.scope(text)
            channels = sorted({ch for c in ev.labels for ch in self.tax[c].channels})
            sw = source_weight(item)
            for j, m in enumerate(mentions):
                sent = sentiments[(i, j)]
                imp = rule_impact(ev, sent, cl.novelty, cl.n_sources, sw)
                signals.append(Signal(
                    signal_id=stable_id(item.item_id, m.entity),
                    item_id=item.item_id,
                    cluster_id=cl.cluster_id,
                    entity=m.entity,
                    entity_type="market" if m.entity == MARKET_ENTITY else "company",
                    source=item.source,
                    source_type=item.source_type,
                    published_at=item.published_at,
                    sentiment=sent,
                    event=ev,
                    impact=imp,
                    scope=scope,
                    channels=channels,
                    novelty=round(cl.novelty, 4),
                    n_sources=cl.n_sources,
                    source_weight=sw,
                    evidence=Evidence(headline=(item.title or text[:200]), url=item.url,
                                      span=" ".join(m.sentences[:2])[:500] or None),
                    model_version=MODEL_VERSION,
                    tags=[f"8k:{c.strip()}" for c in item.labels_hint.get("8k_items", "").split(",") if c.strip()],
                ))
        return signals
