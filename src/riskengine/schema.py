"""Data contracts.

RawItem: one piece of text as it arrives from any source.
Signal:  the engine's structured output, one per (item, entity). This is the
         contract every downstream consumer (API, Module A, Module B) relies on.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["news", "social", "filing"]
EntityType = Literal["company", "market"]

# Entity used for signals that concern the whole market rather than one company
# (e.g. a Fed decision with no company mentioned).
MARKET_ENTITY = "MARKET"


def utcnow() -> datetime:
    return datetime.now(UTC)


def stable_id(*parts: object) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:16]


class RawItem(BaseModel):
    item_id: str
    source: str                         # e.g. "gdelt", "google_news", "edgar", "fnspid", "tweets_hist"
    source_type: SourceType
    title: str
    body: str = ""
    url: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: datetime              # when the text became public (UTC)
    ingested_at: datetime = Field(default_factory=utcnow)
    tickers_hint: list[str] = Field(default_factory=list)   # tickers the source itself attached
    labels_hint: dict[str, str] = Field(default_factory=dict)  # e.g. {"8k_items": "2.02,9.01"}
    engagement: dict[str, float] = Field(default_factory=dict)  # likes/retweets/comments
    lang: str = "en"

    @property
    def text(self) -> str:
        if self.body and not self.body.startswith(self.title):
            return f"{self.title}. {self.body}" if self.title else self.body
        return self.body or self.title


class Sentiment(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)          # P(positive) - P(negative)
    confidence: float = Field(ge=0.0, le=1.0)
    label: Literal["positive", "neutral", "negative"]


class EventClassification(BaseModel):
    primary: str                                   # taxonomy key, e.g. "credit_event"
    labels: list[str]                              # all keys above threshold (multi-label)
    probs: dict[str, float]
    subtype: str | None = None


class Impact(BaseModel):
    score: int = Field(ge=1, le=10)
    raw: float                                     # continuous model output before bucketing
    confidence: float = Field(ge=0.0, le=1.0)
    method: str                                    # "rule_v0", "lgbm_event_study_v1", ...


class Scope(BaseModel):
    countries: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)
    currencies: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    headline: str
    url: str | None = None
    span: str | None = None                        # sentence(s) that mention the entity


class Signal(BaseModel):
    signal_id: str
    item_id: str
    cluster_id: str
    entity: str                                    # ticker or MARKET_ENTITY
    entity_type: EntityType
    source: str
    source_type: SourceType
    published_at: datetime
    processed_at: datetime = Field(default_factory=utcnow)
    sentiment: Sentiment
    event: EventClassification
    impact: Impact
    scope: Scope = Field(default_factory=Scope)
    channels: list[str] = Field(default_factory=list)
    novelty: float = Field(ge=0.0, le=1.0)         # 1 = first time this story is seen
    n_sources: int = 1                             # distinct sources in the story cluster so far
    source_weight: float = 1.0                     # credibility weight of this source
    evidence: Evidence
    model_version: str
    tags: list[str] = Field(default_factory=list)   # e.g. "8k:1.03" for filing item codes
