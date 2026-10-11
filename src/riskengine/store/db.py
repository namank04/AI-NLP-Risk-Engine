"""DuckDB storage for raw items and signals.

Signals are stored flattened (one column per field) so downstream consumers
can aggregate with SQL; nested parts (probs, scope) are JSON columns.
DuckDB is single-writer: one process (the API server or a batch script)
owns the database file at a time.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from riskengine.config import DB_PATH
from riskengine.schema import (
    EventClassification,
    Evidence,
    Impact,
    RawItem,
    Scope,
    Sentiment,
    Signal,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_items (
    item_id VARCHAR PRIMARY KEY, source VARCHAR, source_type VARCHAR, title VARCHAR, body VARCHAR,
    url VARCHAR, publisher VARCHAR, author VARCHAR, published_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ,
    tickers_hint VARCHAR[], labels_hint JSON, engagement JSON
);
CREATE TABLE IF NOT EXISTS signals (
    signal_id VARCHAR PRIMARY KEY, item_id VARCHAR, cluster_id VARCHAR, entity VARCHAR, entity_type VARCHAR,
    source VARCHAR, source_type VARCHAR, published_at TIMESTAMPTZ, processed_at TIMESTAMPTZ,
    sentiment_score DOUBLE, sentiment_conf DOUBLE, sentiment_label VARCHAR,
    event_primary VARCHAR, event_labels VARCHAR[], event_subtype VARCHAR, event_probs JSON,
    impact_score INTEGER, impact_raw DOUBLE, impact_conf DOUBLE, impact_method VARCHAR,
    scope JSON, channels VARCHAR[], novelty DOUBLE, n_sources INTEGER, source_weight DOUBLE,
    headline VARCHAR, url VARCHAR, span VARCHAR, model_version VARCHAR, tags VARCHAR[]
);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS tags VARCHAR[];
CREATE INDEX IF NOT EXISTS idx_signals_entity_time ON signals (entity, published_at);
"""

SIGNAL_COLUMNS = [
    "signal_id", "item_id", "cluster_id", "entity", "entity_type", "source", "source_type", "published_at",
    "processed_at", "sentiment_score", "sentiment_conf", "sentiment_label", "event_primary", "event_labels",
    "event_subtype", "event_probs", "impact_score", "impact_raw", "impact_conf", "impact_method", "scope",
    "channels", "novelty", "n_sources", "source_weight", "headline", "url", "span", "model_version", "tags",
]


def signal_to_row(s: Signal) -> dict:
    return {
        "signal_id": s.signal_id, "item_id": s.item_id, "cluster_id": s.cluster_id, "entity": s.entity,
        "entity_type": s.entity_type, "source": s.source, "source_type": s.source_type,
        "published_at": s.published_at, "processed_at": s.processed_at,
        "sentiment_score": s.sentiment.score, "sentiment_conf": s.sentiment.confidence,
        "sentiment_label": s.sentiment.label, "event_primary": s.event.primary, "event_labels": s.event.labels,
        "event_subtype": s.event.subtype, "event_probs": json.dumps(s.event.probs),
        "impact_score": s.impact.score, "impact_raw": s.impact.raw, "impact_conf": s.impact.confidence,
        "impact_method": s.impact.method, "scope": s.scope.model_dump_json(), "channels": s.channels,
        "novelty": s.novelty, "n_sources": s.n_sources, "source_weight": s.source_weight,
        "headline": s.evidence.headline, "url": s.evidence.url, "span": s.evidence.span,
        "model_version": s.model_version, "tags": s.tags,
    }


def row_to_signal(r: dict) -> Signal:
    def js(v):
        return json.loads(v) if isinstance(v, str) else (v or {})

    return Signal(
        signal_id=r["signal_id"], item_id=r["item_id"], cluster_id=r["cluster_id"], entity=r["entity"],
        entity_type=r["entity_type"], source=r["source"], source_type=r["source_type"],
        published_at=r["published_at"], processed_at=r["processed_at"],
        sentiment=Sentiment(score=r["sentiment_score"], confidence=r["sentiment_conf"], label=r["sentiment_label"]),
        event=EventClassification(primary=r["event_primary"], labels=list(r["event_labels"] or []),
                                  probs=js(r["event_probs"]), subtype=r["event_subtype"]),
        impact=Impact(score=r["impact_score"], raw=r["impact_raw"], confidence=r["impact_conf"],
                      method=r["impact_method"]),
        scope=Scope(**js(r["scope"])), channels=list(r["channels"] or []), novelty=r["novelty"],
        n_sources=r["n_sources"], source_weight=r["source_weight"],
        evidence=Evidence(headline=r["headline"], url=r["url"], span=r["span"]),
        model_version=r["model_version"], tags=list(r.get("tags") or []),
    )


class SignalStore:
    def __init__(self, path: Path | str = DB_PATH, read_only: bool = False):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path, read_only=read_only)
        self._lock = threading.Lock()
        if not read_only:
            self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    # ---- writes -------------------------------------------------------------
    def add_items(self, items: list[RawItem]) -> None:
        if not items:
            return
        df = pd.DataFrame([{
            "item_id": i.item_id, "source": i.source, "source_type": i.source_type, "title": i.title,
            "body": i.body, "url": i.url, "publisher": i.publisher, "author": i.author,
            "published_at": i.published_at, "ingested_at": i.ingested_at, "tickers_hint": i.tickers_hint,
            "labels_hint": json.dumps(i.labels_hint), "engagement": json.dumps(i.engagement),
        } for i in items])
        with self._lock:
            self.con.register("_items_df", df)
            self.con.execute("INSERT OR REPLACE INTO raw_items SELECT * FROM _items_df")
            self.con.unregister("_items_df")

    def add_signals(self, signals: list[Signal]) -> None:
        if not signals:
            return
        df = pd.DataFrame([signal_to_row(s) for s in signals], columns=SIGNAL_COLUMNS)
        with self._lock:
            self.con.register("_sig_df", df)
            self.con.execute(f"INSERT OR REPLACE INTO signals SELECT {', '.join(SIGNAL_COLUMNS)} FROM _sig_df")
            self.con.unregister("_sig_df")

    # ---- reads --------------------------------------------------------------
    def query_df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self._lock:
            return self.con.execute(sql, params or []).df()

    def signals(self, entity: str | None = None, since: datetime | None = None, until: datetime | None = None,
                event: str | None = None, min_impact: int | None = None, limit: int = 200) -> list[Signal]:
        where, params = [], []
        if entity:
            where.append("entity = ?")
            params.append(entity)
        if since:
            where.append("published_at >= ?")
            params.append(since)
        if until:
            where.append("published_at < ?")
            params.append(until)
        if event:
            where.append("list_contains(event_labels, ?)")
            params.append(event)
        if min_impact:
            where.append("impact_score >= ?")
            params.append(min_impact)
        sql = "SELECT * FROM signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)
        # plain Python rows: pandas would turn lists into arrays and NULLs into NaN
        with self._lock:
            cur = self.con.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [row_to_signal(dict(zip(cols, r, strict=True))) for r in rows]

    def count(self, table: str = "signals") -> int:
        with self._lock:
            return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
