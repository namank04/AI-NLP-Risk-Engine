"""Gold-set store for hand labeling (see docs/labeling_guidelines.md)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

from riskengine.config import GOLD_DIR

POOL = GOLD_DIR / "pool.jsonl"
LABELS = GOLD_DIR / "labels.jsonl"


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for rec in _read_jsonl(LABELS):
        latest[rec["item_id"]] = rec
    return latest


def gold_item_ids() -> set[str]:
    return {r["item_id"] for r in _read_jsonl(POOL)}


class GoldStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.pool = _read_jsonl(POOL)
        self.labels = load_labels()

    def reload(self) -> None:
        with self._lock:
            self.pool = _read_jsonl(POOL)
            self.labels = load_labels()

    def next(self, after: str | None = None) -> dict | None:
        """Next unlabeled item (optionally the first one after `after` in pool order)."""
        start = 0
        if after:
            ids = [p["item_id"] for p in self.pool]
            start = ids.index(after) + 1 if after in ids else 0
        for p in self.pool[start:] + self.pool[:start]:
            if p["item_id"] not in self.labels:
                return p
        return None

    def get(self, item_id: str) -> dict | None:
        item = next((p for p in self.pool if p["item_id"] == item_id), None)
        if item is None:
            return None
        return item | {"label": self.labels.get(item_id)}

    def label(self, rec: dict) -> dict:
        item = next((p for p in self.pool if p["item_id"] == rec["item_id"]), None)
        if item is None:
            raise KeyError(rec["item_id"])
        sug = item.get("suggestions", {})
        sug_sent = {e["entity"]: e["sentiment"] for e in sug.get("entities", [])}
        rec = dict(rec)
        rec["labeled_at"] = datetime.now(UTC).isoformat()
        rec["changed_event"] = sorted(rec.get("events", [])) != sorted(sug.get("events", []))
        rec["changed_sentiment"] = any(sug_sent.get(e["entity"]) != e.get("sentiment") for e in rec.get("entities", []))
        with self._lock:
            LABELS.parent.mkdir(parents=True, exist_ok=True)
            with open(LABELS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.labels[rec["item_id"]] = rec
        return rec

    def stats(self) -> dict:
        labeled = [r for r in self.labels.values() if not r.get("skipped")]
        by_event: dict[str, int] = {}
        for r in labeled:
            for e in r.get("events", [])[:1]:
                by_event[e] = by_event.get(e, 0) + 1
        by_source: dict[str, int] = {}
        pool_by_id = {p["item_id"]: p for p in self.pool}
        for r in labeled:
            src = pool_by_id.get(r["item_id"], {}).get("source", "?")
            by_source[src] = by_source.get(src, 0) + 1
        n = len(labeled)
        return {
            "pool": len(self.pool), "labeled": n, "skipped": len(self.labels) - n,
            "remaining": len(self.pool) - len(self.labels), "by_event": by_event, "by_source": by_source,
            "baseline_event_changed": round(sum(r.get("changed_event", False) for r in labeled) / n, 3) if n else None,
            "baseline_sentiment_changed": round(sum(r.get("changed_sentiment", False) for r in labeled) / n, 3) if n else None,
        }
