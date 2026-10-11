"""Evaluate predictors on the gold set.

Usage: python scripts/evaluate_models.py baseline [teacher]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, ROOT  # noqa: E402
from riskengine.evaluation import evaluate, to_markdown  # noqa: E402
from riskengine.gold import POOL, _read_jsonl, load_labels  # noqa: E402
from riskengine.schema import RawItem  # noqa: E402


def pool_items(labeled: set[str]) -> list[dict]:
    return [p for p in _read_jsonl(POOL) if p["item_id"] in labeled]


def as_raw(p: dict) -> RawItem:
    return RawItem(item_id=p["item_id"], source=p["source"], source_type=p["source_type"], title=p["title"],
                   body=p.get("body", ""), url=p.get("url"), publisher=p.get("publisher"),
                   published_at=datetime.fromisoformat(p["published_at"]))


def predict_baseline(pool: list[dict]) -> dict[str, dict]:
    from build_gold_pool import to5  # type: ignore

    from riskengine.nlp.pipeline import RiskEngine

    engine = RiskEngine()
    out: dict[str, dict] = {}
    for s in engine.process([as_raw(p) for p in pool]):
        rec = out.setdefault(s.item_id, {"entities": [], "events": s.event.labels if s.event.primary != "none" else ["none"],
                                         "subtype": s.event.subtype})
        rec["entities"].append({"entity": s.entity, "relevant": True, "sentiment": to5(s.sentiment.score),
                                "score": s.sentiment.score})
    for p in pool:
        out.setdefault(p["item_id"], {"entities": [], "events": ["none"], "subtype": None})
    return out


def predict_teacher(pool: list[dict]) -> dict[str, dict]:
    from riskengine.teacher.llm import Teacher

    t = Teacher()
    rows = []
    for p in pool:
        ents = [e["entity"] for e in p["suggestions"]["entities"]]
        text = f"{p['title']}. {p.get('body', '')}".strip(". ")
        rows.append((p["item_id"], text, ents))
    labels = t.label_many(rows)
    return {p["item_id"]: lab for p, lab in zip(pool, labels, strict=True) if lab is not None}


def main() -> None:
    names = sys.argv[1:] or ["baseline"]
    gold = load_labels()
    pool = pool_items(set(gold))
    print(f"{len(gold)} gold labels, {len(pool)} pool items")
    for name in names:
        preds = {"baseline": predict_baseline, "teacher": predict_teacher}[name](pool)
        res = evaluate(gold, preds)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / f"eval_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (ROOT / "docs" / f"eval_{name}.md").write_text(to_markdown(name, res), encoding="utf-8")
        print(to_markdown(name, res))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
