"""Top up rare event classes in the gold pool.

Usage: python scripts/boost_gold_pool.py [--per-class 25]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from build_gold_pool import to5  # noqa: E402

from riskengine.config import RAW_DIR  # noqa: E402
from riskengine.gold import POOL, _read_jsonl  # noqa: E402
from riskengine.ingest.historical import fnspid_items  # noqa: E402
from riskengine.nlp.events import EventClassifier  # noqa: E402
from riskengine.nlp.pipeline import RiskEngine  # noqa: E402

RARE = ["credit_event", "management_change", "operational"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pool = _read_jsonl(POOL)
    seen = {p["item_id"] for p in pool}

    df = pd.read_parquet(RAW_DIR / "fnspid_articles_universe.parquet").sample(frac=1.0, random_state=args.seed)
    lex = EventClassifier(use_zero_shot=False)
    want = {c: [] for c in RARE}
    for chunk_start in range(0, len(df), 5000):
        chunk = df.iloc[chunk_start:chunk_start + 5000]
        items = list(fnspid_items(chunk))
        for it, body in zip(items, chunk["Article"], strict=True):
            it.body = body[:1500] if isinstance(body, str) else ""
        preds = lex.classify([it.title for it in items])          # headline lexicon only: precise and fast
        for it, p in zip(items, preds, strict=True):
            if it.item_id in seen or p.primary not in want or len(want[p.primary]) >= args.per_class * 3:
                continue
            want[p.primary].append(it)
        if all(len(v) >= args.per_class * 3 for v in want.values()):
            break

    chosen = [it for c in RARE for it in rng.sample(want[c], min(args.per_class, len(want[c])))]
    engine = RiskEngine()
    sigs = engine.process(chosen)
    by_item: dict[str, list] = {}
    for s in sigs:
        by_item.setdefault(s.item_id, []).append(s)
    added = []
    for c in chosen:
        ss = by_item.get(c.item_id)
        if not ss:
            continue
        ev = ss[0].event
        added.append({
            "item_id": c.item_id, "source": "fnspid_articles", "source_type": c.source_type, "title": c.title,
            "body": c.body[:1500], "url": c.url, "publisher": c.publisher, "published_at": c.published_at.isoformat(),
            "suggestions": {"entities": [{"entity": s.entity, "sentiment": to5(s.sentiment.score),
                                          "score": s.sentiment.score} for s in ss],
                            "events": ev.labels if ev.primary != "none" else ["none"],
                            "primary": ev.primary, "subtype": ev.subtype},
            "stratum": "rare_boost",
        })
    # interleave the boost items into the unlabeled tail so labeling order stays mixed
    labeled_prefix = len(pool)
    tail = pool[labeled_prefix:] + added
    rng.shuffle(tail)
    with open(POOL, "w", encoding="utf-8") as fh:
        for r in pool[:labeled_prefix] + tail:
            fh.write(json.dumps(r) + "\n")
    print(f"added {len(added)} rare-class items "
          f"{pd.Series([a['suggestions']['primary'] for a in added]).value_counts().to_dict()}; pool now {len(pool) + len(added)}")


if __name__ == "__main__":
    main()
