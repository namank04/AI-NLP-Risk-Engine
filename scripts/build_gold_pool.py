"""Build the gold labeling pool (data/gold/pool.jsonl).

Usage: python scripts/build_gold_pool.py [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import GOLD_DIR, RAW_DIR, REPLAY_DIR  # noqa: E402
from riskengine.gold import POOL, _read_jsonl  # noqa: E402
from riskengine.ingest.historical import fnspid_items, tweet_items  # noqa: E402
from riskengine.nlp.pipeline import RiskEngine  # noqa: E402
from riskengine.schema import RawItem  # noqa: E402

QUOTAS = {"fnspid_articles": 230, "fnspid_headlines": 120, "tweets_hist": 150, "gdelt": 220, "edgar": 80}


def to5(score: float) -> int:
    return int(np.digitize(score, [-0.6, -0.2, 0.2, 0.6]) - 2)


def candidates(source: str, n: int, rng: random.Random) -> list[RawItem]:
    if source == "fnspid_articles":
        p = RAW_DIR / "fnspid_articles_universe.parquet"
        if not p.exists():
            return []
        df = pd.read_parquet(p).sample(n=n, random_state=rng.randint(0, 10**6))
        items = list(fnspid_items(df))
        for it, body in zip(items, df["Article"], strict=True):
            it.body = body if isinstance(body, str) else ""
            it.source = "fnspid_articles"
        return items
    if source == "fnspid_headlines":
        df = pd.read_parquet(RAW_DIR / "fnspid_headlines_universe.parquet")
        return list(fnspid_items(df.sample(n=min(n, len(df)), random_state=rng.randint(0, 10**6))))
    if source == "tweets_hist":
        df = pd.read_parquet(RAW_DIR / "stock_tweets_universe.parquet")
        return list(tweet_items(df.sample(n=n, random_state=rng.randint(0, 10**6))))
    if source == "gdelt":
        p = REPLAY_DIR / "recent.parquet"
        if not p.exists():
            return []
        from riskengine.service import load_replay

        items = [i for i in load_replay("recent") if i.source == "gdelt"]
        return rng.sample(items, min(n, len(items)))
    if source == "edgar":
        from riskengine.ingest.edgar import EdgarSource

        items = EdgarSource().fetch(datetime.now(UTC) - timedelta(days=540))
        return rng.sample(items, min(n, len(items)))
    raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--oversample", type=int, default=5)
    ap.add_argument("--max-none", type=float, default=0.25)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    existing = _read_jsonl(POOL)
    seen = {p["item_id"] for p in existing}
    engine = RiskEngine()
    picked: list[dict] = []
    for source, quota in QUOTAS.items():
        cands = [c for c in candidates(source, quota * args.oversample, rng) if c.item_id not in seen]
        if not cands:
            print(f"{source}: no data yet, skipped")
            continue
        sigs = engine.process(cands)
        by_item: dict[str, list] = {}
        for s in sigs:
            by_item.setdefault(s.item_id, []).append(s)
        rows = []
        for c in cands:
            ss = by_item.get(c.item_id)
            if not ss:
                continue
            ev = ss[0].event
            rows.append({
                "item_id": c.item_id, "source": source, "source_type": c.source_type,
                "title": c.title, "body": (c.body or "")[:1500], "url": c.url, "publisher": c.publisher,
                "published_at": c.published_at.isoformat(),
                "suggestions": {
                    "entities": [{"entity": s.entity, "sentiment": to5(s.sentiment.score),
                                  "score": s.sentiment.score} for s in ss],
                    "events": ev.labels if ev.primary != "none" else ["none"],
                    "primary": ev.primary, "subtype": ev.subtype,
                },
            })
        freq = pd.Series([r["suggestions"]["primary"] for r in rows]).value_counts()
        weights = np.array([1 / np.sqrt(freq[r["suggestions"]["primary"]]) for r in rows])
        order = list(np.random.default_rng(rng.randint(0, 10**6)).choice(
            len(rows), size=len(rows), replace=False, p=weights / weights.sum()))
        chosen, n_none = [], 0
        for i in order:
            r = rows[i]
            if r["suggestions"]["primary"] == "none":
                if n_none >= args.max_none * quota:
                    continue
                n_none += 1
            chosen.append(r)
            if len(chosen) >= quota:
                break
        dist = pd.Series([r["suggestions"]["primary"] for r in chosen]).value_counts().to_dict()
        print(f"{source}: {len(cands)} candidates -> {len(rows)} linked -> {len(chosen)} picked {dist}")
        picked += chosen

    rng.shuffle(picked)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL, "w", encoding="utf-8") as fh:
        for r in existing + picked:
            fh.write(json.dumps(r) + "\n")
    print(f"pool: {len(existing)} existing + {len(picked)} new = {len(existing) + len(picked)}")


if __name__ == "__main__":
    main()
