"""Run the engine over a historical corpus and save every signal as parquet.

The output feeds the impact model (event-study labels) and the Module A
walk-forward backtest, so both are evaluated on exactly what the live engine
would have produced at the time.

Usage: python scripts/run_engine_batch.py --start 2022-01-01 --end 2023-12-16 --name fnspid_2022_2023
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from riskengine.gold import gold_item_ids  # noqa: E402
from riskengine.ingest.historical import fnspid_items  # noqa: E402
from riskengine.nlp.pipeline import RiskEngine  # noqa: E402


def flatten(s) -> dict:
    return {
        "signal_id": s.signal_id, "item_id": s.item_id, "cluster_id": s.cluster_id, "entity": s.entity,
        "entity_type": s.entity_type, "source": s.source, "published_at": s.published_at,
        "sentiment": s.sentiment.score, "sentiment_conf": s.sentiment.confidence,
        "event_primary": s.event.primary, "event_labels": ",".join(s.event.labels), "event_subtype": s.event.subtype,
        **{f"p_{k}": v for k, v in s.event.probs.items()},
        "impact_rule": s.impact.score, "impact_rule_raw": s.impact.raw, "novelty": s.novelty,
        "n_sources": s.n_sources, "source_weight": s.source_weight, "headline": s.evidence.headline,
        "publisher": None, "model_version": s.model_version,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-12-16")
    ap.add_argument("--name", default="fnspid_2022_2023")
    ap.add_argument("--body-chars", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(RAW_DIR / "fnspid_articles_universe.parquet")
    df = df[(df["published_at"] >= args.start) & (df["published_at"] < args.end)].sort_values("published_at")
    items = list(fnspid_items(df))
    publishers = {}
    for it, body, pub in zip(items, df["Article"], df["Publisher"], strict=True):
        it.body = body[: args.body_chars] if isinstance(body, str) else ""
        publishers[it.item_id] = pub if isinstance(pub, str) else None
    held_out = gold_item_ids()
    print(f"{len(items):,} articles {args.start}..{args.end} ({len(held_out)} gold ids kept for evaluation only)")

    engine = RiskEngine()
    out_dir = PROCESSED_DIR / f"signals_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k in range(0, len(items), args.chunk):
        part = out_dir / f"part_{k // args.chunk:05d}.parquet"
        if part.exists():                       # resumable
            # replay the chunk through the clusterer so novelty/breadth stay consistent
            for it in items[k:k + args.chunk]:
                engine.clusterer.assign(it)
            continue
        sigs = engine.process(items[k:k + args.chunk])
        rows = [flatten(s) for s in sigs]
        for r in rows:
            r["publisher"] = publishers.get(r["item_id"])
            r["is_gold"] = r["item_id"] in held_out
        pd.DataFrame(rows).to_parquet(part, index=False)
        done = min(k + args.chunk, len(items))
        rate = done / (time.time() - t0)
        print(f"  {done:,}/{len(items):,} items, {rate:.1f} items/s, eta {(len(items) - done) / rate / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("part_*.parquet"))], ignore_index=True)
    allp.to_parquet(PROCESSED_DIR / f"signals_{args.name}.parquet", index=False)
    print(f"saved {len(allp):,} signals -> signals_{args.name}.parquet")


if __name__ == "__main__":
    main()
