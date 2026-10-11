"""Run the LLM teacher (Groq free tier by default; set LLM_API_KEY in .env)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from riskengine.gold import POOL, _read_jsonl, gold_item_ids  # noqa: E402
from riskengine.teacher.llm import DailyLimitReached, Teacher  # noqa: E402

TRAIN_END = "2023-07-01"


def gold_rows() -> list[tuple[str, str, list[str]]]:
    rows = []
    for p in _read_jsonl(POOL):
        text = f"{p['title']}. {p.get('body', '')}".strip(". ")
        rows.append((p["item_id"], text, [e["entity"] for e in p["suggestions"]["entities"]]))
    return rows


def silver_rows(n: int, seed: int = 11) -> list[tuple[str, str, list[str]]]:
    sig = pd.read_parquet(PROCESSED_DIR / "signals_fnspid_2022_2023.parquet")
    sig = sig[pd.to_datetime(sig["published_at"], utc=True) < pd.Timestamp(TRAIN_END, tz="UTC")]
    sig = sig[~sig["item_id"].isin(gold_item_ids())]
    items = sig.groupby("item_id").agg(entities=("entity", lambda s: sorted(set(s))), primary=("event_primary", "first"))
    # oversample rare classes: weight ~ 1 / sqrt(class frequency)
    freq = items["primary"].map(items["primary"].value_counts())
    picked = items.sample(n=min(n, len(items)), weights=1 / freq ** 0.5, random_state=seed)
    arts = pd.read_parquet(RAW_DIR / "fnspid_articles_universe.parquet", columns=["Url", "ticker", "Article_title", "Article"])
    from riskengine.schema import stable_id

    arts["item_id"] = [stable_id("fnspid", u, t) for u, t in zip(arts["Url"], arts["ticker"], strict=True)]
    arts = arts.drop_duplicates("item_id").set_index("item_id")
    rows = []
    for item_id, r in picked.iterrows():
        if item_id not in arts.index:
            continue
        a = arts.loc[item_id]
        body = a["Article"] if isinstance(a["Article"], str) else ""
        rows.append((item_id, f"{a['Article_title']}. {body}", list(r["entities"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", choices=["gold", "silver"])
    ap.add_argument("--n", type=int, default=3000, help="silver sample size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()
    teacher = Teacher()
    if not teacher.cfg.api_key:
        sys.exit("Set LLM_API_KEY in .env (Groq: console.groq.com -> API Keys).")
    if args.ping:
        print(f"{teacher.cfg.model} @ {teacher.cfg.base_url}: {teacher.ping()[:200]}")
        return
    if not args.target:
        ap.error("choose gold or silver")
    rows = gold_rows() if args.target == "gold" else silver_rows(args.n)
    todo = [r for r in rows if teacher._key(r[0], 0) not in teacher.cache]
    print(f"{args.target}: {len(rows)} items, {len(rows) - len(todo)} already cached, {len(todo)} to label "
          f"with {teacher.cfg.model}")
    done = 0
    try:
        for k in range(0, len(todo), args.batch):
            for _attempt in range(3):
                try:
                    labels = teacher.label_batch(todo[k:k + args.batch])
                    break
                except DailyLimitReached:
                    raise
                except RuntimeError:
                    time.sleep(60)          # per-minute token limit: wait for the window to reset
            else:
                continue
            done += sum(lab is not None for lab in labels)
            print(f"\r  {k + len(labels)}/{len(todo)} sent, {done} usable labels", end="", flush=True)
    except DailyLimitReached as exc:
        print(f"\nDaily quota reached ({exc}). Run the same command tomorrow to continue.")
        return
    print("\ndone")


if __name__ == "__main__":
    main()
