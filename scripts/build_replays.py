"""Build replay streams (data/replay/<name>.parquet) for deterministic demos.

Usage: python scripts/build_replays.py recent svb_2023 [--step 30]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import RAW_DIR  # noqa: E402
from riskengine.ingest.edgar import EdgarSource  # noqa: E402
from riskengine.ingest.gdelt import GdeltGkgSource, _slots  # noqa: E402
from riskengine.ingest.historical import fnspid_items  # noqa: E402
from riskengine.nlp.entities import default_linker  # noqa: E402
from riskengine.schema import RawItem  # noqa: E402
from riskengine.service import save_replay  # noqa: E402

WINDOWS = {
    "svb_2023": ("2023-03-06", "2023-03-24"),
    "ukraine_2022": ("2022-02-17", "2022-03-10"),
    "covid_2020": ("2020-02-19", "2020-03-27"),
}


def relevant(items: list[RawItem]) -> list[RawItem]:
    lk = default_linker()
    out = []
    for i in items:
        hint = "ECON_" in i.labels_hint.get("gdelt_themes", "")
        if i.tickers_hint or lk.link(i.text, require_context=i.source == "gdelt", context_hint=hint):
            out.append(i)
    return out


def gkg(start: datetime, end: datetime, step_min: int) -> list[RawItem]:
    src = GdeltGkgSource()
    slots = [s for s in _slots(start, end) if (s.hour * 60 + s.minute) % step_min == 0]
    items: list[RawItem] = []
    for k, slot in enumerate(slots):
        items += relevant(src.fetch(slot, slot))
        print(f"\r  GKG {k + 1}/{len(slots)} slots, {len(items):,} relevant items", end="", flush=True)
    print()
    return items


def fnspid(start: datetime, end: datetime) -> list[RawItem]:
    out: list[RawItem] = []
    for fname, has_body in (("fnspid_articles_universe.parquet", True), ("fnspid_headlines_universe.parquet", False)):
        p = RAW_DIR / fname
        if not p.exists():
            print(f"  ({fname} not downloaded yet, skipped)")
            continue
        df = pd.read_parquet(p)
        df = df[(df["published_at"] >= start) & (df["published_at"] < end)]
        items = list(fnspid_items(df))
        if has_body:
            bodies = dict(zip(df["Url"], df["Article"], strict=False))
            for it in items:
                body = bodies.get(it.url)
                if isinstance(body, str):
                    it.body = body
        out += items
        print(f"  {fname}: {len(items):,} items")
    return out


def dedupe(items: list[RawItem]) -> list[RawItem]:
    seen, out = set(), []
    for i in sorted(items, key=lambda x: x.published_at):
        if i.item_id not in seen:
            seen.add(i.item_id)
            out.append(i)
    return out


def build(name: str, step: int, days: int) -> None:
    print(f"== {name}")
    if name == "recent":
        end = datetime.now(UTC) - timedelta(minutes=30)
        start = end - timedelta(days=days)
        items = gkg(start, end, step) + EdgarSource().fetch(end - timedelta(days=30))
    else:
        s, e = WINDOWS[name]
        start = datetime.fromisoformat(s).replace(tzinfo=UTC)
        end = datetime.fromisoformat(e).replace(tzinfo=UTC)
        items = gkg(start, end, step) + fnspid(start, end)
    items = dedupe(items)
    save_replay(name, items)
    by_src = pd.Series([i.source for i in items]).value_counts().to_dict()
    print(f"  saved {len(items):,} items {by_src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=["recent", *WINDOWS])
    ap.add_argument("--step", type=int, default=30, help="GKG sampling interval in minutes (15 = every file)")
    ap.add_argument("--days", type=int, default=3, help="lookback for `recent`")
    args = ap.parse_args()
    for n in args.names:
        build(n, args.step, args.days)


if __name__ == "__main__":
    main()
