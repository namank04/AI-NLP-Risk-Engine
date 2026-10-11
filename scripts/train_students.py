"""Train the event-classification student and compare it with the zero-shot baseline.

Usage: python scripts/train_students.py [--epochs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import PROCESSED_DIR, ROOT, taxonomy  # noqa: E402

CLASSES = [c for c in taxonomy() if c != "none"]
OUT = ROOT / "models" / "students" / "event"
TOPIC_MAP = {   # zeroshot/twitter-financial-news-topic label id -> engine class (None = no event; absent = dropped)
    0: "earnings", 1: "macroeconomic", 3: "macroeconomic", 4: "earnings", 5: "earnings", 6: "macroeconomic",
    8: "macroeconomic", 9: None, 10: "macroeconomic", 11: None, 12: "regulatory_legal", 13: "merger_acquisition",
    14: "macroeconomic", 15: None, 16: "geopolitical", 17: "management_change", 18: None, 19: None,
}


def onehot(classes: list[str]) -> list[float]:
    return [1.0 if c in classes else 0.0 for c in CLASSES]


def public_data():
    from datasets import load_dataset

    ds = load_dataset("zeroshot/twitter-financial-news-topic")
    out = {}
    for split in ("train", "validation"):
        rows = [(r["text"], onehot([TOPIC_MAP[r["label"]]] if TOPIC_MAP[r["label"]] else []))
                for r in ds[split] if r["label"] in TOPIC_MAP]
        out[split] = rows
    return out


def silver_data() -> list[tuple[str, list[float]]]:
    from riskengine.gold import gold_item_ids
    from riskengine.teacher.llm import CACHE

    if not CACHE.exists():
        return []
    gold = gold_item_ids()
    texts = {}
    sig_path = PROCESSED_DIR / "signals_fnspid_2022_2023.parquet"
    if sig_path.exists():
        import pandas as pd

        s = pd.read_parquet(sig_path, columns=["item_id", "headline"]).drop_duplicates("item_id")
        texts = dict(zip(s["item_id"], s["headline"], strict=True))
    rows = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        lab = rec.get("label")
        if rec.get("pass") != 0 or not lab or rec["item_id"] in gold or rec["item_id"] not in texts:
            continue
        rows.append((texts[rec["item_id"]], onehot([e for e in lab["events"] if e != "none"])))
    return rows


def macro_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    per = {}
    for j, c in enumerate(CLASSES):
        tp = ((y[:, j] == 1) & (p[:, j] == 1)).sum()
        fp = ((y[:, j] == 0) & (p[:, j] == 1)).sum()
        fn = ((y[:, j] == 1) & (p[:, j] == 0)).sum()
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        per[c] = round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4)
    return float(np.mean(list(per.values()))), per


def baseline_predict(texts: list[str]) -> np.ndarray:
    from riskengine.nlp.events import EventClassifier

    ec = EventClassifier(use_zero_shot=True, student=False)
    preds = ec.classify(texts)
    return np.array([onehot([c for c in p.labels if c != "none"]) for p in preds])


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(7)

    pub = public_data()
    silver = silver_data()
    train = pub["train"] + silver
    val = pub["validation"]
    print(f"train {len(train):,} (public {len(pub['train']):,} + silver {len(silver):,}), validation {len(val):,}")

    base = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(CLASSES), problem_type="multi_label_classification", ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def collate(batch):
        enc = tok([b[0] for b in batch], padding=True, truncation=True, max_length=96, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        return enc

    dl = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for k, batch in enumerate(dl):
            batch = {kk: v.to(dev) for kk, v in batch.items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item()
            if k % 100 == 0:
                print(f"\r  epoch {ep + 1}/{args.epochs} step {k}/{len(dl)} loss {tot / (k + 1):.4f}", end="", flush=True)
        print()
    print(f"trained in {time.time() - t0:.0f}s")

    model.eval()
    probs = []
    with torch.inference_mode():
        for k in range(0, len(val), 64):
            enc = tok([t for t, _ in val[k:k + 64]], padding=True, truncation=True, max_length=96, return_tensors="pt").to(dev)
            probs.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    p = np.concatenate(probs)
    y = np.array([lab for _, lab in val])
    stud_f1, stud_per = macro_f1(y, (p >= 0.5).astype(int))
    # save before loading the baseline so a memory spike there cannot lose the model
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    import gc

    gc.collect()
    base_f1, base_per = macro_f1(y, baseline_predict([t for t, _ in val]))
    adopt = stud_f1 > base_f1
    res = {"validation_items": len(val), "train_items": len(train), "silver_items": len(silver),
           "student_macro_f1": round(stud_f1, 4), "baseline_macro_f1": round(base_f1, 4),
           "student_per_class": stud_per, "baseline_per_class": base_per, "adopted": adopt}
    # classes the student actually saw positives for; the engine keeps zero-shot for the rest
    counts = np.array([lab for _, lab in train]).sum(axis=0)
    covered = [c for c, n in zip(CLASSES, counts, strict=True) if n >= 50]
    (OUT / "coverage.json").write_text(json.dumps({"covered": covered, "train_positives":
                                                   dict(zip(CLASSES, counts.astype(int).tolist(), strict=True))}, indent=2))
    res["covered_classes"] = covered
    (OUT / "ADOPTED").unlink(missing_ok=True)
    if adopt:
        (OUT / "ADOPTED").write_text("student beat the zero-shot baseline on held-out validation\n")
    (PROCESSED_DIR / "student_event_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
