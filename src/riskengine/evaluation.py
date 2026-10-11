"""Score any predictor against the gold set."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from riskengine.config import taxonomy

CLASSES = list(taxonomy())


def _sign(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(gold: dict[str, dict], preds: dict[str, dict]) -> dict:
    ids = [i for i, g in gold.items() if not g.get("skipped") and i in preds and preds[i] is not None]
    out: dict = {"n_items": len(ids), "n_gold": sum(not g.get("skipped") for g in gold.values())}
    if not ids:
        return out

    # ---- linking -------------------------------------------------------------
    linked = kept = added = 0
    for i in ids:
        g_ents = {e["entity"]: e for e in gold[i]["entities"]}
        p_ents = {e["entity"] for e in preds[i]["entities"]}
        linked += len(p_ents)
        kept += sum(1 for e in p_ents if g_ents.get(e, {}).get("relevant", False))
        added += sum(1 for e, ge in g_ents.items() if ge.get("relevant") and e not in p_ents)
    out["linking"] = {"linked": linked, "precision": round(kept / linked, 4) if linked else None,
                      "missed_entities": added}

    # ---- sentiment -----------------------------------------------------------
    y, yhat, scores = [], [], []
    for i in ids:
        g_ents = {e["entity"]: e for e in gold[i]["entities"] if e.get("relevant")}
        for pe in preds[i]["entities"]:
            ge = g_ents.get(pe["entity"])
            if ge is not None and pe.get("relevant", True):
                y.append(int(ge["sentiment"]))
                yhat.append(int(pe["sentiment"]))
                scores.append(float(pe.get("score", pe["sentiment"])))
    if y:
        ya, ha = np.array(y), np.array(yhat)
        f1s = []
        for c in (-1, 0, 1):
            ys, hs = np.vectorize(_sign)(ya), np.vectorize(_sign)(ha)
            tp = ((ys == c) & (hs == c)).sum()
            p = tp / max(1, (hs == c).sum())
            r = tp / max(1, (ys == c).sum())
            f1s.append(_f1(p, r))
        rho = spearmanr(scores, y).statistic if len(set(y)) > 1 else float("nan")
        out["sentiment"] = {
            "n": len(y), "macro_f1_3class": round(float(np.mean(f1s)), 4),
            "accuracy_3class": round(float((np.vectorize(_sign)(ya) == np.vectorize(_sign)(ha)).mean()), 4),
            "exact_5pt": round(float((ya == ha).mean()), 4),
            "within1_5pt": round(float((np.abs(ya - ha) <= 1).mean()), 4),
            "spearman": round(float(rho), 4),
        }

    # ---- events ----------------------------------------------------------------
    tp = {c: 0 for c in CLASSES}
    fp = dict(tp)
    fn = dict(tp)
    primary_ok = 0
    confusion = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for i in ids:
        gs, ps = set(gold[i]["events"]), set(preds[i]["events"])
        for c in CLASSES:
            tp[c] += c in gs and c in ps
            fp[c] += c not in gs and c in ps
            fn[c] += c in gs and c not in ps
        g0, p0 = gold[i]["events"][0], preds[i]["events"][0]
        primary_ok += g0 == p0
        if g0 in confusion and p0 in confusion[g0]:
            confusion[g0][p0] += 1
    per_class = {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(_f1(p, r), 4),
                        "support": tp[c] + fn[c]}
    supported = [c for c in CLASSES if per_class[c]["support"] > 0]
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    out["events"] = {
        "primary_accuracy": round(primary_ok / len(ids), 4),
        "micro_f1": round(_f1(TP / max(1, TP + FP), TP / max(1, TP + FN)), 4),
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in supported])), 4) if supported else None,
        "per_class": per_class,
        "confusion": confusion,
    }
    return out


def to_markdown(name: str, res: dict) -> str:
    lines = [f"# Evaluation: {name}", "", f"Gold items scored: {res.get('n_items', 0)} (of {res.get('n_gold', 0)} labeled)", ""]
    if "linking" in res:
        lk = res["linking"]
        lines += ["## Entity linking", "", f"- precision: {lk['precision']}", f"- missed entities: {lk['missed_entities']}", ""]
    if "sentiment" in res:
        lines += ["## Sentiment", ""] + [f"- {k}: {v}" for k, v in res["sentiment"].items()] + [""]
    if "events" in res:
        ev = res["events"]
        lines += ["## Events", "", f"- primary accuracy: {ev['primary_accuracy']}", f"- micro F1: {ev['micro_f1']}",
                  f"- macro F1: {ev['macro_f1']}", "", "| class | P | R | F1 | support |", "|---|---|---|---|---|"]
        for c, m in ev["per_class"].items():
            if m["support"]:
                lines.append(f"| {c} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")
    return "\n".join(lines) + "\n"
