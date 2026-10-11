"""Event classification baseline (multi-label over config/taxonomy.yaml)."""

from __future__ import annotations

import json
import re

import numpy as np

from riskengine.config import ROOT, edgar_item_map, taxonomy
from riskengine.nlp.models import batched_logits, load_classifier
from riskengine.schema import EventClassification

ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# (class, subtype, pattern). Ordered: first match per class sets the subtype.
LEXICON: list[tuple[str, str, str]] = [
    ("credit_event", "bankruptcy", r"\b(bankrupt\w*|chapter 11|chapter 7|insolven\w+|receivership)\b"),
    ("credit_event", "failure_to_pay", r"\b(default(s|ed)? on|missed (a |an )?(coupon|interest|debt) payment|failure to pay)\b"),
    ("credit_event", "restructuring", r"\b(debt restructuring|restructur\w+ (its|of) debt|distressed exchange|haircut)\b"),
    # "S&P" alone is usually the stock index, so the agency must be named in full
    ("credit_event", "rating_action", r"\b(credit ratings?|(downgrad\w+|cut\w*|lower\w*)\b.{0,40}\b(credit|ratings?|to junk|notch\w*|outlook|Baa\d|Ba\d|BBB-?)\b|(Moody'?s|Fitch|S&P Global Ratings|rating agenc\w+)\b.{0,40}\b(downgrad\w+|cut\w*)|outlook (to|cut to) negative|junk status|placed on (credit ?)?watch)"),
    ("credit_event", "covenant_breach", r"\b(covenant (breach|waiver|violation))\b"),
    ("credit_event", "liquidity_stress", r"\b(liquidity (crunch|crisis|squeeze)|bank run|going concern|deposit (outflows|flight))\b"),
    ("merger_acquisition", "terminated", r"\b(terminat\w+|scrap\w*|abandon\w*|call(s|ed)? off) (the |its )?(deal|merger|acquisition|takeover)\b"),
    ("merger_acquisition", "completed", r"\b(complet\w+|clos(es|ed)) (the |its )?(acquisition|merger|purchase)\b"),
    ("merger_acquisition", "announced", r"\b(to acquire|to buy|agrees? to (buy|acquire)|acquisition of|merger|takeover|buyout|tender offer|takeover bid)\b"),
    ("merger_acquisition", "divestiture", r"\b(divest\w*|spin[- ]?off|sell(s|ing)? (its )?(unit|division|stake|business))\b"),
    ("earnings", "beat", r"\b(beats?|tops?|exceed\w*|surpass\w*) (\w+ ){0,3}(estimates|expectations|forecasts|consensus)\b"),
    ("earnings", "miss", r"\b(miss(es|ed)?|falls? short of|below) (\w+ ){0,3}(estimates|expectations|forecasts|consensus)\b"),
    ("earnings", "guidance_cut", r"\b(cuts?|lowers?|slash\w*|reduc\w+) (\w+ ){0,3}(guidance|outlook|forecast)\b"),
    ("earnings", "guidance_raise", r"\b(raises?|lifts?|boosts?|increas\w+) (\w+ ){0,3}(guidance|outlook|forecast)\b"),
    ("earnings", "analyst_rating", r"\b(price target|upgrades?|initiates? coverage|maintains (buy|hold|sell|neutral|overweight|underweight|outperform))\b"),
    ("earnings", "results", r"\b(quarterly (results|earnings|profit|revenue)|Q[1-4] (results|earnings)|EPS|earnings (call|report|season))\b"),
    ("management_change", "departure", r"\b(CEO|CFO|chief executive|chairman|chief financial officer)\b.{0,40}\b(resign\w*|step(s|ping)? down|depart\w*|ousted|fired|retir\w+)\b"),
    ("management_change", "appointment", r"\b(appoint\w*|names?|hires?|taps?)\b.{0,40}\b(CEO|CFO|chief executive|chairman|president)\b"),
    ("regulatory_legal", "antitrust", r"\b(antitrust|anti-trust|monopol\w+|competition (regulator|authority|probe))\b"),
    ("regulatory_legal", "fine_settlement", r"\b(fined|fines?|penalt(y|ies)|settle(s|d|ment))\b"),
    ("regulatory_legal", "lawsuit", r"\b(lawsuit|sues?|sued|class action|litigation|court rul\w+|verdict)\b"),
    ("regulatory_legal", "investigation", r"\b(probe|investigat\w+|subpoena|SEC charges|DOJ|FTC|inquiry)\b"),
    ("operational", "cyber_incident", r"\b(cyber ?attack|data breach|hack(ed|ers?)|ransomware|outage)\b"),
    ("operational", "layoffs", r"\b(layoffs?|lay(s|ing)? off|job cuts?|cut(s|ting)? \d[\d,]* jobs)\b"),
    ("operational", "outage_accident", r"\b(recall\w*|grounded|crash(ed|es)?|explosion|fire at|plant shutdown|halts? production)\b"),
    ("operational", "supply_chain", r"\b(supply[- ]chain|shortages?|chip shortage|disruption\w*)\b"),
    ("operational", "labour_action", r"\b(strikes?|walkout|union (vote|deal))\b"),
    ("product_launch", "approval", r"\b(FDA approv\w+|wins? approval|regulatory approval|clearance for)\b"),
    ("product_launch", "launch", r"\b(launch\w*|unveil\w*|introduc\w+|rolls? out|debut\w*|releases? (a |its )?new)\b"),
    ("macroeconomic", "central_bank", r"\b(Federal Reserve|the Fed|FOMC|ECB|central bank|rate (hike|cut)s?|interest rates?|Powell|monetary policy)\b"),
    ("macroeconomic", "inflation", r"\b(inflation|CPI|consumer prices|PCE)\b"),
    ("macroeconomic", "employment", r"\b(jobs report|payrolls|unemployment|jobless claims)\b"),
    ("macroeconomic", "growth", r"\b(GDP|recession|economic growth|PMI|slowdown)\b"),
    ("macroeconomic", "sovereign_debt", r"\b(Treasury yields?|bond yields?|debt ceiling|sovereign (debt|default)|government shutdown)\b"),
    ("geopolitical", "sanctions", r"\b(sanction\w*|export (controls?|ban)|embargo|blacklist\w*|entity list)\b"),
    ("geopolitical", "trade_policy", r"\b(tariffs?|trade war|trade deal|import duties)\b"),
    ("geopolitical", "armed_conflict", r"\b(war|invasion|invade\w*|military|missiles?|airstrikes?|troops|ceasefire|attack on)\b"),
    ("geopolitical", "election_political", r"\b(election|coup|impeach\w*|political crisis)\b"),
    ("geopolitical", "diplomatic_tension", r"\b(geopolitic\w*|diplomatic|tensions? (with|between))\b"),
]
LEXICON_P = 0.85
EDGAR_P = 0.95


STUDENT_DIR = ROOT / "models" / "students" / "event"


class EventClassifier:
    """student=None uses the fine-tuned student when it has been adopted (beat the baseline)."""

    def __init__(self, use_zero_shot: bool = True, threshold: float = 0.5, batch_size: int = 16,
                 student: bool | None = None):
        self.tax = taxonomy()
        self.classes = [k for k in self.tax if k != "none"]
        self.threshold = threshold
        self.batch_size = batch_size
        self.lexicon = [(c, st, re.compile(p, re.I)) for c, st, p in LEXICON]
        self.conflict = next(pat for c, st, pat in self.lexicon if st == "armed_conflict")
        self.student = (STUDENT_DIR / "ADOPTED").exists() if student is None else student
        self.covered: list[str] = []
        if self.student:
            self.s_tok, self.s_model = load_classifier(str(STUDENT_DIR))
            self.s_classes = [self.s_model.config.id2label[i] for i in range(self.s_model.config.num_labels)]
            cov = STUDENT_DIR / "coverage.json"
            self.covered = json.loads(cov.read_text())["covered"] if cov.exists() else list(self.s_classes)
        # zero-shot still scores the classes the student never saw positives for (hybrid)
        self.use_zero_shot = use_zero_shot and (not self.student or set(self.classes) - set(self.covered))
        if self.use_zero_shot:
            self.tok, self.model = load_classifier(ZERO_SHOT_MODEL)
            labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
            self.entail_idx = labels.get("entailment", 0)

    def _lexicon(self, text: str) -> dict[str, tuple[float, str]]:
        out: dict[str, tuple[float, str]] = {}
        for cls, subtype, pat in self.lexicon:
            if cls not in out and pat.search(text):
                out[cls] = (LEXICON_P, subtype)
        return out

    def _zero_shot(self, texts: list[str]) -> np.ndarray:
        """P(entailment) for every (text, class hypothesis) pair -> [n_texts, n_classes]."""
        import torch

        hyps = [self.tax[c].hypothesis for c in self.classes]
        premises = [t for t in texts for _ in hyps]
        hypotheses = [h for _ in texts for h in hyps]
        logits = batched_logits(self.model, self.tok, premises, hypotheses, batch_size=self.batch_size * 4,
                                max_length=128, truncation="only_first")
        if logits.shape[-1] == 2:   # entailment vs not_entailment
            p = torch.softmax(logits, -1)[:, self.entail_idx]
        else:                        # entailment / neutral / contradiction: ignore neutral
            ent = logits[:, self.entail_idx]
            contra = logits[:, [j for j in range(logits.shape[-1]) if j != self.entail_idx][-1]]
            p = torch.sigmoid(ent - contra)
        return p.numpy().reshape(len(texts), len(hyps))

    def _student(self, texts: list[str]) -> np.ndarray:
        import torch

        logits = batched_logits(self.s_model, self.s_tok, texts, batch_size=self.batch_size * 4, max_length=96)
        p = torch.sigmoid(logits).numpy()
        idx = [self.s_classes.index(c) for c in self.classes]
        return p[:, idx]

    def classify(self, texts: list[str], labels_hints: list[dict[str, str]] | None = None) -> list[EventClassification]:
        labels_hints = labels_hints or [{} for _ in texts]
        if not texts:
            zs = np.zeros((0, len(self.classes)))
        else:
            zs = self._zero_shot(texts) if self.use_zero_shot else np.zeros((len(texts), len(self.classes)))
            if self.student:
                st = self._student(texts)
                for j, c in enumerate(self.classes):
                    if c in self.covered:
                        zs[:, j] = st[:, j]
        item_map = edgar_item_map()
        out = []
        for i, text in enumerate(texts):
            probs = {c: float(zs[i, j]) for j, c in enumerate(self.classes)}
            subtypes: dict[str, str] = {}
            lexicon_hits = self._lexicon(text)
            for cls, (p, st) in lexicon_hits.items():
                if p > probs[cls]:
                    probs[cls] = p
                subtypes.setdefault(cls, st)
            for code in filter(None, labels_hints[i].get("8k_items", "").split(",")):
                cls = item_map.get(code.strip())
                if cls:
                    probs[cls] = max(probs[cls], EDGAR_P)
            labels = [c for c, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= self.threshold]
            # explicit evidence of armed conflict or credit distress dominates the model's ranking
            if "geopolitical" in labels and self.conflict.search(text):
                subtypes["geopolitical"] = "armed_conflict"
                labels.remove("geopolitical")
                labels.insert(0, "geopolitical")
            if "credit_event" in lexicon_hits and "credit_event" in labels:
                labels.remove("credit_event")
                labels.insert(0, "credit_event")
            probs["none"] = float(1.0 - max(probs.values())) if probs else 1.0
            primary = labels[0] if labels else "none"
            out.append(EventClassification(
                primary=primary,
                labels=labels or ["none"],
                probs={k: round(v, 4) for k, v in probs.items()},
                subtype=subtypes.get(primary),
            ))
        return out
