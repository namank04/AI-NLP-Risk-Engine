"""Entity-level sentiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskengine.nlp.models import batched_logits, load_classifier
from riskengine.schema import Sentiment

DEFAULT_NEWS_MODEL = "ProsusAI/finbert"
DEFAULT_SOCIAL_MODEL = "StephanAkkerman/FinTwitBERT-sentiment"

_LABEL_ALIASES = {
    "positive": "positive", "bullish": "positive", "pos": "positive",
    "negative": "negative", "bearish": "negative", "neg": "negative",
    "neutral": "neutral", "neu": "neutral",
}


@dataclass
class SentenceScores:
    pos: np.ndarray
    neu: np.ndarray
    neg: np.ndarray


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_NEWS_MODEL, batch_size: int = 32, max_length: int = 128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tok, self.model = load_classifier(model_name)
        id2label = {int(k): _LABEL_ALIASES.get(v.lower(), v.lower()) for k, v in self.model.config.id2label.items()}
        self.idx = {lab: i for i, lab in id2label.items()}
        missing = {"positive", "negative"} - set(self.idx)
        if missing:
            raise ValueError(f"{model_name} has no {missing} label: {id2label}")

    def score_sentences(self, sentences: list[str]) -> SentenceScores:
        import torch

        logits = batched_logits(self.model, self.tok, sentences, batch_size=self.batch_size,
                                max_length=self.max_length)
        p = torch.softmax(logits, dim=-1).numpy() if len(sentences) else np.zeros((0, len(self.idx)))
        neu = p[:, self.idx["neutral"]] if "neutral" in self.idx else np.zeros(len(p))
        return SentenceScores(pos=p[:, self.idx["positive"]], neu=neu, neg=p[:, self.idx["negative"]])

    def entity_sentiment(self, groups: list[list[str]]) -> list[Sentiment]:
        """One Sentiment per group of sentences (all sentences mentioning one entity)."""
        flat = [s for g in groups for s in g]
        scores = self.score_sentences(flat)
        out, k = [], 0
        for g in groups:
            n = len(g)
            if n == 0:
                out.append(Sentiment(score=0.0, confidence=0.0, label="neutral"))
                continue
            pos, neu, neg = scores.pos[k:k + n], scores.neu[k:k + n], scores.neg[k:k + n]
            k += n
            # weight sentences by how opinionated they are, so one strong sentence
            # is not washed out by neutral boilerplate around it
            w = 1.0 - neu + 1e-3
            p_pos, p_neg, p_neu = (np.average(x, weights=w) for x in (pos, neg, neu))
            score = float(np.clip(p_pos - p_neg, -1, 1))
            label = max((("positive", p_pos), ("neutral", p_neu), ("negative", p_neg)), key=lambda t: t[1])[0]
            conf = float(max(p_pos, p_neu, p_neg))
            out.append(Sentiment(score=round(score, 4), confidence=round(conf, 4), label=label))
        return out
