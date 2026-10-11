"""Impact score (1-10): predicted market impact severity."""

from __future__ import annotations

import math

from riskengine.config import taxonomy
from riskengine.schema import EventClassification, Impact, Sentiment


def rule_impact(event: EventClassification, sentiment: Sentiment, novelty: float, n_sources: int,
                source_weight: float) -> Impact:
    tax = taxonomy()
    base = tax[event.primary].base_impact
    # blend in other confidently-detected classes (a sanctions story that is also a credit event)
    secondary = [tax[c].base_impact for c in event.labels[1:] if c in tax]
    if secondary:
        base = max(base, 0.7 * base + 0.3 * max(secondary))
    tone = 0.6 + 0.8 * abs(sentiment.score)                 # 0.6 .. 1.4
    fresh = 0.6 + 0.4 * novelty                             # 0.6 .. 1.0
    # coverage breadth helps, but saturates: 200 outlets covering a political story
    # is attention, not market impact
    breadth = min(1.35, 1.0 + 0.07 * math.log2(max(1, n_sources)))
    cred = 0.7 + 0.3 * source_weight                        # 0.7 .. 1.0
    raw = base * tone * fresh * breadth * cred
    score = int(min(10, max(1, round(raw))))
    return Impact(score=score, raw=round(raw, 3), confidence=0.3, method="rule_v0")
