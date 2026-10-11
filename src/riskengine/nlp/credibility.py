"""Source credibility weights in [0, 1].

Tiering follows the usual newsroom hierarchy: regulatory filings and wire
services first, major financial press next, aggregators/blogs lower, and
anonymous social posts lowest unless they attract engagement.
"""

from __future__ import annotations

import math

from riskengine.schema import RawItem

TIER_1 = {"sec.gov", "sec edgar", "reuters.com", "reuters", "bloomberg.com", "bloomberg", "apnews.com",
          "wsj.com", "ft.com", "federalreserve.gov"}
TIER_2 = {"cnbc.com", "cnbc", "marketwatch.com", "barrons.com", "nytimes.com", "economist.com", "bbc.com",
          "bbc.co.uk", "theguardian.com", "fortune.com", "forbes.com", "businessinsider.com", "axios.com",
          "nasdaq.com", "finance.yahoo.com", "investing.com", "foxbusiness.com"}
TIER_3 = {"benzinga.com", "benzinga", "seekingalpha.com", "fool.com", "zacks.com", "zacks", "thestreet.com",
          "investorplace.com", "247wallst.com", "gurufocus.com", "insidermonkey.com", "benzinga insights",
          "benzinga newsdesk"}


def _domain(publisher: str) -> str:
    p = publisher.lower().strip()
    return p[4:] if p.startswith("www.") else p


def source_weight(item: RawItem) -> float:
    if item.source_type == "filing":
        return 1.0
    if item.source_type == "social":
        # base 0.3, up to ~0.7 for posts with real engagement
        eng = sum(item.engagement.values()) if item.engagement else 0.0
        return round(min(0.7, 0.3 + 0.08 * math.log1p(eng)), 3)
    pub = _domain(item.publisher or "")
    if pub in TIER_1:
        return 1.0
    if pub in TIER_2:
        return 0.85
    if pub in TIER_3:
        return 0.65
    return 0.5
