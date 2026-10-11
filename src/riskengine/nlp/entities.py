"""Entity linking (text -> tickers) and scope extraction (countries, commodities...).

Evidence strength, strongest first:
  1.0  cashtag ($AAPL) or a ticker the source attached itself
  0.9  unambiguous alias ("Microsoft", "JPMorgan")
  0.7  ambiguous alias ("Apple", "Meta") written in title case AND near a finance cue
  0.5  product mention ("iPhone", "AWS"), only when the company is not otherwise named
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from riskengine.config import Company, universe
from riskengine.schema import MARKET_ENTITY, Scope

FINANCE_CUES = re.compile(
    r"\b(shares?|stocks?|earnings|revenue|profit|quarter|investors?|analysts?|CEO|CFO|Inc\.?|Corp\.?|"
    r"company|firm|market|valuation|guidance|dividend|buyback|sales|deal|acquisition|antitrust|lawsuit|"
    r"price target|downgrade|upgrade|NYSE|Nasdaq|S&P|traders?)\b",
    re.I,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"'])")
# Contrast markers that separate clauses about different companies:
# "Apple beats estimates, while Tesla misses" -> two clauses.
CLAUSE_SPLIT = re.compile(r"\s*(?:,\s*|\s)(?:while|whereas|but|although|though|even as|meanwhile)\s+|\s*;\s*|\s+-{1,2}\s+",
                          re.I)

MACRO_STRONG = re.compile(
    r"\b(Federal Reserve|the Fed|FOMC|central bank|ECB|Bank of England|Bank of Japan|RBI|PBOC|interest rates?|"
    r"rate (hike|cut)s?|inflation|CPI|PCE|recession|GDP|unemployment rate|jobs report|payrolls|Treasury yields?|"
    r"bond yields?|tariffs?|trade war|OPEC|oil prices?|crude|stock market|Wall Street|S&P 500|"
    r"Dow Jones|Nasdaq Composite|stocks|equities|sovereign debt|debt ceiling|credit rating|banking crisis|"
    r"bank run|liquidity crisis|bond market|currency|devaluation|economy|economic)\b",
    re.I,
)
GEO_WEAK = re.compile(r"\b(invasion|invades?|war|ceasefire|missiles?|airstrikes?|troops|military|"
                      r"coup|geopolitic\w*|blockade|nuclear|sanctions?|embargo)\b", re.I)
ECON_COCUE = re.compile(r"\b(oil|gas|fuel|diesel|crude|energy|markets?|stocks?|shares|prices?|economy|economic|"
                        r"trade|exports?|imports?|supply|investors?|bonds?|currency|banks?|financial|shipping|"
                        r"vessels?|tankers?|refiner\w*|chips?|semiconductors?)\b", re.I)


def is_macro(text: str) -> bool:
    return bool(MACRO_STRONG.search(text) or (GEO_WEAK.search(text) and ECON_COCUE.search(text)))


MACRO_CUES = re.compile(MACRO_STRONG.pattern + "|" + GEO_WEAK.pattern, re.I)   # sentence selection only

COUNTRIES = {
    "US": ["United States", "U.S.", "US", "America", "American", "Washington"],
    "CN": ["China", "Chinese", "Beijing"],
    "RU": ["Russia", "Russian", "Moscow", "Kremlin"],
    "UA": ["Ukraine", "Ukrainian", "Kyiv"],
    "IR": ["Iran", "Iranian", "Tehran"],
    "IL": ["Israel", "Israeli"],
    "SA": ["Saudi Arabia", "Saudi"],
    "TW": ["Taiwan", "Taiwanese"],
    "JP": ["Japan", "Japanese", "Tokyo"],
    "IN": ["India", "Indian", "New Delhi"],
    "GB": ["United Kingdom", "UK", "Britain", "British", "London"],
    "DE": ["Germany", "German", "Berlin"],
    "FR": ["France", "French", "Paris"],
    "EU": ["European Union", "EU", "Eurozone", "Euro area", "Brussels"],
    "KR": ["South Korea", "Korean", "Seoul"],
    "MX": ["Mexico", "Mexican"],
    "CA": ["Canada", "Canadian"],
    "BR": ["Brazil", "Brazilian"],
    "VE": ["Venezuela", "Venezuelan"],
    "KP": ["North Korea"],
}
COMMODITIES = {
    "crude_oil": ["oil", "crude", "Brent", "WTI", "OPEC"],
    "natural_gas": ["natural gas", "LNG"],
    "gold": ["gold"],
    "copper": ["copper"],
    "wheat": ["wheat", "grain"],
    "semiconductors": ["chips", "semiconductors?", "rare earths?"],
}
CURRENCIES = {
    "USD": ["dollar", "greenback", "USD"],
    "EUR": ["euro", "EUR"],
    "JPY": ["yen", "JPY"],
    "CNY": ["yuan", "renminbi", "CNY"],
    "GBP": ["pound sterling", "sterling", "GBP"],
    "RUB": ["ruble", "rouble", "RUB"],
    "INR": ["rupee", "INR"],
}
SECTOR_CUES = {
    "Financials": ["banks?", "banking", "lenders?", "insurers?", "credit card"],
    "Energy": ["oil", "gas", "energy", "refiners?", "drilling", "OPEC"],
    "Information Technology": ["tech", "software", "semiconductors?", "chips?", "cloud", "AI"],
    "Health Care": ["pharma\\w*", "drugs?", "biotech", "health ?care", "FDA", "vaccines?"],
    "Industrials": ["airlines?", "aerospace", "defen[cs]e", "manufactur\\w+", "railroads?"],
    "Consumer Discretionary": ["retail\\w*", "automakers?", "cars?", "electric vehicles?|EVs?"],
    "Consumer Staples": ["consumer staples", "beverages?", "groceries", "food prices"],
    "Utilities": ["utilit(y|ies)", "power grid", "electricity"],
    "Real Estate": ["real estate", "property", "REITs?", "housing", "mortgages?"],
    "Materials": ["chemicals?", "mining", "steel", "metals?"],
    "Communication Services": ["social media", "telecom\\w*", "streaming", "advertising"],
}


@dataclass
class EntityMention:
    entity: str                      # ticker or MARKET_ENTITY
    confidence: float
    evidence: str                    # how it was linked: cashtag/alias/ambiguous/product/source/macro
    sentences: list[str] = field(default_factory=list)


def _alias_regex(alias: str, case_sensitive: bool) -> re.Pattern:
    """Word-bounded alias pattern; a "re:" prefix means the alias is already a regex."""
    flags = 0 if case_sensitive else re.I
    body = alias[3:] if alias.startswith("re:") else re.escape(alias)
    return re.compile(r"(?<![\w$])" + body + r"(?!\w)", flags)


def _gazetteer(spec: dict[str, list[str]]) -> dict[str, re.Pattern]:
    out = {}
    for key, terms in spec.items():
        cs = [t for t in terms if t.isupper() and len(t) <= 4]       # short acronyms: exact case
        ci = [t for t in terms if t not in cs]
        parts = []
        # lookarounds rather than \b so that terms ending in punctuation ("U.S.") still match
        if ci:
            parts.append(r"(?i:(?<!\w)(?:" + "|".join(t if "\\" in t or "?" in t or "|" in t else re.escape(t)
                                                        for t in ci) + r")(?!\w))")
        if cs:
            parts.append(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in cs) + r")(?!\w)")
        out[key] = re.compile("|".join(parts))
    return out


@dataclass
class _CompanyPatterns:
    company: Company
    cashtags: list[re.Pattern]
    strong: list[re.Pattern]
    ambiguous: list[re.Pattern]
    products: list[re.Pattern]


class EntityLinker:
    def __init__(self, companies: tuple[Company, ...] | None = None):
        companies = companies or universe().companies
        self.patterns: list[_CompanyPatterns] = []
        for c in companies:
            tickers = {c.ticker} | {a for a in c.aliases if a.isupper() and len(a) <= 5}
            self.patterns.append(_CompanyPatterns(
                company=c,
                cashtags=[re.compile(r"\$" + re.escape(t) + r"\b", re.I) for t in tickers],
                strong=[_alias_regex(a, case_sensitive=a.isupper()) for a in c.aliases],
                ambiguous=[_alias_regex(a, case_sensitive=True) for a in c.ambiguous_aliases],
                products=[_alias_regex(p, case_sensitive=False) for p in c.products],
            ))
        self.countries = _gazetteer(COUNTRIES)
        self.commodities = _gazetteer(COMMODITIES)
        self.currencies = _gazetteer(CURRENCIES)
        self.sectors = _gazetteer(SECTOR_CUES)

    @staticmethod
    def sentences(text: str) -> list[str]:
        return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]

    def link(self, text: str, tickers_hint: list[str] | None = None, require_context: bool = False,
             context_hint: bool = False) -> list[EntityMention]:
        """Link text to tickers (or MARKET).

        require_context: the text comes from a general-news firehose, so a company
            name alone is not enough ("Tesla Tower", "deal at Amazon"); a finance cue
            in the text or an upstream economic tag (context_hint) is required.
        """
        sents = self.sentences(text)
        has_finance_cue = bool(FINANCE_CUES.search(text))
        found: dict[str, EntityMention] = {}

        def add(ticker: str, conf: float, how: str, pats: list[re.Pattern]):
            hit_sents = [s for s in sents if any(p.search(s) for p in pats)] if pats else []
            cur = found.get(ticker)
            if cur is None or conf > cur.confidence:
                found[ticker] = EntityMention(ticker, conf, how, hit_sents or (cur.sentences if cur else []))
            elif hit_sents:
                cur.sentences = sorted(set(cur.sentences) | set(hit_sents), key=sents.index)

        for cp in self.patterns:
            t = cp.company.ticker
            if any(p.search(text) for p in cp.cashtags):
                add(t, 1.0, "cashtag", cp.cashtags + cp.strong)
            if any(p.search(text) for p in cp.strong):
                add(t, 0.9, "alias", cp.strong)
            if cp.ambiguous and has_finance_cue and any(p.search(text) for p in cp.ambiguous):
                add(t, 0.7, "ambiguous_alias", cp.ambiguous)
        for cp in self.patterns:
            t = cp.company.ticker
            if t not in found and cp.products and any(p.search(text) for p in cp.products):
                add(t, 0.5, "product", cp.products)

        for t in tickers_hint or []:
            if t in found:
                found[t].confidence = 1.0
            else:
                cp = next((p for p in self.patterns if p.company.ticker == t), None)
                pats = (cp.cashtags + cp.strong + cp.ambiguous + cp.products) if cp else []
                add(t, 1.0, "source", pats)

        if require_context and not (has_finance_cue or context_hint or is_macro(text)):
            found = {t: m for t, m in found.items() if m.evidence in ("cashtag", "source")}

        if len(found) > 1:
            self._target_clauses(found)

        if not found and is_macro(text):
            found[MARKET_ENTITY] = EntityMention(MARKET_ENTITY, 0.8, "macro",
                                                 [s for s in sents if MACRO_CUES.search(s)])

        for m in found.values():
            if not m.sentences:
                m.sentences = sents[:2]       # fall back to the lead (headline + first sentence)
        return sorted(found.values(), key=lambda m: -m.confidence)

    def _patterns_for(self, ticker: str) -> list[re.Pattern]:
        cp = next((p for p in self.patterns if p.company.ticker == ticker), None)
        return (cp.cashtags + cp.strong + cp.ambiguous + cp.products) if cp else []

    def _target_clauses(self, found: dict[str, EntityMention]) -> None:
        """Where one sentence names several entities, keep only the clauses about each."""
        pats = {t: self._patterns_for(t) for t in found}
        for t, m in found.items():
            refined = []
            for s in m.sentences:
                others = [o for o in found if o != t and any(p.search(s) for p in pats[o])]
                if not others:
                    refined.append(s)
                    continue
                clauses = [c for c in CLAUSE_SPLIT.split(s) if c and c.strip()]
                mine = [c.strip() for c in clauses if any(p.search(c) for p in pats[t])
                        and not any(p.search(c) for o in others for p in pats[o])]
                refined.extend(mine or [s])     # no clean split: keep the shared sentence
            m.sentences = refined

    def scope(self, text: str) -> Scope:
        def hits(gaz: dict[str, re.Pattern]) -> list[str]:
            return [k for k, p in gaz.items() if p.search(text)]

        return Scope(countries=hits(self.countries), sectors=hits(self.sectors),
                     commodities=hits(self.commodities), currencies=hits(self.currencies))


@lru_cache
def default_linker() -> EntityLinker:
    return EntityLinker()
