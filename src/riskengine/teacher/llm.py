"""LLM teacher over any OpenAI-compatible chat endpoint.

Works with a free hosted tier (default: Groq) or a local Ollama server; configure
with LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (see .env.example). No paid API is needed.

The teacher labels exactly what a human labels in the gold set (per-entity
relevance + 5-point sentiment, multi-label event classes, subtype), following
docs/labeling_guidelines.md, so its quality can be measured against the gold
set before its labels are used to train students.

Free tiers cap tokens per day, so `label_batch` packs several items into one
request (the instructions are sent once), and a daily-quota response stops the
run cleanly. Every label is cached in data/processed/teacher_cache.jsonl keyed by
(model, prompt version, item, pass), so runs resume where they stopped.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from riskengine.config import PROCESSED_DIR, taxonomy
from riskengine.schema import MARKET_ENTITY

PROMPT_VERSION = "v1"
CACHE = PROCESSED_DIR / "teacher_cache.jsonl"
NL = "\n"


class DailyLimitReached(RuntimeError):
    """The provider's daily request/token quota is exhausted."""


def _system_prompt() -> str:
    classes = NL.join(f"- {k}: {v.description}" + (f" Subtypes: {', '.join(v.subtypes)}." if v.subtypes else "")
                      for k, v in taxonomy().items())
    return (
        "You are a senior financial-news analyst labeling data for a risk engine.\n"
        f"For one piece of text and a list of candidate entities (stock tickers, or {MARKET_ENTITY} for broad\n"
        "US/global markets), return JSON only, with this exact shape:\n\n"
        '{"entities": [{"entity": "<ticker>", "relevant": true|false, "sentiment": -2|-1|0|1|2}],\n'
        ' "events": ["<class key>", ...],\n'
        ' "subtype": "<subtype key or null>"}\n\n'
        "Relevance: false when the link is wrong or incidental (e.g. 'Tesla Tower' is Nikola Tesla, a shopping\n"
        f"deal 'at Amazon', a company used only as a size comparison). {MARKET_ENTITY} is relevant only for news that can\n"
        "move broad US/global equity or credit prices (central banks, inflation, growth, rates, geopolitics,\n"
        "commodities, systemic credit); purely local news is not relevant.\n\n"
        "Sentiment is toward that entity from its investors' point of view: -2 clearly bad (default, junk downgrade,\n"
        "guidance cut, fraud probe), -1 mildly bad, 0 neutral/factual/mixed, +1 mildly good, +2 clearly good.\n"
        "Judge against expectations ('loss narrows' is positive; 'profit up but misses estimates' is negative).\n"
        f"For {MARKET_ENTITY}, think of broad equities and credit: hawkish surprises and hotter inflation are negative.\n"
        "Hype posts ('to the moon') are at most +1.\n\n"
        'Events: list every class that applies, most important first; use ["none"] only if nothing applies.\n'
        'Analyst stock upgrades/price targets are "earnings" (subtype analyst_rating); rating-agency credit actions\n'
        "are \"credit_event\" (subtype rating_action); 'stocks to buy' listicles are \"none\".\n"
        f"Classes:\n{classes}\n"
        "Return JSON only, no prose."
    )


@dataclass
class TeacherConfig:
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"))
    temperature: float = 0.0
    max_tokens: int = 400
    concurrency: int = 1
    # free tiers allow ~30 requests/minute; leave headroom
    min_interval_s: float = field(default_factory=lambda: float(os.environ.get("LLM_MIN_INTERVAL_S", "2.5")))
    timeout_s: float = 180.0


_JSON = re.compile(r"\{.*\}", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_response(text: str, entities: list[str]) -> dict | None:
    """Validate and normalise the model's JSON; None if unusable."""
    m = _JSON.search(_THINK.sub("", text))
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    valid = set(taxonomy())
    events = [e for e in obj.get("events", []) if e in valid] or ["none"]
    if "none" in events and len(events) > 1:
        events = [e for e in events if e != "none"]
    ents = {}
    for e in obj.get("entities", []):
        name = str(e.get("entity", "")).upper()
        if name in entities:
            try:
                s = int(round(float(e.get("sentiment", 0))))
            except (TypeError, ValueError):
                s = 0
            ents[name] = {"entity": name, "relevant": bool(e.get("relevant", True)), "sentiment": max(-2, min(2, s))}
    if not ents:
        return None
    sub = obj.get("subtype")
    subtypes = set(taxonomy()[events[0]].subtypes)
    return {"entities": [ents.get(n, {"entity": n, "relevant": True, "sentiment": 0}) for n in entities],
            "events": events, "subtype": sub if sub in subtypes else None}


def parse_batch(raw: str) -> dict[int, dict]:
    """Map item number -> item object from a batched JSON reply."""
    m = _JSON.search(_THINK.sub("", raw))
    if not m:
        return {}
    try:
        items = json.loads(m.group(0)).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        return {}
    out = {}
    for k, it in enumerate(items):
        if isinstance(it, dict):
            try:
                out[int(it.get("item", k + 1))] = it
            except (TypeError, ValueError):
                out[k + 1] = it
    return out


class Teacher:
    def __init__(self, cfg: TeacherConfig | None = None):
        self.cfg = cfg or TeacherConfig()
        self.client = httpx.Client(base_url=self.cfg.base_url.rstrip("/"), timeout=self.cfg.timeout_s,
                                   headers={"Authorization": f"Bearer {self.cfg.api_key}"})
        self.system = _system_prompt()
        self._lock = threading.Lock()
        self._last = 0.0
        self.cache: dict[str, dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def _key(self, item_id: str, pass_no: int) -> str:
        return f"{self.cfg.model}|{PROMPT_VERSION}|{item_id}|{pass_no}"

    def _throttle(self) -> None:
        if self.cfg.min_interval_s <= 0:
            return
        with self._lock:
            wait = self._last + self.cfg.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _chat(self, user: str, temperature: float, max_tokens: int | None = None) -> str:
        body = {"model": self.cfg.model, "temperature": temperature, "max_tokens": max_tokens or self.cfg.max_tokens,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        for attempt in range(6):
            self._throttle()
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.TransportError:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code == 429:
                msg = r.text.lower()
                # free tiers cap tokens/requests per day: stop cleanly, the cache resumes tomorrow
                if "per day" in msg or "tpd" in msg or "rpd" in msg:
                    raise DailyLimitReached(r.text[:300])
                wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 5 * 2 ** attempt)
                time.sleep(min(wait, 120))
                continue
            if r.status_code >= 500:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("LLM endpoint unavailable")

    def _store(self, key: str, item_id: str, pass_no: int, label: dict | None, raw: str) -> None:
        rec = {"key": key, "item_id": item_id, "pass": pass_no, "label": label, "raw": raw[:2000]}
        with self._lock:
            self.cache[key] = rec
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + NL)

    def label_one(self, item_id: str, text: str, entities: list[str], pass_no: int = 0) -> dict | None:
        key = self._key(item_id, pass_no)
        if key in self.cache:
            return self.cache[key]["label"]
        user = f"Candidate entities: {', '.join(entities)}{NL}{NL}Text:{NL}{text[:2500]}"
        # pass 0 is greedy; pass 1 samples, giving an independent second opinion for agreement filtering
        raw = self._chat(user, self.cfg.temperature if pass_no == 0 else 0.7)
        label = parse_response(raw, entities)
        self._store(key, item_id, pass_no, label, raw)
        return label

    def label_batch(self, rows: list[tuple[str, str, list[str]]], pass_no: int = 0,
                    max_chars: int = 900) -> list[dict | None]:
        """Label several items in one request (the instructions are sent once).

        Returns labels aligned with `rows`; cached items are not re-sent.
        """
        out: list[dict | None] = [None] * len(rows)
        todo = []
        for i, (item_id, _text, _ents) in enumerate(rows):
            key = self._key(item_id, pass_no)
            if key in self.cache:
                out[i] = self.cache[key]["label"]
            else:
                todo.append(i)
        if not todo:
            return out
        parts = [f"### item {n}{NL}Candidate entities: {', '.join(rows[i][2])}{NL}Text: {rows[i][1][:max_chars]}"
                 for n, i in enumerate(todo, 1)]
        user = ("Label each item below independently. Return JSON of the form "
                '{"items": [{"item": 1, "entities": [...], "events": [...], "subtype": ...}, ...]} '
                f"with exactly {len(todo)} entries, one per item, in order.{NL}{NL}" + (NL + NL).join(parts))
        raw = self._chat(user, self.cfg.temperature if pass_no == 0 else 0.7, max_tokens=120 * len(todo) + 100)
        items = parse_batch(raw)
        for n, i in enumerate(todo, 1):
            item_id, _, ents = rows[i]
            it = items.get(n)
            label = parse_response(json.dumps(it), ents) if it else None
            self._store(self._key(item_id, pass_no), item_id, pass_no, label, json.dumps(it) if it else raw)
            out[i] = label
        return out

    def label_many(self, rows: list[tuple[str, str, list[str]]], pass_no: int = 0, progress: bool = True) -> list:
        done = [0]

        def run(row):
            out = None
            try:
                out = self.label_one(*row, pass_no=pass_no)
            except DailyLimitReached:
                raise
            except RuntimeError:
                pass
            done[0] += 1
            if progress and done[0] % 10 == 0:
                print(f"\r  teacher pass {pass_no}: {done[0]}/{len(rows)}", end="", flush=True)
            return out

        with ThreadPoolExecutor(self.cfg.concurrency) as pool:
            out = list(pool.map(run, rows))
        if progress:
            print()
        return out

    def ping(self) -> str:
        """Cheap connectivity check; returns the model's reply or raises."""
        return self._chat('Reply with {"ok": true}', 0.0)
