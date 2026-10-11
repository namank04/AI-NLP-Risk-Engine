"""Paths, settings and the YAML configs (universe, taxonomy)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # large downloads, gitignored
REPLAY_DIR = DATA_DIR / "replay"      # recorded streams for deterministic demos
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
DB_PATH = Path(os.environ.get("RISKENGINE_DB", DATA_DIR / "riskengine.duckdb"))


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    aliases: tuple[str, ...] = ()
    ambiguous_aliases: tuple[str, ...] = ()
    products: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventClass:
    key: str
    label: str
    hypothesis: str
    description: str
    subtypes: tuple[str, ...]
    base_impact: int
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Universe:
    companies: tuple[Company, ...]
    market_proxy: str
    sector_etfs: dict[str, str] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return [c.ticker for c in self.companies]

    def by_ticker(self, ticker: str) -> Company:
        for c in self.companies:
            if c.ticker == ticker:
                return c
        raise KeyError(ticker)


@lru_cache
def universe() -> Universe:
    raw = yaml.safe_load((CONFIG_DIR / "universe.yaml").read_text(encoding="utf-8"))
    companies = tuple(
        Company(
            ticker=c["ticker"],
            name=c["name"],
            sector=c["sector"],
            aliases=tuple(c.get("aliases", [])),
            ambiguous_aliases=tuple(c.get("ambiguous_aliases", [])),
            products=tuple(c.get("products", [])),
        )
        for c in raw["companies"]
    )
    bm = raw.get("benchmarks", {})
    return Universe(companies, bm.get("market", "SPY"), dict(bm.get("sector_etfs", {})))


@lru_cache
def taxonomy() -> dict[str, EventClass]:
    raw = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    return {
        key: EventClass(
            key=key,
            label=v["label"],
            hypothesis=v["hypothesis"],
            description=v["description"],
            subtypes=tuple(v.get("subtypes", [])),
            base_impact=int(v["base_impact"]),
            channels=tuple(v.get("channels", [])),
        )
        for key, v in raw["classes"].items()
    }


@lru_cache
def edgar_item_map() -> dict[str, str]:
    raw = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    return dict(raw.get("edgar_8k_items", {}))
