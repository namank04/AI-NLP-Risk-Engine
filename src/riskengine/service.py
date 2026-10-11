"""Runtime wiring: engine + store + bus + modules + replay/live runners.

    sources ──► EngineService.ingest() ──► RiskEngine ──► SignalStore
                                              │
                                              └──► Bus "signals" ──► Module A (IndexRebalancer)
                                                                 ├─► Module B (StressTrigger)
                                                                 └─► SSE clients (dashboard)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from riskengine.bus import Bus
from riskengine.config import REPLAY_DIR
from riskengine.modules.index.rebalancer import IndexRebalancer
from riskengine.modules.stress.trigger import StressTrigger
from riskengine.nlp.pipeline import EngineConfig, RiskEngine
from riskengine.schema import RawItem, Signal
from riskengine.store import SignalStore

log = logging.getLogger(__name__)

TOPIC_SIGNALS = "signals"
TOPIC_INDEX = "index"
TOPIC_STRESS = "stress"
TOPIC_STATUS = "status"
LIVE = "live"


def load_replay(name: str) -> list[RawItem]:
    df = pd.read_parquet(REPLAY_DIR / f"{name}.parquet")
    items = []
    for r in df.to_dict("records"):
        # parquet/pandas represent missing strings as NaN floats
        r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        r["tickers_hint"] = list(r["tickers_hint"]) if r.get("tickers_hint") is not None else []
        for k in ("labels_hint", "engagement"):
            v = r.get(k)
            if isinstance(v, str):
                v = json.loads(v)
            # older files stored structs: parquet unifies keys across rows and fills gaps with None
            r[k] = {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else {}
        items.append(RawItem(**{k: v for k, v in r.items() if k in RawItem.model_fields}))
    return sorted(items, key=lambda i: i.published_at)


def save_replay(name: str, items: list[RawItem]) -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([i.model_dump() for i in items])
    for k in ("labels_hint", "engagement"):     # JSON text: per-source keys differ
        df[k] = df[k].apply(json.dumps)
    df.to_parquet(REPLAY_DIR / f"{name}.parquet", index=False)


@dataclass
class ReplayStatus:
    running: bool = False
    name: str | None = None
    speed: float = 0.0
    sim_time: str | None = None
    processed: int = 0
    total: int = 0
    signals: int = 0


class EngineService:
    def __init__(self, store: SignalStore | None = None, engine: RiskEngine | None = None,
                 engine_cfg: EngineConfig | None = None):
        self.store = store or SignalStore()
        self.bus = Bus()
        self._engine = engine
        self._engine_cfg = engine_cfg
        self._engine_lock = threading.Lock()
        self.index = IndexRebalancer()
        self.stress = StressTrigger()
        self.bus.subscribe(TOPIC_SIGNALS, self.index.on_signal)
        self.bus.subscribe(TOPIC_SIGNALS, self.stress.on_signal)
        self.index.listeners.append(lambda snap: self.bus.publish(TOPIC_INDEX, snap.to_dict()))
        self.stress.listeners.append(lambda rec: self.bus.publish(TOPIC_STRESS, rec))
        self.replay = ReplayStatus()
        self._replay_thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def engine(self) -> RiskEngine:
        with self._engine_lock:
            if self._engine is None:
                log.info("loading NLP models ...")
                self._engine = RiskEngine(self._engine_cfg)
            return self._engine

    def ingest(self, items: list[RawItem]) -> list[Signal]:
        if not items:
            return []
        self.store.add_items(items)
        signals = self.engine.process(items)
        self.store.add_signals(signals)
        for s in sorted(signals, key=lambda s: s.published_at):
            self.bus.publish(TOPIC_SIGNALS, s)
        return signals

    def reset_state(self) -> None:
        """Fresh story clusters and module state, e.g. before re-running a replay."""
        from riskengine.nlp.dedup import StoryClusterer

        if self._engine is not None:
            self._engine.clusterer = StoryClusterer()
        self.index.reset()
        self.stress.runs.clear()
        self.stress._cooldown.clear()

    # ---- replay -------------------------------------------------------------
    def start_replay(self, name: str, speed: float = 600.0, start: datetime | None = None,
                     end: datetime | None = None, batch_window_s: float = 900.0) -> ReplayStatus:
        """Replay a recorded stream. speed = simulated seconds per wall-clock second."""
        self.stop_replay()
        items = load_replay(name)
        if start:
            items = [i for i in items if i.published_at >= start]
        if end:
            items = [i for i in items if i.published_at < end]
        if items:
            # benchmark weights and risk model as the market stood when the stream begins
            self.index.as_of(items[0].published_at)
        self.replay = ReplayStatus(running=True, name=name, speed=speed, total=len(items))
        self._stop.clear()
        self._replay_thread = threading.Thread(target=self._run_replay, args=(items, speed, batch_window_s),
                                               daemon=True)
        self._replay_thread.start()
        return self.replay

    def start_live(self, poll_s: float = 300.0, lookback: timedelta = timedelta(hours=1), sources=None) -> ReplayStatus:
        """Poll GDELT (15-minute files) and SEC EDGAR 8-Ks and push new items through the engine."""
        self.stop_replay()
        self.index.as_of(datetime.now(UTC))
        self.replay = ReplayStatus(running=True, name=LIVE, speed=1.0)
        self._stop.clear()
        self._replay_thread = threading.Thread(target=self._run_live, args=(poll_s, lookback, sources), daemon=True)
        self._replay_thread.start()
        return self.replay

    def _run_live(self, poll_s: float, lookback: timedelta, sources) -> None:
        if sources is None:
            from riskengine.ingest.edgar import EdgarSource
            from riskengine.ingest.gdelt import GdeltGkgSource

            sources = [GdeltGkgSource(), EdgarSource()]
        since = datetime.now(UTC) - lookback
        seen: set[str] = set()
        try:
            _ = self.engine
            while not self._stop.is_set():
                polled = datetime.now(UTC)
                items: list[RawItem] = []
                for src in sources:
                    try:
                        items += src.fetch(since)
                    except Exception:
                        log.exception("live fetch failed: %s", type(src).__name__)
                fresh = sorted((i for i in items if i.item_id not in seen), key=lambda i: i.published_at)
                seen.update(i.item_id for i in fresh)
                sigs = self.ingest(fresh)
                self.replay.processed += len(fresh)
                self.replay.total = self.replay.processed
                self.replay.signals += len(sigs)
                self.replay.sim_time = polled.isoformat()
                self.bus.publish(TOPIC_STATUS, self.replay.__dict__.copy())
                # GDELT posts each file ~15 minutes after its slot: overlap the window, dedupe by id
                since = polled - timedelta(minutes=45)
                self._stop.wait(poll_s)
        except Exception:
            log.exception("live feed failed")
        finally:
            self.replay.running = False
            self.bus.publish(TOPIC_STATUS, self.replay.__dict__.copy())

    def stop_replay(self) -> None:
        self._stop.set()
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=10)
        self.replay.running = False

    def _run_replay(self, items: list[RawItem], speed: float, window_s: float) -> None:
        try:
            _ = self.engine  # load models before the clock starts
            i = 0
            while i < len(items) and not self._stop.is_set():
                t0 = items[i].published_at
                batch = []
                while i < len(items) and (items[i].published_at - t0).total_seconds() <= window_s:
                    batch.append(items[i])
                    i += 1
                wall_start = time.monotonic()
                sigs = self.ingest(batch)
                self.replay.processed = i
                self.replay.signals += len(sigs)
                self.replay.sim_time = batch[-1].published_at.isoformat()
                self.bus.publish(TOPIC_STATUS, self.replay.__dict__.copy())
                if i < len(items) and speed > 0:
                    gap = (items[i].published_at - batch[-1].published_at).total_seconds() / speed
                    remaining = min(gap, 5.0) - (time.monotonic() - wall_start)   # cap idle gaps at 5 s
                    if remaining > 0:
                        self._stop.wait(remaining)
        except Exception:
            log.exception("replay failed")
        finally:
            self.replay.running = False
            self.bus.publish(TOPIC_STATUS, self.replay.__dict__.copy())
