"""In-process publish/subscribe for signals."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class Bus:
    def __init__(self):
        self._callbacks: dict[str, list[Callable[[Any], None]]] = {}
        self._queues: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, fn: Callable[[Any], None]) -> None:
        with self._lock:
            self._callbacks.setdefault(topic, []).append(fn)

    def open_queue(self, topic: str, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.setdefault(topic, []).append((asyncio.get_running_loop(), q))
        return q

    def close_queue(self, topic: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._queues[topic] = [(lp, qq) for lp, qq in self._queues.get(topic, []) if qq is not q]

    def publish(self, topic: str, msg: Any) -> None:
        with self._lock:
            callbacks = list(self._callbacks.get(topic, []))
            queues = list(self._queues.get(topic, []))
        for fn in callbacks:
            try:
                fn(msg)
            except Exception:
                log.exception("subscriber %s failed on topic %s", fn, topic)
        for loop, q in queues:
            loop.call_soon_threadsafe(_put_drop_oldest, q, msg)


def _put_drop_oldest(q: asyncio.Queue, msg: Any) -> None:
    # slow SSE clients lose the oldest messages rather than stalling the engine
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(msg)
