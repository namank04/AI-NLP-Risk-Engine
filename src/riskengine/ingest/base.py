"""Shared plumbing for source adapters: polite HTTP with per-host rate limits."""

from __future__ import annotations

import abc
import threading
import time
from datetime import datetime

import httpx

from riskengine.schema import RawItem


class RateLimiter:
    """Minimum interval between calls; thread-safe."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class HttpClient:
    def __init__(self, min_interval_s: float, user_agent: str = "riskengine/0.1", timeout_s: float = 30.0,
                 retries: int = 3):
        self.limiter = RateLimiter(min_interval_s)
        self.retries = retries
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=timeout_s, follow_redirects=True)

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            self.limiter.wait()
            try:
                resp = self.client.get(url, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"status {resp.status_code}", request=resp.request, response=resp)
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt * self.limiter.min_interval_s)
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts") from last_exc


class Source(abc.ABC):
    name: str

    @abc.abstractmethod
    def fetch(self, since: datetime) -> list[RawItem]:
        """Return items published at or after `since` (UTC)."""
