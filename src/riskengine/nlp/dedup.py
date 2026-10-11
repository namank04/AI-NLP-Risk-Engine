"""Story clustering, novelty and source counting."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from datasketch import MinHash, MinHashLSH

from riskengine.schema import RawItem, stable_id

_PUBLISHER_SUFFIX = re.compile(r"\s+[-|–—]\s+[^-|–—]{2,40}$")   # "Headline - Reuters"
_NON_WORD = re.compile(r"[^a-z0-9$ ]+")


def normalize(text: str) -> str:
    text = _PUBLISHER_SUFFIX.sub("", text.strip())
    text = _NON_WORD.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    if len(text) <= k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


@dataclass
class _Cluster:
    cluster_id: str
    first_seen: datetime
    last_seen: datetime
    size: int = 0
    live_keys: int = 0
    publishers: set[str] = field(default_factory=set)


@dataclass
class ClusterResult:
    cluster_id: str
    novelty: float
    n_sources: int
    is_duplicate: bool


class StoryClusterer:
    def __init__(self, threshold: float = 0.55, num_perm: int = 64, window: timedelta = timedelta(hours=72)):
        self.threshold = threshold
        self.num_perm = num_perm
        self.window = window
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.keys: deque[tuple[datetime, str]] = deque()          # (time, item key) for expiry
        self.key_cluster: dict[str, str] = {}
        self.clusters: dict[str, _Cluster] = {}

    def _minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for s in _shingles(text):
            m.update(s.encode("utf-8"))
        return m

    def _expire(self, now: datetime) -> None:
        while self.keys and now - self.keys[0][0] > self.window:
            _, key = self.keys.popleft()
            self.lsh.remove(key)
            cid = self.key_cluster.pop(key)
            c = self.clusters[cid]
            c.live_keys -= 1
            if c.live_keys == 0:
                del self.clusters[cid]

    def assign(self, item: RawItem) -> ClusterResult:
        now = item.published_at
        self._expire(now)
        text = normalize(item.title or item.body[:300])
        if not text:
            text = item.item_id
        mh = self._minhash(text)
        key = item.item_id
        if key in self.key_cluster:                                   # exact re-delivery of same item
            c = self.clusters[self.key_cluster[key]]
            return ClusterResult(c.cluster_id, 0.0, len(c.publishers), True)

        candidates = [self.key_cluster[k] for k in self.lsh.query(mh) if k in self.key_cluster]
        if candidates:
            # attach to the oldest matching story
            cid = min(candidates, key=lambda c: self.clusters[c].first_seen)
            cluster = self.clusters[cid]
        else:
            cid = stable_id("cluster", key)
            cluster = self.clusters[cid] = _Cluster(cid, now, now)

        prior = cluster.size
        cluster.size += 1
        cluster.last_seen = max(cluster.last_seen, now)
        cluster.live_keys += 1
        cluster.publishers.add((item.publisher or item.author or item.source).lower())
        self.lsh.insert(key, mh)
        self.keys.append((now, key))
        self.key_cluster[key] = cid
        return ClusterResult(cid, novelty=1.0 / (1.0 + prior), n_sources=len(cluster.publishers),
                             is_duplicate=prior > 0)
