import time
from datetime import UTC, datetime

from riskengine.schema import RawItem
from riskengine.service import EngineService
from riskengine.store import SignalStore


class FakeSource:
    def __init__(self):
        self.calls = 0

    def fetch(self, since):
        self.calls += 1
        n = 2 if self.calls == 1 else 3        # second poll repeats both items and adds one
        return [RawItem(item_id=f"i{k}", source="gdelt", source_type="news", title=f"t{k}",
                        published_at=datetime.now(UTC)) for k in range(n)]


class FakeEngine:
    def __init__(self):
        self.seen = []

    def process(self, items):
        self.seen += [i.item_id for i in items]
        return []


def test_live_polls_and_ingests_each_item_once():
    eng, src = FakeEngine(), FakeSource()
    svc = EngineService(store=SignalStore(":memory:"), engine=eng)
    svc.index.as_of = lambda when: None
    svc.start_live(poll_s=0.05, sources=[src])
    deadline = time.time() + 5
    while src.calls < 3 and time.time() < deadline:
        time.sleep(0.02)
    svc.stop_replay()
    assert eng.seen == ["i0", "i1", "i2"]
    assert svc.replay.name == "live" and svc.replay.processed == 3 and not svc.replay.running
