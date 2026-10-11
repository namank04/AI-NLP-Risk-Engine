from datetime import UTC, datetime, timedelta

from riskengine.nlp.dedup import StoryClusterer, normalize
from riskengine.schema import RawItem

T0 = datetime(2024, 3, 1, 12, tzinfo=UTC)


def item(i, title, publisher, minutes=0):
    return RawItem(item_id=f"id{i}", source="test", source_type="news", title=title, publisher=publisher,
                   published_at=T0 + timedelta(minutes=minutes))


def test_normalize_strips_publisher_suffix_and_punct():
    assert normalize("Exxon to buy Pioneer for $60 billion - Reuters") == "exxon to buy pioneer for $60 billion"


def test_syndicated_copies_join_one_cluster():
    c = StoryClusterer()
    a = c.assign(item(1, "Exxon to buy Pioneer Natural Resources for $60 billion", "reuters.com"))
    b = c.assign(item(2, "Exxon to buy Pioneer Natural Resources for $60 billion - CNBC", "cnbc.com", 5))
    d = c.assign(item(3, "Exxon to buy Pioneer Natural Resources for $60bn", "ft.com", 9))
    assert a.cluster_id == b.cluster_id == d.cluster_id
    assert (a.novelty, b.novelty) == (1.0, 0.5)
    assert d.n_sources == 3 and d.is_duplicate


def test_different_stories_stay_apart():
    c = StoryClusterer()
    a = c.assign(item(1, "Exxon to buy Pioneer Natural Resources for $60 billion", "reuters.com"))
    b = c.assign(item(2, "Boeing grounds 737 MAX fleet after door plug blowout", "reuters.com", 1))
    assert a.cluster_id != b.cluster_id and b.novelty == 1.0


def test_story_expires_after_window():
    c = StoryClusterer(window=timedelta(hours=1))
    a = c.assign(item(1, "Exxon to buy Pioneer Natural Resources for $60 billion", "reuters.com"))
    b = c.assign(item(2, "Exxon to buy Pioneer Natural Resources for $60 billion", "cnbc.com", 180))
    assert a.cluster_id != b.cluster_id and b.novelty == 1.0


def test_same_publisher_does_not_inflate_breadth():
    c = StoryClusterer()
    c.assign(item(1, "Fed raises rates by 25 basis points", "reuters.com"))
    r = c.assign(item(2, "Fed raises rates by 25 basis points (updated)", "reuters.com", 3))
    assert r.n_sources == 1


def test_replay_roundtrip_keeps_per_source_hints(tmp_path, monkeypatch):
    import riskengine.service as svc

    monkeypatch.setattr(svc, "REPLAY_DIR", tmp_path)
    a = item(1, "Exxon 8-K", "SEC EDGAR")
    a.labels_hint = {"8k_items": "2.02"}
    b = item(2, "Oil jumps", "reuters.com", 5)
    b.labels_hint = {"gdelt_tone": "-1.2"}
    svc.save_replay("t", [a, b])
    back = {i.item_id: i for i in svc.load_replay("t")}
    assert back["id1"].labels_hint == {"8k_items": "2.02"}
    assert back["id2"].labels_hint == {"gdelt_tone": "-1.2"}
