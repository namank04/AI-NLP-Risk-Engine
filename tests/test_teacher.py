from riskengine.teacher.llm import _system_prompt, parse_response


def test_parses_clean_json():
    raw = '{"entities":[{"entity":"BA","relevant":true,"sentiment":-2}],"events":["credit_event"],"subtype":"rating_action"}'
    out = parse_response(raw, ["BA"])
    assert out == {"entities": [{"entity": "BA", "relevant": True, "sentiment": -2}],
                   "events": ["credit_event"], "subtype": "rating_action"}


def test_strips_reasoning_and_prose():
    raw = '<think>hmm, downgrade...</think>Sure! {"entities":[{"entity":"ba","sentiment":"-1"}],"events":["credit_event"]}'
    out = parse_response(raw, ["BA"])
    assert out["entities"][0]["sentiment"] == -1 and out["events"] == ["credit_event"]


def test_drops_invalid_classes_and_clamps_sentiment():
    raw = '{"entities":[{"entity":"TSLA","sentiment":5}],"events":["bogus","none","earnings"],"subtype":"nope"}'
    out = parse_response(raw, ["TSLA"])
    assert out["events"] == ["earnings"] and out["subtype"] is None
    assert out["entities"][0]["sentiment"] == 2


def test_missing_entity_filled_neutral_and_unknown_ignored():
    raw = '{"entities":[{"entity":"AAPL","sentiment":1},{"entity":"XYZ","sentiment":2}],"events":[]}'
    out = parse_response(raw, ["AAPL", "TSLA"])
    assert [e["entity"] for e in out["entities"]] == ["AAPL", "TSLA"]
    assert out["entities"][1]["sentiment"] == 0 and out["events"] == ["none"]


def test_garbage_returns_none():
    assert parse_response("I cannot help with that", ["AAPL"]) is None
    assert parse_response('{"entities": []}', ["AAPL"]) is None


def test_prompt_lists_every_class():
    p = _system_prompt()
    for k in ["geopolitical", "macroeconomic", "credit_event", "none", "rating_action"]:
        assert k in p


def test_parse_batch_maps_items_by_number():
    from riskengine.teacher.llm import parse_batch

    raw = ('{"items": [{"item": 2, "entities": [{"entity": "BA", "sentiment": -2}], "events": ["credit_event"]},'
           ' {"item": 1, "entities": [{"entity": "AAPL", "sentiment": 1}], "events": ["earnings"]}]}')
    items = parse_batch(raw)
    assert set(items) == {1, 2} and items[2]["events"] == ["credit_event"]
    assert parse_batch("no json here") == {}


def test_label_batch_uses_cache_and_one_request(tmp_path, monkeypatch):
    import riskengine.teacher.llm as llm

    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache.jsonl")
    t = llm.Teacher(llm.TeacherConfig(api_key="x", min_interval_s=0))
    calls = []

    def fake_chat(user, temperature, max_tokens=None):
        calls.append(user)
        return ('{"items": [{"item": 1, "entities": [{"entity": "BA", "sentiment": -2}], "events": ["credit_event"],'
                ' "subtype": "rating_action"}, {"item": 2, "entities": [{"entity": "KO", "sentiment": 0}], "events": ["none"]}]}')

    monkeypatch.setattr(t, "_chat", fake_chat)
    rows = [("a", "Boeing cut to junk", ["BA"]), ("b", "Coke listicle", ["KO"])]
    out = t.label_batch(rows)
    assert out[0]["events"] == ["credit_event"] and out[0]["subtype"] == "rating_action"
    assert out[1]["events"] == ["none"] and len(calls) == 1
    assert t.label_batch(rows) == out and len(calls) == 1          # second call served from cache
