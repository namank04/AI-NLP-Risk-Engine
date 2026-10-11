import pytest

from riskengine.evaluation import evaluate

GOLD = {
    "a": {"entities": [{"entity": "BA", "relevant": True, "sentiment": -2}], "events": ["credit_event"]},
    "b": {"entities": [{"entity": "TSLA", "relevant": False, "sentiment": 0}], "events": ["none"]},
    "c": {"entities": [{"entity": "AAPL", "relevant": True, "sentiment": 1},
                       {"entity": "MSFT", "relevant": True, "sentiment": 0}], "events": ["earnings"]},
    "d": {"skipped": True, "entities": [], "events": []},
}


def test_perfect_predictor_scores_one():
    preds = {k: v for k, v in GOLD.items() if not v.get("skipped")}
    preds["b"] = {"entities": [], "events": ["none"]}
    r = evaluate(GOLD, preds)
    assert r["n_items"] == 3
    assert r["events"]["primary_accuracy"] == 1.0 and r["events"]["micro_f1"] == 1.0
    assert r["sentiment"]["exact_5pt"] == 1.0


def test_linking_precision_and_missed():
    preds = {
        "a": {"entities": [{"entity": "BA", "sentiment": -1}], "events": ["credit_event"]},
        "b": {"entities": [{"entity": "TSLA", "sentiment": 1}], "events": ["product_launch"]},
        "c": {"entities": [{"entity": "AAPL", "sentiment": 2}], "events": ["earnings", "product_launch"]},
    }
    r = evaluate(GOLD, preds)
    assert r["linking"]["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["linking"]["missed_entities"] == 1          # MSFT
    assert r["sentiment"]["n"] == 2                      # BA, AAPL (TSLA irrelevant)
    assert r["sentiment"]["accuracy_3class"] == 1.0 and r["sentiment"]["exact_5pt"] == 0.0
    assert r["events"]["primary_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["events"]["per_class"]["product_launch"]["precision"] == 0.0
