"""Labeling endpoints for building the gold evaluation set."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from riskengine.config import taxonomy
from riskengine.gold import GoldStore

router = APIRouter(prefix="/api/gold", tags=["gold"])
_store: GoldStore | None = None


def store() -> GoldStore:
    global _store
    if _store is None:
        _store = GoldStore()
    return _store


class EntityLabel(BaseModel):
    entity: str
    relevant: bool = True
    sentiment: int = Field(0, ge=-2, le=2)


class LabelIn(BaseModel):
    item_id: str
    entities: list[EntityLabel] = []
    events: list[str] = []
    subtype: str | None = None
    skipped: bool = False
    seconds: float | None = None


@router.get("/stats")
def stats():
    return store().stats()


@router.get("/impact")
def impact_eval():
    """Event-study evaluation of the impact model (scripts/train_impact_model.py)."""
    import json

    from riskengine.config import PROCESSED_DIR

    p = PROCESSED_DIR / "impact_eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@router.get("/backtest")
def backtest():
    """Module A walk-forward backtest results (scripts/run_backtest.py)."""
    import json

    from riskengine.config import PROCESSED_DIR

    p = PROCESSED_DIR / "backtest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@router.get("/eval")
def evaluations():
    """All saved evaluation results (data/processed/eval_<name>.json), keyed by predictor."""
    import json

    from riskengine.config import PROCESSED_DIR

    return {p.stem.removeprefix("eval_"): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(PROCESSED_DIR.glob("eval_*.json"))}


@router.post("/reload")
def reload():
    store().reload()
    return store().stats()


@router.get("/next")
def next_item(after: str | None = None):
    item = store().next(after)
    return {"item": item, "stats": store().stats()}


@router.get("/item/{item_id}")
def get_item(item_id: str):
    item = store().get(item_id)
    if item is None:
        raise HTTPException(404, item_id)
    return item


@router.post("/label")
def label(rec: LabelIn):
    valid = set(taxonomy())
    bad = [e for e in rec.events if e not in valid]
    if bad:
        raise HTTPException(400, f"unknown event classes {bad}")
    if not rec.skipped and not rec.events:
        raise HTTPException(400, "pick at least one event class (None/Noise counts)")
    try:
        saved = store().label(rec.model_dump())
    except KeyError:
        raise HTTPException(404, rec.item_id) from None
    return {"saved": saved, "next": store().next(rec.item_id), "stats": store().stats()}
