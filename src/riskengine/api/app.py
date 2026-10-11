"""HTTP API: the engine's output contract for downstream consumers + dashboard backend.

Run:  uvicorn riskengine.api.app:app --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from riskengine.api.gold_routes import router as gold_router
from riskengine.config import REPLAY_DIR, taxonomy, universe
from riskengine.modules.stress.scenarios import idiosyncratic, scenario_config, systemic
from riskengine.schema import RawItem, stable_id, utcnow
from riskengine.service import LIVE, TOPIC_INDEX, TOPIC_SIGNALS, TOPIC_STATUS, TOPIC_STRESS, EngineService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
svc: EngineService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    svc = EngineService()
    yield
    svc.stop_replay()
    svc.store.close()


app = FastAPI(title="AI/NLP Financial Risk Engine", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(gold_router)


def service() -> EngineService:
    assert svc is not None
    return svc


# ---- engine ----------------------------------------------------------------
@app.get("/api/health")
def health():
    s = service()
    return {"status": "ok", "models_loaded": s._engine is not None, "signals": s.store.count("signals"),
            "raw_items": s.store.count("raw_items"), "replay": s.replay.__dict__}


@app.get("/api/universe")
def get_universe():
    u = universe()
    return [{"ticker": c.ticker, "name": c.name, "sector": c.sector,
             "benchmark_weight": round(float(service().index.benchmark[c.ticker]), 6)} for c in u.companies]


@app.get("/api/taxonomy")
def get_taxonomy():
    return {k: {"label": v.label, "description": v.description, "subtypes": list(v.subtypes),
                "base_impact": v.base_impact, "channels": list(v.channels)} for k, v in taxonomy().items()}


@app.get("/api/signals")
def get_signals(entity: str | None = None, since: datetime | None = None, until: datetime | None = None,
                event: str | None = None, min_impact: int | None = Query(None, ge=1, le=10),
                limit: int = Query(200, le=5000)):
    sigs = service().store.signals(entity=entity, since=since, until=until, event=event, min_impact=min_impact,
                                   limit=limit)
    return [s.model_dump(mode="json") for s in sigs]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=20000)
    title: str = ""
    source_type: Literal["news", "social", "filing"] = "news"
    publisher: str | None = None
    store: bool = False


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Score arbitrary text. With store=true it also enters the live stream."""
    s = service()
    item = RawItem(item_id=stable_id("manual", req.title, req.text, utcnow()), source="manual",
                   source_type=req.source_type, title=req.title, body=req.text, publisher=req.publisher,
                   published_at=utcnow())
    if req.store:
        sigs = s.ingest([item])
    else:
        # score without touching shared story-cluster state
        from riskengine.nlp.dedup import StoryClusterer

        eng = s.engine
        saved, eng.clusterer = eng.clusterer, StoryClusterer()
        try:
            sigs = eng.process([item])
        finally:
            eng.clusterer = saved
    return [x.model_dump(mode="json") for x in sigs]


@app.get("/api/entities/summary")
def entities_summary(days: int = 7):
    """Per-entity aggregates over the last `days` of signal time."""
    store = service().store
    df = store.query_df(f"""
        WITH latest AS (SELECT max(published_at) AS t FROM signals)
        SELECT entity, count(*) AS n, avg(sentiment_score) AS avg_sentiment,
               sum(sentiment_score * source_weight) / nullif(sum(source_weight), 0) AS weighted_sentiment,
               max(impact_score) AS max_impact, avg(impact_score) AS avg_impact,
               max(published_at) AS last_signal
        FROM signals, latest
        WHERE published_at >= latest.t - INTERVAL {int(days)} DAY
        GROUP BY entity ORDER BY n DESC""")
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/api/entities/{entity}/timeseries")
def entity_timeseries(entity: str, freq: Literal["hour", "day", "week"] = "day"):
    df = service().store.query_df(f"""
        SELECT date_trunc('{freq}', published_at) AS t, count(*) AS n, avg(sentiment_score) AS sentiment,
               max(impact_score) AS max_impact, mode(event_primary) AS top_event
        FROM signals WHERE entity = ? GROUP BY 1 ORDER BY 1""", [entity])
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/api/events/distribution")
def event_distribution():
    df = service().store.query_df("""
        SELECT event_primary AS event, count(*) AS n, avg(impact_score) AS avg_impact,
               avg(sentiment_score) AS avg_sentiment FROM signals GROUP BY 1 ORDER BY n DESC""")
    return df.round(4).to_dict("records")


# ---- Module A ----------------------------------------------------------------
@app.get("/api/index/current")
def index_current():
    idx = service().index
    if not idx.history:
        idx.maybe_rebalance(datetime.now(UTC), force=True)
    return idx.history[-1].to_dict()


@app.get("/api/index/history")
def index_history(limit: int = Query(500, le=20000)):
    return [s.to_dict() for s in service().index.history[-limit:]]


# ---- Module B ----------------------------------------------------------------
@app.get("/api/stress/portfolio")
def stress_portfolio():
    tr = service().stress
    pf = tr.pf
    by = lambda col: (pf.groupby(col)["notional"].sum().round(2).sort_values(ascending=False).to_dict())  # noqa: E731
    return {"n_positions": len(pf), "total_notional": round(float(pf["notional"].sum()), 2),
            "market_value": round(float(pf["market_value"].sum()), 2), "baseline": tr.baseline,
            "by_instrument": by("instrument"), "by_sector": by("sector"), "by_rating": by("rating"),
            "by_country": by("country"), "positions": json.loads(pf.to_json(orient="records"))}


@app.get("/api/stress/scenarios")
def stress_scenarios():
    cfg = scenario_config()
    return {"trigger": cfg["trigger"],
            "systemic": {k: {"name": v["name"], "episode": v["episode"]} for k, v in cfg["systemic"].items()},
            "counterparties": sorted(service().stress.pf["counterparty"].unique().tolist())}


class StressRequest(BaseModel):
    kind: Literal["systemic", "idiosyncratic"] = "systemic"
    event_class: str = "geopolitical"
    severity: float = Field(0.8, ge=0.0, le=1.5)
    scope_countries: list[str] = []
    counterparty: str | None = None
    subtype: str | None = None


@app.post("/api/stress/run")
def stress_run(req: StressRequest):
    tr = service().stress
    if req.kind == "systemic":
        if req.event_class not in scenario_config()["systemic"]:
            raise HTTPException(400, f"no systemic template for {req.event_class}")
        scn = systemic(req.event_class, impact=0, scope_countries=req.scope_countries, severity=req.severity,
                       trigger={"manual": True})
    else:
        hit = tr.pf[tr.pf["counterparty"] == req.counterparty]
        if hit.empty:
            raise HTTPException(404, f"unknown counterparty {req.counterparty}")
        scn = idiosyncratic(req.counterparty, hit["sector"].iloc[0], impact=0, subtype=req.subtype,
                            severity=req.severity, trigger={"manual": True})
    tr.run(scn)
    return tr.runs[-1]


@lru_cache(maxsize=64)
def _curve(event_class: str, scope: tuple[str, ...]) -> dict:
    from riskengine.modules.stress.engine import severity_curve

    return severity_curve(service().stress.pf,
                          lambda s: systemic(event_class, impact=0, scope_countries=list(scope), severity=s))


@app.get("/api/stress/curve")
def stress_curve(event_class: str = "credit_event", scope: str = ""):
    """Reverse stress test: CET1 ratio vs severity and the severities that breach buffer / minimum."""
    if event_class not in scenario_config()["systemic"]:
        raise HTTPException(400, f"no systemic template for {event_class}")
    return _curve(event_class, tuple(sorted(c for c in scope.split(",") if c)))


@app.get("/api/stress/compare")
def stress_compare(severity: float = Query(1.0, ge=0.0, le=1.5)):
    """Every systemic scenario at the same severity, for side-by-side comparison."""
    from riskengine.modules.stress.engine import run_stress

    pf = service().stress.pf
    out = []
    for key, tpl in scenario_config()["systemic"].items():
        r = run_stress(pf, systemic(key, impact=0, severity=severity))
        out.append({"key": key, "name": tpl["name"], "episode": tpl["episode"], "summary": r.summary,
                    "attribution": r.attribution})
    return out


@app.get("/api/stress/library")
def stress_library():
    """Scenario templates with their anchor episode and the market moves measured over it."""
    lib = []
    for key, tpl in scenario_config()["systemic"].items():
        cal = tpl.get("calibrated", {})
        lib.append({"key": key, "name": tpl["name"], "episode": tpl["episode"],
                    "measured": cal, "credit_z": tpl["credit_z"], "revolver_drawdown": tpl.get("revolver_drawdown"),
                    "template": {k: tpl[k] for k in ("rates_bp", "credit_spread_bp", "equity_pct", "fx_pct",
                                                     "commodity_pct") if k in tpl}})
    return lib


@app.get("/api/stress/runs")
def stress_runs(limit: int = 50):
    return [{k: r[k] for k in ("run_id", "at", "scenario", "summary", "attribution")}
            for r in service().stress.runs[-limit:]][::-1]


@app.get("/api/stress/runs/{run_id}")
def stress_run_detail(run_id: str):
    for r in service().stress.runs:
        if r["run_id"] == run_id:
            return r
    raise HTTPException(404, run_id)


# ---- replay ----------------------------------------------------------------
@app.get("/api/replay/list")
def replay_list():
    return [LIVE] + sorted(p.stem for p in REPLAY_DIR.glob("*.parquet"))


class ReplayRequest(BaseModel):
    name: str
    speed: float = Field(3600.0, gt=0)
    start: datetime | None = None
    end: datetime | None = None
    reset: bool = True


@app.post("/api/replay/start")
def replay_start(req: ReplayRequest):
    s = service()
    if req.name != LIVE and not (REPLAY_DIR / f"{req.name}.parquet").exists():
        raise HTTPException(404, req.name)
    if req.reset:
        s.reset_state()
    if req.name == LIVE:
        return s.start_live().__dict__
    return s.start_replay(req.name, req.speed, req.start, req.end).__dict__


@app.post("/api/replay/stop")
def replay_stop():
    service().stop_replay()
    return service().replay.__dict__


# ---- live stream -----------------------------------------------------------
@app.get("/api/stream")
async def stream(request: Request):
    """Server-sent events: `signal`, `index`, `stress`, `status`."""
    s = service()
    topics = {TOPIC_SIGNALS: "signal", TOPIC_INDEX: "index", TOPIC_STRESS: "stress", TOPIC_STATUS: "status"}
    queues = {t: s.bus.open_queue(t) for t in topics}

    async def gen():
        try:
            yield {"event": "status", "data": json.dumps(s.replay.__dict__)}
            while not await request.is_disconnected():
                got = False
                for topic, q in queues.items():
                    while not q.empty():
                        msg = q.get_nowait()
                        data = msg.model_dump(mode="json") if hasattr(msg, "model_dump") else msg
                        yield {"event": topics[topic], "data": json.dumps(data, default=str)}
                        got = True
                if not got:
                    await asyncio.sleep(0.25)
        finally:
            for t, q in queues.items():
                s.bus.close_queue(t, q)

    return EventSourceResponse(gen(), ping=15)

