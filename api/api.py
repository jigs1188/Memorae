"""
api.py — FastAPI HTTP interface for the Memorae personal-memory engine.

Endpoints:
  POST /query          — Run a single custom or preset query
  GET  /queries        — List all built-in preset queries
  POST /queries/all    — Run all preset queries and return results
  GET  /health         — Health check + provider info
  GET  /results        — Return cached results.json if available

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
  docker-compose up
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import SCENARIO_NOW
from core.event_store import load_events, EventStore
from core.query_engine import QueryEngine, QuerySpec, QUERY_SPECS, QueryResult
from llm.llm_client import get_provider_info

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "memorae_mock_events.json"
RESULTS_PATH = Path(__file__).parent.parent / "results.json"

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Memorae — Personal Memory Intelligence Engine",
    description=(
        "A production-grade personal memory query engine that ingests raw event streams "
        "(WhatsApp, Slack, Gmail, Calendar, Notion) and answers natural-language queries "
        "using hybrid BM25 + keyword retrieval, contradiction resolution, and LLM generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (initialized at startup) ─────────────────────────────────────
_store: Optional[EventStore] = None
_engine: Optional[QueryEngine] = None
_now: Optional[datetime] = None


@app.on_event("startup")
async def startup_event():
    global _store, _engine, _now
    _now = datetime.fromisoformat(SCENARIO_NOW.replace("Z", "+00:00"))
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return
    events = load_events(str(DATA_PATH))
    _store = EventStore(events, _now)
    _engine = QueryEngine(_store, _now)
    stats = _store.stats()
    logger.info(
        f"Memorae ready | {stats['total']} events | "
        f"{stats['signal']} signal | {stats['noise']} noise | "
        f"Provider: {get_provider_info()['provider']}"
    )


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language query about your personal data")
    keywords: Optional[list[str]] = Field(
        None,
        description="Optional keyword hints. Auto-extracted from query if not provided."
    )
    must_include: Optional[list[str]] = Field(
        None,
        description="Patterns that MUST appear in selected events (exact match)."
    )
    top_k: int = Field(30, ge=1, le=100, description="Max events to retrieve before context packing")
    system_instruction: Optional[str] = Field(
        None,
        description="Custom system prompt override. Uses a default if not provided."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What should I focus on today?",
                "top_k": 30
            }
        }
    }


class PresetQueryRequest(BaseModel):
    preset_index: int = Field(..., ge=0, description="Index of preset query (0-4). See GET /queries.")


class QueryResponse(BaseModel):
    query: str
    answer: str
    model_used: str
    context_stats: dict[str, Any]
    selected_context: list[dict[str, Any]]
    reasoning: dict[str, Any]
    contradiction_notes: list[str]


class HealthResponse(BaseModel):
    status: str
    scenario_time: str
    events_loaded: int
    provider: str
    primary_model: str
    key_configured: bool
    bm25_enabled: bool


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_engine():
    if _engine is None or _store is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Check that data file exists.")


def _result_to_dict(result: QueryResult) -> dict:
    return {
        "query": result.query,
        "answer": result.answer,
        "model_used": result.model_used,
        "context_stats": {
            "token_estimate": result.token_estimate,
            "events_used": len(result.selected_context),
            "events_dropped": result.dropped_count,
        },
        "selected_context": result.selected_context,
        "reasoning": result.reasoning,
        "contradiction_notes": result.contradiction_notes,
    }


def _spec_from_request(req: QueryRequest) -> QuerySpec:
    keywords = req.keywords or req.query.lower().split()
    system_instruction = req.system_instruction or (
        f"You are a personal AI assistant. Today is {SCENARIO_NOW}. "
        f"Answer this query based on the events: \"{req.query}\". "
        f"Be specific, time-aware, and grounded in the provided context. "
        f"When uncertain, say so explicitly."
    )
    return QuerySpec(
        query=req.query,
        keywords=keywords,
        must_include=req.must_include or [],
        top_k=req.top_k,
        system_instruction=system_instruction,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check. Returns provider info and event store statistics."""
    pinfo = get_provider_info()

    # Check if BM25 is available
    try:
        import rank_bm25
        bm25_ok = True
    except ImportError:
        bm25_ok = False

    return HealthResponse(
        status="ok" if _store is not None else "degraded",
        scenario_time=SCENARIO_NOW,
        events_loaded=len(_store.events) if _store else 0,
        provider=pinfo["provider"],
        primary_model=pinfo["primary_model"],
        key_configured=pinfo["key_configured"],
        bm25_enabled=bm25_ok,
    )


@app.get("/queries", tags=["Queries"])
async def list_preset_queries():
    """
    List all built-in preset queries.
    These are the 5 queries from the assignment spec, pre-configured with
    optimized keyword sets and system prompts.
    """
    return {
        "preset_queries": [
            {
                "index": i,
                "query": spec.query,
                "keywords_count": len(spec.keywords),
                "top_k": spec.top_k,
            }
            for i, spec in enumerate(QUERY_SPECS)
        ]
    }


@app.post("/query", response_model=QueryResponse, tags=["Queries"])
async def run_query(req: QueryRequest):
    """
    Run a single natural-language query against your personal event stream.

    The engine performs 4 steps:
    1. **Source & Signal Selection** — Hybrid BM25 + keyword scoring, noise filtering
    2. **Context Construction** — Token-budget packing, contradiction resolution
    3. **Answer Generation** — LLM (Gemini or OpenAI) with grounded prompting
    4. **Reasoning Explanation** — Returns why events were selected/ignored

    To run one of the built-in preset queries, use `GET /queries` to get the
    index, then pass the same query text here.
    """
    _require_engine()
    spec = _spec_from_request(req)
    try:
        result = _engine.run(spec)
        return _result_to_dict(result)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")
    except Exception as exc:
        logger.exception(f"Unexpected error running query: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/queries/all", tags=["Queries"])
async def run_all_preset_queries():
    """
    Run all 5 preset queries and return results.

    Answers:
    - What should I focus on today?
    - What commitments am I at risk of missing?
    - What have I been procrastinating on?
    - Summarize everything related to the UIE proposal.
    - What personal/family tasks need my attention?

    This saves results to results.json and returns them.
    Warning: This makes 5 LLM calls and may take 30-60 seconds.
    """
    _require_engine()
    results = []
    for spec in QUERY_SPECS:
        try:
            result = _engine.run(spec)
            results.append(_result_to_dict(result))
        except Exception as exc:
            logger.error(f"Failed query '{spec.query}': {exc}")
            results.append({
                "query": spec.query,
                "answer": f"[ERROR: {exc}]",
                "model_used": "none",
                "context_stats": {},
                "selected_context": [],
                "reasoning": {},
                "contradiction_notes": [],
            })

    # Cache to disk
    try:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {RESULTS_PATH}")
    except Exception as exc:
        logger.warning(f"Could not save results: {exc}")

    return {"results": results, "count": len(results)}


@app.get("/results", tags=["Queries"])
async def get_cached_results():
    """
    Return the cached results from the last run of /queries/all or python main.py.
    Returns 404 if no results have been generated yet.
    """
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No cached results found. Run POST /queries/all or python main.py first."
        )
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {"results": data, "count": len(data), "source": str(RESULTS_PATH)}


@app.get("/events/stats", tags=["Events"])
async def event_stats():
    """Return statistics about the loaded event store."""
    _require_engine()
    stats = _store.stats()
    return {
        "scenario_time": SCENARIO_NOW,
        "total_events": stats["total"],
        "signal_events": stats["signal"],
        "noise_events": stats["noise"],
        "events_with_urgency": stats["with_urgency"],
        "data_file": str(DATA_PATH),
    }
