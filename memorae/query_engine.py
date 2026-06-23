"""
query_engine.py — Query processing and answer generation.

For each query this module:
  Step 1 – Source & Signal Selection   (calls EventStore.retrieve)
  Step 2 – Context Construction        (calls context_builder.build_context)
  Step 3 – Answer Generation           (calls LLM via llm_client — OpenAI or Gemini)
  Step 4 – Reasoning Explanation       (assembled from scoring metadata)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

from event_store import EventStore, ScoredEvent
from context_builder import build_context, BuiltContext
from llm_client import generate
from config import SCENARIO_NOW, MAX_SELECTED_EVENTS

logger = logging.getLogger(__name__)


# ── Query descriptors ──────────────────────────────────────────────────────────

@dataclass
class QuerySpec:
    """Defines everything the engine needs to process a single query."""
    query: str
    keywords: list[str]
    must_include: list[str] = field(default_factory=list)
    top_k: int = MAX_SELECTED_EVENTS
    system_instruction: str = ""


# Pre-defined query specs for the five required queries
QUERY_SPECS: list[QuerySpec] = [
    QuerySpec(
        query="What should I focus on today?",
        keywords=[
            "due", "deadline", "today", "apr 13", "standup", "review",
            "submit", "send", "finish", "overdue", "urgent", "before",
            "proposal", "uie", "rubric", "meeting", "sync",
        ],
        must_include=["apr 13", "today", "standup", "uie", "rubric"],
        system_instruction=(
            "You are a personal AI assistant. Today is 2026-04-13 03:00 UTC. "
            "Based on the events below, identify the 3-5 highest-priority items "
            "the user should focus on TODAY. Be specific: include names, deadlines, "
            "and concrete next actions. Flag anything overdue. Be concise."
        ),
    ),
    QuerySpec(
        query="What commitments am I at risk of missing?",
        keywords=[
            "due", "deadline", "risk", "overdue", "promised", "blocked",
            "missing", "waiting", "confirm", "send", "approve", "rubric",
            "proposal", "redlines", "payment", "renewal", "insurance",
            "dentist", "school", "form", "negotiation",
        ],
        must_include=["due", "deadline", "overdue", "risk", "promised", "missing", "confirm"],
        system_instruction=(
            "You are a personal AI assistant. Today is 2026-04-13 03:00 UTC. "
            "Review the events and identify ALL commitments the user has made or "
            "is responsible for that are: (a) already past due, or (b) at risk of "
            "slipping in the next 48 hours. For each, state: what is owed, to whom, "
            "by when, and what the consequence of missing it is. "
            "Flag anything where a stakeholder has already sent a nudge or reminder."
        ),
    ),
    QuerySpec(
        query="What have I been procrastinating on?",
        keywords=[
            "nudge", "reminder", "again", "still", "slips", "forgot",
            "not sent", "missing", "pending", "need to", "owe",
            "before it slips", "friendly nudge", "export", "screenshots",
            "receipts", "upload", "redlines", "owner", "postmortem",
        ],
        must_include=["nudge", "again", "slips", "still", "friendly", "owe", "before"],
        system_instruction=(
            "You are a personal AI assistant. Today is 2026-04-13 03:00 UTC. "
            "From the event stream, identify tasks the user has been repeatedly "
            "reminded about, promised to do but has not yet done, or explicitly "
            "noted as something that 'keeps slipping'. "
            "Group related items, note how long they've been pending, and who is waiting."
        ),
    ),
    QuerySpec(
        query="Summarize everything related to the UIE proposal.",
        keywords=[
            "uie", "unified intelligence engine", "proposal", "nina", "appendix",
            "risk", "rollout", "migration", "diagrams", "external-safe",
            "ravi", "cedric", "procurement", "48.5k", "42k", "failure modes",
            "northstar", "elt", "data room", "ingest", "retry",
        ],
        must_include=["uie", "proposal", "nina", "appendix", "rollout", "ravi", "cedric"],
        top_k=40,
        system_instruction=(
            "You are a personal AI assistant. Today is 2026-04-13 03:00 UTC. "
            "Summarize EVERYTHING related to the UIE (Unified Intelligence Engine) proposal. "
            "Cover: current status, what is still outstanding, key stakeholders and their "
            "requirements, deadline updates (including any superseded deadlines), "
            "open action items, and known risks. "
            "Note any contradictions or updates in the data (e.g., deadline changes, "
            "estimate revisions). Structure your answer clearly."
        ),
    ),
    QuerySpec(
        query="What personal/family tasks need my attention?",
        keywords=[
            "pari", "mom", "karan", "school", "cardiology", "insurance",
            "dentist", "apartment", "maintenance", "parent-teacher",
            "bus form", "pickup", "pharmacy", "immunization",
        ],
        must_include=["pari", "mom", "karan", "school", "insurance", "dentist", "apartment"],
        system_instruction=(
            "You are a personal AI assistant. Today is 2026-04-13 03:00 UTC. "
            "From the events, extract all personal and family obligations. "
            "Rank them by urgency. Note what still needs action, what is pending "
            "someone else, and what has upcoming deadlines in the next 3 days."
        ),
    ),
]


# ── Output model ───────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    answer: str
    model_used: str
    selected_context: list[dict]
    reasoning: dict[str, Any]
    contradiction_notes: list[str]
    token_estimate: int
    dropped_count: int


# ── Engine ─────────────────────────────────────────────────────────────────────

class QueryEngine:
    def __init__(self, store: EventStore, now: datetime):
        self.store = store
        self.now = now

    def _build_llm_prompt(
        self,
        spec: QuerySpec,
        ctx: BuiltContext,
    ) -> str:
        return (
            f"{spec.system_instruction}\n\n"
            f"{ctx.context_text}\n\n"
            f"--- END OF CONTEXT ---\n\n"
            f"Answer the user query: \"{spec.query}\"\n\n"
            f"Provide a specific, time-aware answer grounded in the events above. "
            f"When uncertain, say so and explain why. "
            f"IMPORTANT: Do NOT repeat the system instruction or query back to the user. Just provide the final answer directly."
        )

    def _build_reasoning(
        self,
        spec: QuerySpec,
        scored: list[ScoredEvent],
        ctx: BuiltContext,
    ) -> dict[str, Any]:
        why_selected_items = []
        for se in ctx.events:
            why_selected_items.append({
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content_preview": se.event.content[:100],
                "score": round(se.score, 3),
                "reason": se.why_selected,
                "score_breakdown": {k: round(v, 3) for k, v in se.breakdown.items()},
            })

        why_ignored_items = []
        for se in ctx.dropped_events[:10]:  # show first 10 dropped
            why_ignored_items.append({
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content_preview": se.event.content[:80],
                "score": round(se.score, 3),
                "reason": "below token budget threshold",
            })

        # Noise events (excluded before scoring)
        noise_count = sum(1 for e in self.store.events if e.is_noise)

        return {
            "selection_strategy": (
                f"Keywords used: {spec.keywords[:8]}... "
                f"Must-include patterns: {spec.must_include}. "
                f"Events scored on: recency (decay τ=72h), urgency signals, "
                f"keyword relevance, source priority tier."
            ),
            "why_selected": why_selected_items,
            "why_ignored": (
                f"{noise_count} events pre-filtered as noise (social chatter, OTPs, "
                f"ride receipts, newsletters, duplicate coffee-machine messages). "
                f"{len(ctx.dropped_events)} events dropped due to token budget."
            ),
            "dropped_sample": why_ignored_items,
            "uncertainty": self._assess_uncertainty(spec, scored),
            "contradiction_resolution": ctx.contradiction_notes,
        }

    def _assess_uncertainty(
        self, spec: QuerySpec, scored: list[ScoredEvent]
    ) -> str:
        uncertainties = []

        if "procrastinating" in spec.query:
            uncertainties.append(
                "The system cannot distinguish between 'not done yet' and "
                "'done but not recorded in the event stream'. Tasks noted "
                "as recurring reminders are treated as likely incomplete."
            )
        if "focus on today" in spec.query:
            uncertainties.append(
                "The system does not have confirmation of what Aarav has already "
                "completed this morning (events only go to 03:00 UTC on Apr 13). "
                "Items may already be in-progress."
            )
        if "uie" in spec.query.lower():
            uncertainties.append(
                "It is unclear whether Aarav has already sent the UIE appendix "
                "to Nina before the review at 14:30 IST. The last event is at "
                "~03:00 UTC (08:30 IST) so morning actions are not captured."
            )

        return "; ".join(uncertainties) if uncertainties else "No major uncertainties identified."

    def run(self, spec: QuerySpec) -> QueryResult:
        logger.info(f"\n{'='*60}\nProcessing query: {spec.query}\n{'='*60}")

        # Step 1: Source & Signal Selection
        scored = self.store.retrieve(
            keywords=spec.keywords,
            exclude_noise=True,
            top_k=spec.top_k,
            must_include_patterns=spec.must_include if spec.must_include else None,
        )
        logger.info(f"Step 1: Selected {len(scored)} events after scoring")

        # Step 2: Context Construction
        ctx = build_context(scored, query=spec.query, now=self.now)
        logger.info(f"Step 2: Context built ~{ctx.token_estimate} tokens, {len(ctx.events)} events")

        # Step 3: Answer Generation
        prompt = self._build_llm_prompt(spec, ctx)
        answer, model_used = generate(prompt, temperature=0.2, max_output_tokens=1500)
        logger.info(f"Step 3: Answer generated by {model_used}")

        # Step 4: Reasoning Explanation
        reasoning = self._build_reasoning(spec, scored, ctx)
        logger.info(f"Step 4: Reasoning assembled")

        # Serialize selected context for output
        selected_ctx_output = [
            {
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content": se.event.content,
                "relevance_score": round(se.score, 3),
            }
            for se in ctx.events
        ]

        return QueryResult(
            query=spec.query,
            answer=answer,
            model_used=model_used,
            selected_context=selected_ctx_output,
            reasoning=reasoning,
            contradiction_notes=ctx.contradiction_notes,
            token_estimate=ctx.token_estimate,
            dropped_count=len(ctx.dropped_events),
        )

    def run_all(self) -> list[QueryResult]:
        results = []
        for spec in QUERY_SPECS:
            result = self.run(spec)
            results.append(result)
        return results
