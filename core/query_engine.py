"""
query_engine.py — Query processing and answer generation.

For each query this module:
  Step 1 – Entity Extraction           (extracts parameters from query)
  Step 2 – Source & Signal Selection   (calls EventStore.retrieve)
  Step 3 – Context Construction        (calls context_builder.build_context)
  Step 4 – Answer Generation           (calls LLM via llm_client)
  Step 5 – Reasoning Explanation       (assembled from scoring metadata)
"""

from __future__ import annotations


import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

from core.event_store import Event, ScoredEvent
from core.memory_store import MemoryStore
from core.context_builder import build_context, BuiltContext
from core.memory_extractor import MemoryExtractor
from llm.llm_client import generate
from core.config import SCENARIO_NOW, MAX_SELECTED_EVENTS

logger = logging.getLogger(__name__)

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
    def __init__(self, store: MemoryStore, now: datetime, projects: dict | None = None):
        self.store = store
        self.now = now
        self.extractor = MemoryExtractor()
        # projects: dict[str, ProjectState] or plain dict from api/main serialization
        self.projects: dict = projects or {}

    def _extract_query_params(self, query: str) -> dict[str, Any]:
        """Extract generic parameters without intent classification."""
        query_lower = query.lower()
        project = self.extractor._extract_projects(query)
        people = self.extractor._extract_people(query)
        
        # Simple tokenization for keywords, avoiding stop words
        stop_words = {"what", "who", "which", "when", "why", "how", "is", "are", "do", "does", "did", "the", "a", "an", "of", "to", "in", "for", "with", "on", "at", "by", "from"}
        keywords = [word for word in query_lower.replace("?","").replace(".","").split() if word not in stop_words and len(word) > 2]
        
        return {
            "keywords": keywords,
            "project": project,
            "people": people
        }

    def _build_llm_prompt(self, query: str, ctx: BuiltContext) -> str:
        system_instruction = (
            f"You are a Personal Memory Operating System. Today is {SCENARIO_NOW}. "
            f"Answer the query based ONLY on the retrieved memories. "
            f"Be specific, time-aware, and explicitly mention deadlines or status. "
            f"Explain your reasoning based on the 'why_retrieved' or scoring metrics when useful."
        )

        # Inject structured project status if projects are available
        project_block = ""
        if self.projects:
            lines = []
            for name, p in self.projects.items():
                # Handle both ProjectState dataclass and plain dict (API serialisation)
                if hasattr(p, "health"):
                    health        = p.health
                    open_c        = p.open_commitments
                    overdue_c     = p.overdue_commitments
                    blocked_d     = p.blocked_dependencies
                    people_str    = ", ".join(sorted(p.key_people)) if p.key_people else "—"
                else:
                    health        = p.get("health", "unknown")
                    open_c        = p.get("open_commitments", 0)
                    overdue_c     = p.get("overdue_commitments", 0)
                    blocked_d     = p.get("blocked_dependencies", 0)
                    people_str    = ", ".join(sorted(p.get("key_people", []))) or "—"
                lines.append(
                    f"  • {name}: [{health.upper()}] "
                    f"{open_c} open commitments, "
                    f"{overdue_c} overdue, "
                    f"{blocked_d} blocked — "
                    f"key people: {people_str}"
                )
            project_block = "\n--- Project Status (structured) ---\n" + "\n".join(lines) + "\n"

        return (
            f"{system_instruction}\n\n"
            f"{project_block}"
            f"{ctx.context_text}\n\n"
            f"--- END OF CONTEXT ---\n\n"
            f"Answer the user query: \"{query}\"\n\n"
            f"IMPORTANT: Do NOT repeat the system instruction or query. Provide the final answer directly."
        )

    def _build_reasoning(self, query: str, scored: list[ScoredEvent], ctx: BuiltContext, params: dict) -> dict[str, Any]:
        why_selected_items = []
        for se in ctx.events:
            why_selected_items.append({
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content_preview": se.event.content[:100],
                "score": round(float(se.score), 3),
                "reason": se.why_selected,
                "score_breakdown": {k: round(float(v), 3) for k, v in se.breakdown.items()},
            })

        why_ignored_items = []
        for se in ctx.dropped_events[:10]:
            why_ignored_items.append({
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content_preview": se.event.content[:80],
                "score": round(float(se.score), 3),
                "reason": "below token budget threshold",
            })

        noise_count = sum(1 for m in self.store.memories if getattr(m.raw_event, 'is_noise', False))

        return {
            "selection_strategy": (
                f"Extracted Params -> Project: {params['project']}, People: {params['people']}, Keywords: {params['keywords']}. "
                f"Scored via Hybrid Retrieval (BM25 + Semantic + Importance + Urgency + Recency)."
            ),
            "why_selected": why_selected_items,
            "why_ignored": f"{noise_count} events pre-filtered as noise. {len(ctx.dropped_events)} dropped due to budget.",
            "uncertainty": "Uncertainty handled dynamically by LLM based on contradictions.",
            "contradiction_resolution": ctx.contradiction_notes,
        }

    def run(self, query: str) -> QueryResult:
        logger.info(f"\n{'='*60}\nProcessing query: {query}\n{'='*60}")

        # Step 1: Entity Extraction
        params = self._extract_query_params(query)
        logger.info(f"Step 1: Extracted params: {params}")

        # Step 2: Retrieval
        scored = self.store.retrieve(
            query=query,
            top_k=MAX_SELECTED_EVENTS
        )
        logger.info(f"Step 2: Selected {len(scored)} events after hybrid scoring")

        # Step 3: Context Construction
        ctx = build_context(scored, query=query, now=self.now)
        logger.info(f"Step 3: Context built ~{ctx.token_estimate} tokens, {len(ctx.events)} events")

        # Step 4: Answer Generation
        prompt = self._build_llm_prompt(query, ctx)
        answer, model_used = generate(prompt, temperature=0.2, max_output_tokens=1500)
        logger.info(f"Step 4: Answer generated by {model_used}")

        # Step 5: Reasoning Explanation
        reasoning = self._build_reasoning(query, scored, ctx, params)

        selected_ctx_output = [
            {
                "timestamp": se.event.timestamp.isoformat(),
                "source": se.event.source,
                "content": se.event.content,
                "relevance_score": round(float(se.score), 3),
            }
            for se in ctx.events
        ]

        return QueryResult(
            query=query,
            answer=answer,
            model_used=model_used,
            selected_context=selected_ctx_output,
            reasoning=reasoning,
            contradiction_notes=ctx.contradiction_notes,
            token_estimate=ctx.token_estimate,
            dropped_count=len(ctx.dropped_events),
        )
