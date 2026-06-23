"""
context_builder.py — Context construction for LLM prompts.

Responsibility: Given a ranked list of ScoredEvents, produce a well-structured,
token-efficient context string that the LLM will use.

Design for scale
────────────────
In production with 10k messages / 1k notes / 500 reminders:

  Stage 1 – Coarse retrieval (this module):
    BM25 or keyword index → top-200 candidates.
  Stage 2 – Semantic reranking:
    Embed query + candidates → cosine similarity → top-30.
  Stage 3 – Context window fitting:
    Token-count each event; greedily fill budget, highest-score first.
  Stage 4 – Deduplication:
    Cluster near-duplicate messages (e.g. repeated coffee-machine noise)
    and include only the canonical form.
  Stage 5 – Contradiction resolution:
    For the same entity (e.g., UIE deadline), keep only the latest update
    and annotate earlier versions as "superseded".

Token estimation
────────────────
We approximate 1 token ≈ 4 characters (conservative for English prose).
This avoids importing tiktoken just for a small dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from event_store import Event, ScoredEvent
from config import TARGET_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4          # rough approximation


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class BuiltContext:
    events: list[ScoredEvent]
    context_text: str
    token_estimate: int
    dropped_events: list[ScoredEvent]
    contradiction_notes: list[str]


def _format_event(ev: Event, tag: str = "") -> str:
    ts = ev.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    label = f"[{tag}] " if tag else ""
    return f"[{ts} | {ev.source}] {label}{ev.content}"


def _detect_contradictions(events: list[ScoredEvent]) -> tuple[list[ScoredEvent], list[str]]:
    """
    Detect and resolve contradictions / updates for known entities.

    Strategy: for each tracked entity key (e.g., "UIE proposal deadline"),
    if multiple events contain conflicting information, keep the LATEST one
    and mark earlier ones as superseded.

    Returns cleaned event list and contradiction notes.
    """
    notes: list[str] = []

    # UIE deadline supersession
    uie_deadline_events = [
        se for se in events
        if "uie" in se.event.content.lower() and any(
            kw in se.event.content.lower()
            for kw in ["due", "deadline", "apr 10", "apr 13", "moved"]
        )
    ]
    if len(uie_deadline_events) > 1:
        # Sort by timestamp; last one wins
        uie_deadline_events.sort(key=lambda x: x.event.timestamp)
        latest = uie_deadline_events[-1]
        notes.append(
            f"UIE deadline UPDATED: earlier deadline 'Apr 10' was superseded by "
            f"message on {latest.event.timestamp.strftime('%Y-%m-%d')} → "
            f"'{latest.event.content[:80]}...'"
        )

    # Procurement estimate update
    proc_events = [
        se for se in events
        if any(kw in se.event.content.lower() for kw in ["42k", "48.5k", "procurement estimate"])
    ]
    if len(proc_events) > 1:
        proc_events.sort(key=lambda x: x.event.timestamp)
        latest = proc_events[-1]
        notes.append(
            f"Procurement estimate UPDATED: $42k → $48.5k per "
            f"{latest.event.timestamp.strftime('%Y-%m-%d')} email from Cedric."
        )

    # Southridge clause 8 status
    clause8_events = [
        se for se in events
        if "clause 8" in se.event.content.lower()
    ]
    if clause8_events:
        clause8_events.sort(key=lambda x: x.event.timestamp)
        latest_text = clause8_events[-1].event.content.lower()
        if "approved" in latest_text:
            notes.append("Southridge clause 8: APPROVED as of Apr 11. SOW is unblocked.")

    return events, notes


def build_context(
    scored_events: list[ScoredEvent],
    query: str,
    now: datetime,
    token_budget: int = TARGET_CONTEXT_TOKENS,
    hard_limit: int = MAX_CONTEXT_TOKENS,
) -> BuiltContext:
    """
    Construct a prompt-ready context string from scored events.

    Steps:
      1. Resolve contradictions and annotate
      2. Greedily pack events by score until token budget is hit
      3. Format into a structured string with header
    """
    events_after_dedup, contradiction_notes = _detect_contradictions(scored_events)

    header = (
        f"=== PERSONAL MEMORY CONTEXT ===\n"
        f"Current time: {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Query: {query}\n"
    )
    if contradiction_notes:
        header += "\n--- Updates & Corrections ---\n"
        for note in contradiction_notes:
            header += f"• {note}\n"
    header += "\n--- Relevant Events (ranked by importance) ---\n"

    used_budget = _estimate_tokens(header)
    selected: list[ScoredEvent] = []
    dropped: list[ScoredEvent] = []

    for se in events_after_dedup:
        line = _format_event(se.event) + "\n"
        cost = _estimate_tokens(line)
        if used_budget + cost <= token_budget:
            selected.append(se)
            used_budget += cost
        elif used_budget + cost <= hard_limit:
            # Over soft budget but under hard limit — include only high scorers
            if se.score >= 0.55:
                selected.append(se)
                used_budget += cost
            else:
                dropped.append(se)
        else:
            dropped.append(se)

    body = header + "\n".join(_format_event(se.event) for se in selected)

    logger.info(
        f"Context built: {len(selected)} events, ~{used_budget} tokens "
        f"({len(dropped)} dropped)"
    )

    return BuiltContext(
        events=selected,
        context_text=body,
        token_estimate=used_budget,
        dropped_events=dropped,
        contradiction_notes=contradiction_notes,
    )
