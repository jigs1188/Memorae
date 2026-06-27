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
    For any entity where dates or numeric values change across timestamps,
    emit an UPDATE annotation pointing to the latest event.

Token estimation
────────────────
We approximate 1 token ≈ 4 characters (conservative for English prose).
This avoids importing tiktoken just for a small dataset.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.event_store import Event, ScoredEvent
from core.config import TARGET_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS

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


def _format_event(se: ScoredEvent, tag: str = "") -> str:
    ev = se.event
    ts = ev.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    label = f"[{tag}] " if tag else ""
    
    # Inject metadata if available
    meta_tags = []
    if se.metadata:
        if se.metadata.get("project"):
            meta_tags.append(f"Project: {se.metadata['project']}")
        if se.metadata.get("people"):
            meta_tags.append(f"People: {', '.join(se.metadata['people'])}")
        if se.metadata.get("relationships"):
            rel_strs = []
            for k, v in se.metadata["relationships"].items():
                if isinstance(v, list):
                    rel_strs.append(f"{k}: {', '.join(v)}")
                else:
                    rel_strs.append(f"{k}: {v}")
            meta_tags.append(f"Relations: {' | '.join(rel_strs)}")
            
    meta_str = f" [{'] ['.join(meta_tags)}]" if meta_tags else ""
    
    return f"[{ts} | {ev.source}]{meta_str} {label}{ev.content}"


# ── Patterns for generic contradiction detection ───────────────────────────────

_DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_NUM_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?k?\b"    # $42k, $48.5k, $1,200
    r"|\b\d+(?:\.\d+)?k\b"       # 42k, 48.5k
    r"|\b\d{4,}\b"                # plain numbers ≥ 4 digits
    r"|\b\d+(?:\.\d+)?\s*%",     # percentages
    re.IGNORECASE,
)

_TOPIC_STOP = {
    "the", "a", "an", "is", "are", "was", "be", "been", "has", "have",
    "had", "will", "would", "can", "could", "to", "of", "in", "for",
    "on", "at", "by", "and", "or", "not", "no", "this", "that", "it",
    "we", "i", "you", "he", "she", "they", "with", "as", "if", "but",
    "from", "up", "out", "so", "do", "did", "get", "got",
}


def _topic_key(text: str) -> str:
    """First 2 meaningful content words — used as a rough entity fingerprint.

    Using 2 words (not 3) gives better clustering: 'UIE proposal' matches
    both 'UIE proposal due Apr 10' and 'UIE proposal deadline moved to Apr 13'.
    Min token length of 3 retains short-but-meaningful terms like 'UIE'.
    """
    tokens = [
        t.lower().strip(".,!?;:'\"()")
        for t in text.split()
        if len(t) >= 3 and t.lower().strip(".,!?;:'\"()") not in _TOPIC_STOP
    ]
    return " ".join(tokens[:2])


def _detect_contradictions(events: list[ScoredEvent]) -> tuple[list[ScoredEvent], list[str]]:
    """
    Generic contradiction / update detector — no hardcoded entity names.

    Algorithm:
      1. For each scored event, extract dates and numeric values.
      2. Group events into topic clusters using the first 3 significant words.
      3. Within each cluster, compare the earliest and latest event's
         date/numeric token sets.  If they differ, emit an UPDATE note
         that quotes the latest event directly.

    This works generically for any project, person, or entity where values or
    dates evolve over time.

    Returns (unchanged_event_list, list_of_note_strings).
    """
    notes: list[str] = []

    # Build per-topic groups: topic_key → list of (ScoredEvent, dates, nums)
    clusters: dict[str, list] = defaultdict(list)
    for se in events:
        key = _topic_key(se.event.content)
        if not key:
            continue
        dates = set(d.lower().strip() for d in _DATE_RE.findall(se.event.content))
        nums  = set(n.lower().strip() for n in _NUM_RE.findall(se.event.content))
        if dates or nums:
            clusters[key].append((se, dates, nums))

    # Detect changes between earliest and latest event in each cluster
    for topic, group in clusters.items():
        if len(group) < 2:
            continue

        group.sort(key=lambda x: x[0].event.timestamp)
        earliest_se, e_dates, e_nums = group[0]
        latest_se,   l_dates, l_nums = group[-1]

        if e_dates == l_dates and e_nums == l_nums:
            continue  # no change detected

        changed_parts: list[str] = []
        if e_dates != l_dates:
            old = ", ".join(sorted(e_dates - l_dates)) or "—"
            new = ", ".join(sorted(l_dates - e_dates)) or "—"
            changed_parts.append(f"date '{old}' -> '{new}'")
        if e_nums != l_nums:
            old = ", ".join(sorted(e_nums - l_nums)) or "—"
            new = ", ".join(sorted(l_nums - e_nums)) or "—"
            changed_parts.append(f"value {old} -> {new}")

        notes.append(
            f"UPDATE on '{topic}' ({', '.join(changed_parts)}): "
            f"latest [{latest_se.event.source} {latest_se.event.timestamp.strftime('%Y-%m-%d')}] "
            f"— \"{latest_se.event.content[:120]}\""
        )

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
        line = _format_event(se) + "\n"
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

    body = header + "\n".join(_format_event(se) for se in selected)

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
