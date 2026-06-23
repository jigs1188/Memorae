"""
event_store.py — In-memory event store with indexing, scoring, and retrieval.

This module answers the question: "Given a query intent, which events matter?"

Design for scale (10k messages / 1k notes / 500 reminders):
  - Events are scored once at load time for recency and source priority.
  - Keyword/signal matching is done with lightweight string operations (O(n)).
  - For production scale, this would be backed by a vector DB (e.g., Pinecone,
    Weaviate) for semantic retrieval, plus a metadata index (SQLite / Redis)
    for fast deadline/date range filtering.
  - The scorer is composable: add new dimensions without touching retrieval logic.
"""

from __future__ import annotations

import json
import re
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from dateutil import parser as dateutil_parser

from config import (
    NOISE_PATTERNS,
    URGENCY_SIGNALS,
    SOURCE_PRIORITY,
    WEIGHT_RECENCY,
    WEIGHT_URGENCY,
    WEIGHT_RELEVANCE,
    WEIGHT_SOURCE,
    MAX_SELECTED_EVENTS,
    SCENARIO_NOW,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> datetime:
    dt = dateutil_parser.parse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _contains_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(p in low for p in patterns)


def _keyword_overlap_score(text: str, keywords: list[str]) -> float:
    """Count how many keywords appear in text, normalized to [0, 1]."""
    if not keywords:
        return 0.0
    low = text.lower()
    hits = sum(1 for kw in keywords if kw in low)
    return min(hits / max(len(keywords) * 0.3, 1), 1.0)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Event:
    timestamp: datetime
    source: str
    content: str
    raw: dict[str, Any] = field(repr=False)

    # Derived at load time
    is_noise: bool = False
    has_urgency: bool = False
    source_weight: float = 0.5

    def age_hours(self, now: datetime) -> float:
        delta = now - self.timestamp
        return max(delta.total_seconds() / 3600.0, 0.0)


@dataclass
class ScoredEvent:
    event: Event
    score: float
    why_selected: str
    breakdown: dict[str, float] = field(default_factory=dict)


# ── EventStore ─────────────────────────────────────────────────────────────────

class EventStore:
    """
    Holds all events and exposes query-time retrieval.

    Retrieval pipeline (per query):
      1. Pre-filter:  remove definite noise
      2. Keyword scan: boost events containing query-relevant terms
      3. Temporal scoring: prefer recent, penalise stale
      4. Urgency scoring: boost events with deadline/action signals
      5. Source scoring: prioritize calendar > gmail > notion > slack …
      6. Rank, deduplicate, return top-k
    """

    def __init__(self, events: list[Event], now: datetime):
        self.events = events
        self.now = now
        self._index_events()

    # ── Build-time indexing ────────────────────────────────────────────────────

    def _index_events(self) -> None:
        for ev in self.events:
            ev.is_noise = _contains_any(ev.content, NOISE_PATTERNS)
            ev.has_urgency = _contains_any(ev.content, URGENCY_SIGNALS)
            ev.source_weight = SOURCE_PRIORITY.get(ev.source, 0.5)

    # ── Scoring dimensions ─────────────────────────────────────────────────────

    def _recency_score(self, ev: Event) -> float:
        """Exponential decay: half-life ≈ 3 days (72 hrs)."""
        age = ev.age_hours(self.now)
        return math.exp(-age / 72.0)

    def _urgency_score(self, ev: Event) -> float:
        base = 0.6 if ev.has_urgency else 0.0
        # Extra boost for events very close to now (within 48 h)
        age = ev.age_hours(self.now)
        if age <= 48:
            base = min(base + 0.3, 1.0)
        return base

    def _relevance_score(self, ev: Event, keywords: list[str]) -> float:
        return _keyword_overlap_score(ev.content, keywords)

    def _source_score(self, ev: Event) -> float:
        return ev.source_weight

    def _composite_score(
        self, ev: Event, keywords: list[str]
    ) -> tuple[float, dict[str, float]]:
        r = self._recency_score(ev)
        u = self._urgency_score(ev)
        rel = self._relevance_score(ev, keywords)
        s = self._source_score(ev)

        total = (
            WEIGHT_RECENCY  * r
            + WEIGHT_URGENCY  * u
            + WEIGHT_RELEVANCE * rel
            + WEIGHT_SOURCE    * s
        )
        return total, {"recency": r, "urgency": u, "relevance": rel, "source": s}

    # ── Public retrieval API ───────────────────────────────────────────────────

    def retrieve(
        self,
        *,
        keywords: list[str],
        exclude_noise: bool = True,
        top_k: int = MAX_SELECTED_EVENTS,
        min_score: float = 0.05,
        must_include_patterns: Optional[list[str]] = None,
    ) -> list[ScoredEvent]:
        """
        Return the top-k most relevant, scored events for a query.

        Parameters
        ----------
        keywords:               Query-specific terms for relevance scoring.
        exclude_noise:          Drop definite-noise events unless they match keywords.
        top_k:                  Max events to return.
        min_score:              Hard floor on composite score.
        must_include_patterns:  Events containing any of these strings are always
                                included regardless of score (e.g., a named topic).
        """
        scored: list[ScoredEvent] = []

        for ev in self.events:
            # Pre-filter: skip noise unless query explicitly targets it or must_include
            if exclude_noise and ev.is_noise:
                if must_include_patterns and _contains_any(
                    ev.content, must_include_patterns
                ):
                    pass  # keep
                elif _keyword_overlap_score(ev.content, keywords) < 0.3:
                    continue

            score, breakdown = self._composite_score(ev, keywords)

            # Boost for must_include matches
            if must_include_patterns and _contains_any(
                ev.content, must_include_patterns
            ):
                score = min(score + 0.3, 1.0)
                breakdown["must_include_boost"] = 0.3

            if score < min_score:
                continue

            why = self._explain_selection(ev, breakdown, keywords)
            scored.append(ScoredEvent(event=ev, score=score, breakdown=breakdown, why_selected=why))

        # Sort descending by score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _explain_selection(
        self, ev: Event, breakdown: dict[str, float], keywords: list[str]
    ) -> str:
        parts = []
        if breakdown.get("recency", 0) > 0.7:
            parts.append("very recent")
        elif breakdown.get("recency", 0) > 0.4:
            parts.append("recent")
        if breakdown.get("urgency", 0) > 0.5:
            parts.append("has urgency/deadline signal")
        if breakdown.get("relevance", 0) > 0.2:
            matched = [kw for kw in keywords if kw in ev.content.lower()]
            parts.append(f"keyword match: [{', '.join(matched[:3])}]")
        if breakdown.get("source", 0) > 0.75:
            parts.append(f"high-priority source ({ev.source})")
        if breakdown.get("must_include_boost"):
            parts.append("matches topic filter")
        return "; ".join(parts) if parts else "marginal relevance"

    # ── Utilities ──────────────────────────────────────────────────────────────

    def get_all_with_pattern(self, patterns: list[str]) -> list[Event]:
        """Return every event whose content contains any of the patterns."""
        return [
            ev for ev in self.events if _contains_any(ev.content, patterns)
        ]

    def stats(self) -> dict:
        noise_count = sum(1 for e in self.events if e.is_noise)
        urgent_count = sum(1 for e in self.events if e.has_urgency)
        return {
            "total": len(self.events),
            "noise": noise_count,
            "with_urgency": urgent_count,
            "signal": len(self.events) - noise_count,
        }


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_events(path: str) -> list[Event]:
    with open(path, encoding="utf-8") as f:
        raw_events = json.load(f)

    events = []
    for r in raw_events:
        try:
            ts = _parse_ts(r["timestamp"])
            ev = Event(
                timestamp=ts,
                source=r.get("source", "unknown"),
                content=r.get("content", ""),
                raw=r,
            )
            events.append(ev)
        except Exception as exc:
            logger.warning(f"Skipping malformed event: {exc} | {r}")

    events.sort(key=lambda e: e.timestamp)
    logger.info(f"Loaded {len(events)} events from {path}")
    return events
