"""
event_store.py — In-memory event store with indexing, scoring, and retrieval.

This module answers the question: "Given a query intent, which events matter?"

Retrieval is HYBRID:
  - BM25 (sparse)  — term frequency/IDF-weighted keyword matching via rank_bm25
  - Keyword overlap (exact) — must-include pattern matching, query term scanning
  - Combined relevance score: 0.55 * BM25 + 0.45 * keyword_overlap

Design for scale (10k messages / 1k notes / 500 reminders):
  - BM25 index is built once at load time (O(n log n)).
  - Per-query: BM25 scoring is O(n * avg_doc_len), fully in-memory.
  - For production scale: swap BM25 for Elasticsearch + FAISS for ANN search.
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

# BM25 — graceful fallback if rank_bm25 not installed
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None  # type: ignore

from core.config import SCENARIO_NOW, NOISE_PATTERNS

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
    metadata: dict[str, Any] = field(default_factory=dict)





# ── Loader ─────────────────────────────────────────────────────────────────────

def load_events(path: str) -> list[Event]:
    with open(path, encoding="utf-8") as f:
        raw_events = json.load(f)

    events = []
    for r in raw_events:
        try:
            ts = _parse_ts(r["timestamp"])
            content = r.get("content", "")
            # Apply noise filter: mark low-signal events so downstream
            # components (MemoryExtractor) can skip them.
            ev = Event(
                timestamp=ts,
                source=r.get("source", "unknown"),
                content=content,
                raw=r,
                is_noise=_contains_any(content, NOISE_PATTERNS),
            )
            events.append(ev)
        except Exception as exc:
            logger.warning(f"Skipping malformed event: {exc} | {r}")

    events.sort(key=lambda e: e.timestamp)
    noise_count = sum(1 for e in events if e.is_noise)
    logger.info(
        f"Loaded {len(events)} events from {path} "
        f"({noise_count} flagged as noise, {len(events) - noise_count} signal)"
    )
    return events
