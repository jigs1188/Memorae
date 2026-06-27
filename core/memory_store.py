import logging
import re
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import os
try:
    if os.environ.get('MEMORAE_NO_FAISS') == '1':
        raise ImportError("FAISS explicitly disabled via environment variable")
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from core.memory_extractor import Memory
from core.event_store import Event, ScoredEvent
from core.config import SOURCE_PRIORITY

logger = logging.getLogger(__name__)

# Scoring weights — must sum to 1.0
W_SEMANTIC      = 0.30
W_BM25          = 0.20
W_IMPORTANCE    = 0.15
W_URGENCY       = 0.15
W_RECENCY       = 0.10
W_RELATIONSHIP  = 0.05
W_SOURCE        = 0.05

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Using .split() on the raw string retains trailing punctuation
    (e.g. 'southridge.' never matches 'southridge' in the index).
    This helper fixes that by stripping all non-word, non-space chars first.
    """
    return _PUNCT_RE.sub(" ", text.lower()).split()

class MemoryStore:
    """
    Stores Memory objects and provides Hybrid Retrieval (FAISS + BM25 + Metadata).
    """

    def __init__(self, memories: list[Memory], now: datetime):
        self.memories = memories
        self.now = now
        
        # Initialize models
        self.encoder = None
        self.index = None
        self.bm25 = None
        
        self._build_indices()

    def _build_indices(self):
        if not self.memories:
            return

        corpus = [mem.content for mem in self.memories]
        
        if BM25_AVAILABLE:
            tokenized_corpus = [_tokenize(doc) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built.")

        if FAISS_AVAILABLE:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = self.encoder.encode(corpus)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            logger.info("FAISS vector index built.")

    def stats(self) -> dict:
        total = len(self.memories)
        with_urgency = sum(1 for m in self.memories if self._calculate_urgency(m.content) > 0)
        # Count events flagged as noise during ingestion
        noise_filtered = sum(
            1 for m in self.memories
            if m.raw_event is not None and getattr(m.raw_event, 'is_noise', False)
        )
        return {
            "total": total,
            "signal": total,
            "noise": noise_filtered,
            "with_urgency": with_urgency,
        }

    def _calculate_recency(self, timestamp: datetime) -> float:
        """Score decays over 72 hours (approx 3 days)."""
        dt = (self.now - timestamp).total_seconds()
        hours = max(0, dt / 3600.0)
        # exponential decay
        import math
        return math.exp(-hours / 72.0)

    def _calculate_urgency(self, content: str) -> float:
        """Score urgency using the shared URGENCY_SIGNALS list from config.

        Previously this method only checked 6 of the 20 urgency keywords defined
        in config.py, causing the urgency coverage test to fail and under-surfacing
        actionable events.  Using the shared list ensures a single source of truth.
        """
        from core.config import URGENCY_SIGNALS
        content_lower = content.lower()
        # High-urgency tier: overdue / failure / risk
        if any(kw in content_lower for kw in ("risk", "failure", "overdue", "asap")):
            return 1.0
        # Standard urgency: anything in the shared signals list
        if any(kw in content_lower for kw in URGENCY_SIGNALS):
            return 0.8
        return 0.0

    def _calculate_relationship_score(self, mem: Memory, query: str) -> float:
        query_lower = query.lower()
        score = 0.0
        # relationships values are now list[str] — iterate each person in each relation
        for rel_type, persons in mem.relationships.items():
            if isinstance(persons, list):
                for person in persons:
                    if person.lower() in query_lower:
                        score += 1.0
            else:
                # Backwards-compat: handle old single-string format if present
                if persons.lower() in query_lower:
                    score += 1.0
        return min(score, 1.0)

    def retrieve(self, query: str, top_k: int = 20, min_score: float = 0.1) -> list[ScoredEvent]:
        if not self.memories:
            return []

        query_lower = query.lower()
        
        # 1. BM25 Scores
        bm25_scores = [0.0] * len(self.memories)
        if self.bm25:
            tokenized_query = _tokenize(query)  # strip punctuation before scoring
            raw_scores = self.bm25.get_scores(tokenized_query)
            max_bm25 = max(raw_scores) if max(raw_scores) > 0 else 1.0
            bm25_scores = [s / max_bm25 for s in raw_scores]

        # 2. Semantic Scores
        semantic_scores = [0.0] * len(self.memories)
        if self.index and self.encoder:
            query_emb = self.encoder.encode([query])
            D, I = self.index.search(query_emb, len(self.memories))
            
            # D contains L2 distances. Convert to similarity score [0, 1]
            max_d = max(D[0]) if max(D[0]) > 0 else 1.0
            for i, idx in enumerate(I[0]):
                if idx != -1:
                    semantic_scores[idx] = 1.0 - (D[0][i] / max_d)

        # 3. Combine Hybrid Scores
        scored_events = []
        for i, mem in enumerate(self.memories):
            s_sem = semantic_scores[i]
            s_bm25 = bm25_scores[i]
            s_imp = mem.importance
            s_urg = self._calculate_urgency(mem.content)
            s_rec = self._calculate_recency(mem.timestamp)
            s_rel = self._calculate_relationship_score(mem, query)
            # Source priority: higher-trust sources (calendar, gmail) score higher
            s_src = SOURCE_PRIORITY.get(mem.source, 0.5) if mem.source else 0.5

            final_score = (
                (W_SEMANTIC     * s_sem)  +
                (W_BM25         * s_bm25) +
                (W_IMPORTANCE   * s_imp)  +
                (W_URGENCY      * s_urg)  +
                (W_RECENCY      * s_rec)  +
                (W_RELATIONSHIP * s_rel)  +
                (W_SOURCE       * s_src)
            )

            if final_score >= min_score:
                breakdown = {
                    "semantic": s_sem,
                    "bm25": s_bm25,
                    "importance": s_imp,
                    "urgency": s_urg,
                    "recency": s_rec,
                    "relationship": s_rel,
                }
                
                reason = []
                if s_sem > 0.6: reason.append("semantic match")
                if s_bm25 > 0.6: reason.append("keyword match")
                if s_urg > 0.5: reason.append("urgent")
                
                # We wrap the Memory back into a ScoredEvent to be compatible with QueryEngine
                # Actually, QueryEngine expects ScoredEvent(event: Event, score: float, ...)
                se = ScoredEvent(
                    event=mem.raw_event,
                    score=final_score,
                    breakdown=breakdown,
                    why_selected="; ".join(reason) if reason else "metadata matched",
                    metadata={
                        "project": mem.project,
                        "people": list(mem.people) if mem.people else [],
                        "relationships": mem.relationships
                    }
                )
                scored_events.append(se)

        # Sort descending
        scored_events.sort(key=lambda x: x.score, reverse=True)
        return scored_events[:top_k]
