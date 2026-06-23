# Memorae — Design Document

**Author:** Personal Memory Query Engine Implementation  
**Scenario time:** 2026-04-13 03:00 UTC  
**Dataset:** 200 raw events across WhatsApp, Slack, Gmail, Calendar, Notion, SMS, Chrome extensions

---

## 1. Retrieval Architecture

### Philosophy: Signal-first, not context-first

The core problem with dumping all 200 events into an LLM prompt is not token cost — it's **dilution**. When the context is 80% noise (coffee machine jokes, ride receipts, OTP messages, newsletters), the model's attention is split and signal events are downweighted implicitly.

Our approach: **pre-filter aggressively, score precisely, rank transparently**.

### Retrieval Pipeline (per query)

```
Raw events (200)
      │
      ▼
Stage 1: Noise pre-filter
  • Pattern-match against a "definite noise" blocklist
  • ~45 events removed (social chatter, OTPs, receipts, newsletters)
  • Cost: O(n), no LLM needed
      │
      ▼
Stage 2: Composite scoring (remaining ~155 events)
  • Recency score:   exp(-age_hours / 72)  [half-life = 3 days]
  • Urgency score:   keyword match on 20+ deadline/action signals
  • Relevance score: keyword overlap with query-specific terms
  • Source score:    reminder > calendar > gmail > notion > slack > sms > chrome
  • Combined:        0.35×urgency + 0.30×relevance + 0.20×recency + 0.15×source
      │
      ▼
Stage 3: Must-include boost
  • Query-specific anchor terms always surfaced (e.g., "UIE" for UIE query)
  • Score boosted by +0.3 for must-include matches
      │
      ▼
Stage 4: Top-K selection (default K=30)
      │
      ▼
Stage 5: Contradiction resolution
  • Rule-based: track named entities (UIE deadline, procurement estimate, clause 8)
  • Later events supersede earlier ones for the same entity
  • Annotation injected into context header
      │
      ▼
Stage 6: Token-budget packing
  • Greedy fill: highest-score events first until ~8,000 token soft target
  • Hard limit: 100,000 tokens
  • Approx 4 chars/token (conservative estimate)
      │
      ▼
LLM (Gemini, 0.2 temperature)
```

### Scaling to 10k messages / 1k notes / 500 reminders

| Component | Current (200 events) | Production scale |
|-----------|---------------------|-----------------|
| Noise filter | In-memory string match | Redis blocklist + regex index |
| Scoring | In-memory per-query | Pre-computed nightly; delta updates on new events |
| Relevance | Hybrid (BM25 + Keyword overlap) | Embedding similarity (semantic) |
| Top-K | Linear scan | Approximate nearest-neighbor (FAISS / Pinecone) |
| Contradiction | Rule-based, 3 entities | Entity resolution model + fact graph |
| Context packing | Greedy by score | ILP optimization to maximize coverage within budget |

---

## 2. Memory Architecture

### Three-tier memory model

```
Tier 1: Hot Memory (last 48 hours)
  • Always fully loaded into retrieval candidate pool
  • Events scored with recency boost
  • Contains: today's calendar, recent messages, active tasks

Tier 2: Warm Memory (last 2 weeks)
  • Indexed; retrieved only if query-relevant
  • Compressed via clustering: similar events merged
  • Example: 8 "coffee machine" messages → 1 deduplicated record

Tier 3: Cold Memory (older / archived)
  • Summarized into entity profiles: "UIE proposal", "Southridge SOW"
  • Summaries injected as synthetic events at retrieval time
  • Full events only loaded for specific entity queries
```

### Entity Tracking

The system implicitly tracks named entities and resolves their state:

| Entity | State at 2026-04-13 03:00 UTC |
|--------|-------------------------------|
| UIE proposal | Due today (Apr 13) 14:30 IST; appendix needed; Ravi unconfirmed |
| UIE deadline | Updated Apr 10 → Apr 13 (superseded) |
| Procurement estimate | $42k → $48.5k (Cedric update Apr 10) |
| Southridge SOW | Clause 8 approved Apr 11; negotiation today Apr 14 |
| Hiring rubric | Overdue as of Apr 12; Rhea waiting |
| Car insurance | Due Apr 15 (before portal maintenance) |
| Pari school form | Internal deadline Apr 16 |

---

## 3. Context Construction Strategy

### Why 8,000 tokens (not 100,000)?

Even with a 100k token budget, injecting all 200 events (≈50k chars ≈ 12.5k tokens) would:
1. Include noise that confuses the model
2. Include irrelevant personal preferences (food orders, elevator anecdotes)
3. Bury high-signal items in low-signal content
4. Cost more without better answers

**Empirically, 15–25 well-selected events answer queries better than 200 raw events.**

### Context structure

```
=== PERSONAL MEMORY CONTEXT ===
Current time: 2026-04-13 03:00 UTC
Query: [query text]

--- Updates & Corrections ---
• UIE deadline UPDATED: Apr 10 → Apr 13 (per Apr 9 Slack message)
• Procurement estimate UPDATED: $42k → $48.5k (Cedric email Apr 10)

--- Relevant Events (ranked by importance) ---
[2026-04-13 09:00 UTC | calendar] UIE proposal review with Nina Apr 13 14:30 IST.
[2026-04-13 03:05 UTC | slack] Hiring rubric was due Apr 12. Please send...
[2026-04-13 07:45 UTC | slack] If Ravi does not confirm data-room access by EOD...
...
```

The **Updates & Corrections** section is injected before events to prime the model with resolved facts, preventing it from using stale data even if stale events appear lower in the context.

---

## 4. Contradiction and Recency Handling

### Contradiction types observed in the dataset

| Type | Example | Resolution |
|------|---------|------------|
| Deadline update | Apr 10 → Apr 13 for UIE | Latest timestamp wins; annotate change |
| Estimate revision | $42k → $48.5k procurement | Latest email supersedes earlier |
| Status resolution | Clause 8 blocked → approved | Track status-changing keywords |
| Cancellation | Apr 12 UIE work block cancelled | Calendar cancellation events flagged |
| Karan availability | Can cover pickup → cannot | Latest WhatsApp message wins |

### General rule

> For any tracked entity, the latest timestamp is authoritative. Earlier events are retained but ranked lower and annotated as potentially superseded.

### Deduplication

Repeated identical-content events (e.g., 5 different people saying "coffee machine is making the dramatic steam noise again") are:
1. Detected via content fingerprinting (normalized lowercase match)
2. Collapsed to one canonical event
3. The count of repeats is included in reasoning output

---

## 5. Failure Modes

### Known failure modes and mitigations

| Failure | Trigger | Mitigation |
|---------|---------|------------|
| Stale facts in answer | User asks about UIE before contradiction resolution runs | Always inject correction header before events |
| Over-reliance on recent noise | Burst of spam just before query | Source-tier scoring caps WhatsApp/SMS weight |
| Missing implicit deadlines | "Send before Mom's appointment" (no explicit date) | Urgency keywords detect "before" + "appointment" |
| False positives in urgency | "No urgent matters" flagged as urgent | Negation-aware urgency scoring (planned improvement) |
| LLM hallucination | Model invents a deadline not in context | Low temperature (0.2); "grounded in events" in system prompt |
| API quota exhaustion | Free-tier limits hit | 7-model fallback chain with exponential backoff |
| Contradictory answer if both old and new deadline in context | Both Apr 10 and Apr 13 events selected | Contradiction header primes model; lower score for older event |
| Entity confusion | Multiple "Nina" references (Nina @northstar vs internal) | Domain in email used to disambiguate; external > internal if email present |

### What the system cannot do (without more data)

- Know if a task is **done** (no completion signal in the stream)
- Detect **implicit** commitments ("I'll look into it later")
- Handle **voice memos** or **image attachments**
- Track **reading status** (did Aarav read Nina's email?)

---

## 6. Scaling to Larger Personal-Memory Datasets

### Architecture at scale

```
Ingestion Pipeline
  WhatsApp / Slack / Gmail / Calendar / Notion
          │
          ▼
  Stream Processor (Kafka / Pub/Sub)
  • Deduplicate within 30-min windows
  • Tag source + extract entities (NER)
  • Compute urgency score
          │
          ▼
  Tiered Storage
  • Hot:  Redis (last 48h)           → sub-10ms retrieval
  • Warm: Postgres + pgvector         → semantic search
  • Cold: S3 + summary index          → compressed entity profiles
          │
          ▼
  Query API
  • BM25 keyword retrieval (Elasticsearch)
  • Semantic reranking (embedding model, e.g., text-embedding-004)
  • Rule-based contradiction resolver
  • Token-budget context packer
          │
          ▼
  LLM (Gemini, model tier auto-selected by query complexity)
```

### Performance targets at scale

| Metric | Target |
|--------|--------|
| P50 retrieval latency | < 200ms |
| P99 end-to-end (with LLM) | < 5s |
| Context quality (NDCG@10) | > 0.8 |
| Noise events in context | < 5% |

---

## 7. Evaluation Framework

### 7.1 Offline Evals (no LLM)

Test retrieval quality deterministically:

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `noise_filtered` | Noise events excluded from results | 0 noise events in top-30 |
| `uie_deadline_update` | Apr 13 surfaces, not stale Apr 10 | Apr 13 event in top results |
| `procurement_estimate` | $48.5k ranked above $42k | Latest event is top-ranked |
| `recency_bias` | Apr 12–13 events beat Apr 1 events | Top event timestamp ≥ Apr 9 |
| `calendar_beats_whatsapp` | Calendar events score higher for meetings | avg_calendar > avg_whatsapp |
| `clause8_resolved` | Clause 8 shows as approved | Latest event contains "approved" |
| `urgency_detection_coverage` | Deadline events flagged | ≥ 20 events with urgency signals |

**Run:** `python evaluation.py --offline-only`  
**Expected pass rate:** 100%

---

### 7.2 Regression Tests (LLM-based golden set)

| Test | Query | Must contain | Pass criterion |
|------|-------|-------------|----------------|
| `focus_today_uie_review` | Focus today | nina, 14:30, uie | ≥ 2/3 present |
| `risk_overdue_rubric` | Risk of missing | rubric, rhea | ≥ 1/2 present |
| `procrastination_reimbursement` | Procrastinating | reimburse, screenshots, nudge | ≥ 1/3 present |
| `uie_summary_key_facts` | UIE summary | nina, appendix, risk, 48.5, 14:30 | ≥ 3/5 present |
| `uie_no_stale_deadline` | UIE summary | (absence of uncorrected Apr 10) | Apr 10 either absent or corrected |

**What "good" means for subjective queries:**

For "What should I focus on today?", a good answer:
1. **Specific** — names the meeting, person, and time (not "review the proposal")
2. **Time-aware** — accounts for it being 03:00 UTC (early morning, before standup)
3. **Prioritized** — UIE review before hiring rubric before personal tasks
4. **Complete** — doesn't miss the hiring rubric that was overdue yesterday
5. **Grounded** — every item is traceable to a specific event in the dataset

---

### 7.3 Online Evals (production monitoring)

| Metric | Measurement | Target |
|--------|------------|--------|
| End-to-end latency | Wall clock per query | P50 < 5s, P99 < 15s |
| Context efficiency | tokens_used / budget | < 120% of soft budget |
| Answer word count | len(answer.split()) | 50–600 words |
| Model tier used | Which fallback was triggered | Log distribution |
| User thumbs-up rate | Explicit feedback (if UI exists) | > 75% |
| Fact hallucination rate | Spot-check: claims not in context | < 5% |

---

## 8. Optimization for < 2s Latency, 80% Cost Reduction

### Current bottlenecks

| Component | Typical time | Cost |
|-----------|-------------|------|
| Event scoring (200 events) | ~10ms | — |
| Context building | ~5ms | — |
| LLM generation | 2–8s | ~$0.01–0.05 |

The **LLM call is the sole bottleneck**. Everything else is sub-50ms.

### Strategy to hit < 2s at 80% lower cost

#### 1. Precompute and cache (biggest win)

```
Nightly job:
  • Run "focus today" query at midnight → cache result
  • Cache invalidated when any high-score event arrives
  • 90% of "focus today" queries served from cache in < 50ms

Query-time:
  • Check cache first → hit rate ~70% for recurring queries
  • Cache keyed by: (query_intent, date, top-event-hashes)
```

#### 2. Route by complexity

```
Simple queries (single entity lookup):
  → Use gemini-flash model: $0.0001/1k tokens, ~0.5s latency

Complex queries (cross-entity reasoning):
  → Use gemini-pro model: higher quality needed

Routing rule: count of distinct entities in query
  • 1–2 entities → flash
  • 3+ entities → pro
```

#### 3. Tiered context

```
Short-context mode (< 2s target):
  • Max 15 events, 3,000 tokens
  • Only hot memory (last 48h)
  • No contradiction annotation (use cached resolution)

Full-context mode (best quality):
  • Max 30 events, 8,000 tokens
  • Hot + warm memory
  • Full contradiction resolution
```

#### 4. Pre-summarize warm/cold memory

```
Entity summaries (computed nightly, ~200 tokens each):
  "UIE Proposal: Due Apr 13 14:30 IST. Appendix pending. 
   Ravi unconfirmed on data-room. Procurement: $48.5k."

At query time: inject summary (not raw events) for entities older than 48h.
  → Reduces raw event count by ~60%
  → Still accurate because summaries are entity-resolved
```

#### 5. Streaming responses

```
Stream LLM output to user as it generates.
  → Perceived latency drops from 3s to ~0.8s (time to first token)
  → User sees answer appearing, not waiting for full completion
```

### Tradeoffs

| Optimization | Latency gain | Cost reduction | Quality tradeoff |
|-------------|-------------|----------------|-----------------|
| Response caching | ★★★★★ | ★★★★ | Staleness risk (mitigated by cache invalidation) |
| Model routing (flash) | ★★★★ | ★★★★★ | Slightly lower quality on complex queries |
| Tiered context (3k tokens) | ★★★ | ★★★ | May miss low-scored but relevant events |
| Entity summaries | ★★ | ★★★ | Summary errors propagate; need quality checks |
| Streaming | ★★★★ (perceived) | ✗ | None |

**Combined estimate:** Caching + model routing + tiered context → P50 latency ~0.8s (80% reduction), cost ~$0.002/query (80%+ reduction). Quality degradation: ~10% on complex multi-entity queries, detectable only on regression tests.

---

## 9. External API Usage & Setup

### API Requirements
The system uses external LLM APIs for the generation step (Step 3). It abstracts the provider via a unified client, supporting both **Google Gemini** and **OpenAI**.

- **Google Gemini (Default)**
  - **Setup**: Requires a `GEMINI_API_KEY` in the `.env` file. A free-tier key is pre-configured in the codebase for ease of evaluation.
  - **Models**: `gemini-2.5-flash` with a fallback chain extending to `gemini-2.5-pro` if rate limits are hit.
  - **Expected Cost**: $0 (using the free tier, bounded by 5 RPM).

- **OpenAI (Optional Alternative)**
  - **Setup**: Requires `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in the `.env` file.
  - **Models**: `gpt-4o` as the default, falling back to `gpt-4o-mini` and `gpt-4-turbo`.
  - **Expected Cost**: ~$0.02 to run all 5 preset queries (assuming ~15k input tokens and ~2k output tokens total).

The retrieval pipeline itself (Step 1 & 2) runs entirely locally and incurs no API costs.
