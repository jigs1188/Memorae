# Memorae Design Document

## 1. Architecture Rationale
Memorae is designed as an offline-capable, highly modular personal memory intelligence engine. The core philosophy is to decouple the retrieval logic (Information Retrieval) from the generative logic (LLMs). This separation ensures that the context provided to the model is deterministic, rankable, and inspectable independently of the LLM chosen. By utilizing a hybrid retrieval strategy and a deterministic contradiction resolver, we prevent hallucination of critical facts (e.g., dates and numbers) before the LLM even sees the data.

## 2. Memory Extraction
Raw events (`core/event_store.py`) are passed to the `MemoryExtractor` (`core/memory_extractor.py`) where unstructured text is transformed into semi-structured `Memory` objects.
- **Noise Filtering:** Events containing known low-signal phrases (e.g., OTPs, coffee machine chat) are skipped early.
- **Entity Identification:** Uses a combination of predefined dataset keywords (for deterministic tracking) and spaCy's NER (for zero-shot fallback on unseen entities like `ORG` or `PERSON`).
- **State Transitions:** Captures generic workflow events (e.g., `X is now approved`, `Y is blocked`) using regex heuristics, mapping them to explicit `state_transitions` properties.

## 3. Hybrid Retrieval & Ranking
The `MemoryStore` (`core/memory_store.py`) employs a multi-dimensional scoring algorithm to rank events based on their relevance to a user's query.
The final relevance score is a weighted sum:
- **Semantic (30%)**: Captures intent via FAISS L2 distance (`all-MiniLM-L6-v2`).
- **BM25 (20%)**: Captures exact terminology and rare keywords (`rank_bm25`).
- **Importance (20%)**: Boosts memories containing imperative signals (e.g., `must`, `critical`).
- **Urgency (15%)**: Boosts memories with immediate deadlines or risks.
- **Recency (10%)**: Exponentially decays scores for older events (72-hour half-life).
- **Relationship (5%)**: Minor boost if queried people match the memory's extracted `relationships`.
- **Source Trust (5%)**: Base confidence multiplier (e.g., Calendar events > WhatsApp chatter).

## 4. Contradiction Handling
Before context is packed for the LLM, the `ContextBuilder` runs a deterministic contradiction detector.
1. It groups events into "topics" using a lightweight 2-word fingerprint (ignoring stop words).
2. It scans for explicit dates (`\d{4}-\d{2}-\d{2}`, `Monday`, etc.) and numbers (`$42k`, `48.5`).
3. If the earliest and latest events in a topic cluster disagree on these values, it emits an `UPDATE` annotation.
This prevents the LLM from being confused by stale deadlines, explicitly instructing it that a value has changed.

## 5. Project Layer
The `ProjectBuilder` (`core/project_builder.py`) aggregates individual memories into holistic `ProjectState` objects. It computes a top-level `health` status (`green`, `yellow`, `red`) by counting open commitments, overdue items, and blocked dependencies. This structured summary is injected directly into the LLM system prompt, allowing the model to quickly answer "What is the status of X?" without needing to resynthesize the entire event timeline.

## 6. Context Construction
The `ContextBuilder` (`core/context_builder.py`) is responsible for token-aware context packing. It uses a greedy algorithm:
- Estimates token cost (approx. 4 chars per token).
- Packs the highest-scoring events first until the `TARGET_CONTEXT_TOKENS` budget is hit.
- Allows highly relevant events (score > 0.55) to overflow up to a `MAX_CONTEXT_TOKENS` hard limit.
- Formats the final context string, appending contradiction notes at the top.

## 7. Evaluation Methodology
Memorae relies on a multi-tiered evaluation framework (`evaluation.py`):
- **Offline Evaluation**: Deterministic tests verifying that noise is filtered, recency bias works, and explicit updates (e.g., a moved deadline) are ranked correctly.
- **Regression Evaluation**: End-to-end golden-set tests using LLM generations. Ensures critical phrases (e.g., `14:30`, `UIE`, `Nina`) appear in responses to standard queries.
- **Generalization Evaluation**: Tests unseen queries to ensure the system structure handles out-of-distribution questions robustly.
- **Online Metrics**: Tracks latency, context utilization (token efficiency), and model fallback behavior.

## 8. Scalability
- **Vector Search**: Integrating FAISS allows the semantic search to scale efficiently. In production, this can be offloaded to a dedicated vector database (e.g., Pinecone, Milvus).
- **In-Memory BM25**: While `rank_bm25` is in-memory and fast for thousands of events, scaling to millions of events would require migrating to an inverted index datastore like Elasticsearch.
- **Batch Processing**: The `MemoryExtractor` is stateless and can be parallelized via multiprocessing for bulk ingestion.

## 9. Tradeoffs
- **Token Estimation vs Accuracy**: We use a fast character-to-token heuristic (N / 4) rather than importing a heavy tokenizer like `tiktoken`. This slightly overestimates context size but ensures fast processing without bloated dependencies.
- **Simple Topic Fingerprinting**: The contradiction detector uses the first two meaningful words to cluster events. While computationally cheap and highly effective for brief chat messages, it may incorrectly cluster distinct events if they start with the same keywords.
- **LLM Independence**: Because the logic resides in the extraction and retrieval layers, the LLM is treated as a commodity summarizer. We sacrifice complex Agentic chains (like ReAct) in favor of fast, single-pass generation.

## 10. Future Improvements
- **Agentic Retrieval (Multi-hop)**: Allow the LLM to query the `MemoryStore` iteratively for complex analytical questions.
- **Coreference Resolution**: Improve the `MemoryExtractor` so that pronouns ("he said") resolve back to the last extracted entity.
- **Streaming Ingestion**: Update the FAISS and BM25 indices incrementally rather than rebuilding them per query session.

## 11. Optimization Question
**If the latency target becomes under 2 seconds and cost must drop by 80%, what would you change?**

To hit sub-2 second latency and an 80% cost reduction, the system must shift from heavy runtime generation to aggressive precomputation and cheaper routing:
1. **Precomputation & Summarization**: Instead of packing 30 raw events into every LLM prompt, the `ProjectBuilder` and `MemoryExtractor` would generate daily, rolling summaries of projects and people during ingestion.
2. **Memory Tiers**: Segregate the `MemoryStore` into a "hot" tier (recent events, active projects) stored in memory and a "cold" tier stored on disk.
3. **Model Routing**: Route simple factual queries (e.g., "When is the UIE deadline?") to a significantly smaller, cheaper model (like `gemini-2.5-flash-lite` or a fine-tuned open-weight model like `Llama-3-8B`) using an intent classifier, reserving expensive frontier models (e.g., `gpt-4o`) only for complex synthesis or ambiguity resolution.
4. **Caching**: Implement a semantic cache (e.g., Redis) to serve identical or semantically identical queries instantly without hitting the LLM.
5. **Retrieval**: Replace FAISS embeddings with lightweight sparse embeddings (e.g., SPLADE) which are often cheaper to compute and execute on CPUs.
