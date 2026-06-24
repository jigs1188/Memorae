# Memorae — Personal Memory Intelligence Engine

A production-grade personal memory query system that ingests raw event streams (WhatsApp, Slack, Gmail, Calendar, Notion) and answers natural-language queries using **hybrid BM25 + keyword retrieval**, contradiction resolution, and LLM generation (Gemini or OpenAI).

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org) [![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com) [![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## 🎯 Assessor's Guide (Where to find the logic)

To review the specific steps requested in the assignment, please check the following core files:

1. **Step 1 & 2: Source Selection & Context Construction** 
   - `core/event_store.py`: Contains the hybrid BM25 + keyword scoring, noise filtering, and top-K logic.
   - `core/context_builder.py`: Contains token-budget packing and contradiction resolution.
2. **Step 3 & 4: Answer Generation & Reasoning**
   - `core/query_engine.py`: Orchestrates the 4-step pipeline and constructs the reasoning/uncertainty output.
   - `llm/gemini_client.py`: The robust fallback chain and retry logic.
3. **Evaluation Framework**
   - `evaluation/evaluation.py`: Contains the Offline (deterministic), Regression (LLM golden-set), and Online (latency/efficiency) evaluation suites.

---

## Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/jigs1188/Memorae.git
   cd Memorae
   pip install -r requirements.txt
   ```

2. **API Key Configuration (.env)**
   The project supports both **OpenAI** and **Google Gemini** APIs. A free-tier Gemini key is pre-configured and will work out of the box, but you can also use your own OpenAI or Gemini API keys.

   Copy the provided `.env.example` to `.env`:
   ```bash
   cp ../.env.example ../.env
   ```
   
   Open `.env` and set your preferred provider:
   - **For OpenAI**: Set `LLM_PROVIDER=openai` and add your `OPENAI_API_KEY`. The default model is `gpt-4o`, but you can change it via the `OPENAI_MODEL` variable.
   - **For Gemini**: Set `LLM_PROVIDER=gemini`. You can use the pre-configured key or provide your own `GEMINI_API_KEY`.

## Usage

### 1. Run all queries

```bash
python main.py
```

Results are saved to `results.json` in the current directory. 
🎉 **New**: A beautiful, interactive UI is also automatically generated at `dashboard.html`. Simply double-click it in your browser to explore the answers, context, and reasoning!

### 2. Run a custom query

```bash
python main.py --query "What did Nina ask me to prepare?"
```

### 3. Run via FastAPI (HTTP)

```bash
# Install all deps first:
pip install -r requirements.txt

# Start the server:
python -m uvicorn api.api:app --host 0.0.0.0 --port 8000
```

Interactive docs at: **http://localhost:8000/docs**

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Run any natural-language query |
| `GET /queries` | List all 5 preset queries |
| `POST /queries/all` | Run all 5 preset queries |
| `GET /health` | Health check + provider info |
| `GET /results` | Return cached `results.json` |
| `GET /events/stats` | Event store statistics |

**Example API call:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What should I focus on today?"}'
```

### 4. Run via Docker (single command)

```bash
docker-compose up
```

This builds and starts the FastAPI server on port 8000. The API is ready at **http://localhost:8000**.

To also run the CLI analysis (generates results.json + dashboard.html):
```bash
docker-compose --profile cli up
```

### 5. Run a custom query

```bash
python main.py --query "What did Nina ask me to prepare?"
```

### 6. Dry-run (no LLM — show selected events only)

```bash
python main.py --no-llm
```

### 7. Run the evaluation framework

```bash
# Offline only (no API calls, fast)
python -m evaluation.evaluation --offline-only

# Full eval including regression tests (requires API)
python -m evaluation.evaluation
```

---

## Project Structure

```
memorae_mock_events.json      ← dataset (not modified)
core/
  config.py          ← constants, config, and .env loader
  event_store.py     ← event indexing, scoring, retrieval
  context_builder.py ← token-aware context construction
  query_engine.py    ← 4-step pipeline per query
llm/
  llm_client.py      ← unified router
  gemini_client.py   ← Gemini API client
  openai_client.py   ← OpenAI API client
api/
  api.py             ← FastAPI web server
ui/
  dashboard_export.py← HTML dashboard generator
evaluation/
  evaluation.py      ← offline + regression + online evals
  smoke_test.py      ← quick system checks
main.py              ← CLI entry point
requirements.txt
README.md
.env.example           ← template for API keys
```

---

## API Configuration

The system uses a **Unified LLM Client** (`llm_client.py`) that supports both OpenAI and Google Gemini APIs. It handles rate limits, daily quotas, and model fallbacks transparently.

### 1. OpenAI
Set `LLM_PROVIDER=openai` and add your `OPENAI_API_KEY` in `.env`.
- Default model: `gpt-4o`
- Fallback chain: `gpt-4o` → `gpt-4o-mini` → `gpt-4-turbo` → `gpt-3.5-turbo`
- Rate limits: Auto-retries based on API headers.

### 2. Google Gemini (Free Tier Included)
Set `LLM_PROVIDER=gemini` or simply leave the `.env` blank (it falls back to the included free-tier key).
- Primary model: `gemini-2.5-flash`
- Fallback chain: `gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash` → `gemini-2.5-pro`
- Rate limits: The client automatically detects the strict 5 RPM free-tier limit, waits the required ~40s, and retries. If daily quota is hit, it immediately skips to the next fallback model.

**Expected cost**: With the pre-configured Gemini key, running all 5 queries costs $0. With an OpenAI API key on `gpt-4o`, running all queries costs ~$0.02.

---

## CLI Options

| Flag | Description |
|------|-------------|
| `--query "..."` | Run a single custom query |
| `--output file.json` | Output file (default: `results.json`) |
| `--no-llm` | Dry-run: show event selection without LLM |
| `--data path/to/events.json` | Custom data file path |

---

## Output Format

Each query result in `results.json`:

```json
{
  "query": "What should I focus on today?",
  "answer": "...",
  "model_used": "gemini-2.0-flash",
  "context_stats": {
    "token_estimate": 3240,
    "events_used": 18,
    "events_dropped": 4
  },
  "selected_context": [
    {
      "timestamp": "2026-04-13T09:00:00+00:00",
      "source": "calendar",
      "content": "UIE proposal review with Nina Apr 13 14:30 IST.",
      "relevance_score": 0.872
    }
  ],
  "reasoning": {
    "selection_strategy": "...",
    "why_selected": [...],
    "why_ignored": "...",
    "uncertainty": "...",
    "contradiction_resolution": [...]
  },
  "contradiction_notes": [
    "UIE deadline UPDATED: earlier deadline 'Apr 10' was superseded..."
  ]
}
```
