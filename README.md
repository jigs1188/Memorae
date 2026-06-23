# Memorae — Personal Memory Query Engine

A working implementation of a personal-memory query system that ingests raw event streams and answers natural-language queries with time-aware, context-grounded answers.

---

## Setup

1. **Clone and Install**
   ```bash
   git clone <your-repo>
   cd memorae
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

### 4. Dry-run (no LLM — show selected events only)

```bash
python main.py --no-llm
```

### 5. Run the evaluation framework

```bash
# Offline only (no API calls, fast)
python evaluation.py --offline-only

# Full eval including regression tests (requires API)
python evaluation.py
```

---

## Project Structure

```
memorae_mock_events.json      ← dataset (not modified)
memorae/
  config.py          ← constants, config, and .env loader
  llm_client.py      ← unified router
  gemini_client.py   ← Gemini API client
  openai_client.py   ← OpenAI API client
  event_store.py     ← event indexing, scoring, retrieval
  context_builder.py ← token-aware context construction
  query_engine.py    ← 4-step pipeline per query
  main.py            ← CLI entry point
  dashboard_export.py← HTML dashboard generator
  evaluation.py      ← offline + regression + online evals
  requirements.txt
README.md
.env.example         ← template for API keys
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
