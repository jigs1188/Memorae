"""
config.py — System-wide constants, loaded from .env (root directory).

The system supports two LLM providers:
  • OpenAI  — set LLM_PROVIDER=openai  and OPENAI_API_KEY in .env
  • Gemini  — set LLM_PROVIDER=gemini  and GEMINI_API_KEY in .env

If LLM_PROVIDER is not set, the system auto-detects:
  - If OPENAI_API_KEY is present → OpenAI
  - Otherwise                    → Gemini (free tier key included)
"""

import os
from pathlib import Path

# ── Load .env from project root (parent of memorae/) ──────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # dotenv optional; fall back to OS env vars

# ── Scenario time ──────────────────────────────────────────────────────────────
SCENARIO_NOW = "2026-04-13T03:00:00Z"

# ── Provider selection ─────────────────────────────────────────────────────────
# Explicit > auto-detect
_provider_env = os.getenv("LLM_PROVIDER", "").strip().lower()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()

if _provider_env in ("openai", "gemini"):
    LLM_PROVIDER = _provider_env
elif OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    LLM_PROVIDER = "openai"
else:
    LLM_PROVIDER = "gemini"

# ── OpenAI model ───────────────────────────────────────────────────────────────
# Best model for this task: gpt-4o (excellent reasoning + long context)
# Cheaper alternative:      gpt-4o-mini
# Override via:             OPENAI_MODEL=gpt-4o-mini in .env
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()

# Fallback chain if the chosen model is unavailable / overloaded
OPENAI_MODEL_FALLBACK_CHAIN = [
    OPENAI_DEFAULT_MODEL,
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]
# Deduplicate while preserving order
seen = set()
OPENAI_MODEL_FALLBACK_CHAIN = [
    m for m in OPENAI_MODEL_FALLBACK_CHAIN
    if not (m in seen or seen.add(m))
]

# ── Gemini model fallback chain ────────────────────────────────────────────────
# Verified against genai.list_models() — these are all available for generateContent
GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemini-3.5-flash",          # Best working model on provided key
    "gemini-3.1-flash-lite",     # Latest lite model
    "gemini-flash-latest",       # Standard flash latest
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
]

# ── Context budget ─────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS    = 100_000   # hard ceiling for LLM context window
TARGET_CONTEXT_TOKENS = 8_000     # soft target tokens per query
MAX_SELECTED_EVENTS   = 30        # max events before token-counting

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHT_SEMANTIC   = 0.30
WEIGHT_BM25       = 0.20
WEIGHT_IMPORTANCE = 0.20
WEIGHT_URGENCY    = 0.15
WEIGHT_RECENCY    = 0.10
WEIGHT_RELATIONSHIP = 0.05

# ── Source priority tiers (higher = more actionable) ──────────────────────────
SOURCE_PRIORITY = {
    "reminder":         1.0,
    "calendar":         0.95,
    "gmail":            0.80,
    "notion":           0.75,
    "slack":            0.65,
    "whatsapp":         0.60,
    "sms":              0.40,
    "chrome_extension": 0.10,
}

# ── Noise patterns (definite low-signal events — filtered before scoring) ─────
NOISE_PATTERNS = [
    "coffee machine", "dramatic steam", "which room", "air conditioning",
    "lunch is late", "ride receipt", "otp is", "do not share",
    "resolved", "moving this casual", "receipt from", "receipt attached",
    "newsletter:", "promo:", "webinar replay", "workspace digest",
    "savings link", "saved link",
    "sandwich", "snack drawer", "projector remote",
    "cafeteria sign", "cardamom back", "chai place",
    "elevator skipped", "blue notebook",
    "meme", "piano",
]

# ── Urgency keywords (boost score when present) ───────────────────────────────
URGENCY_SIGNALS = [
    "due", "deadline", "overdue", "by eod", "by noon", "by tonight",
    "before", "urgent", "asap", "risk", "missing", "blocked",
    "negotiate", "renewal", "confirm", "approve", "review",
    "submit", "send", "final", "closes", "expires",
    "late fee", "slot", "release",
]
