"""
llm_client.py — Unified LLM router.

Dynamically imports and routes requests to either:
  - gemini_client.py
  - openai_client.py
Based on the LLM_PROVIDER setting in .env.
"""

from __future__ import annotations
import logging
from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL_FALLBACK_CHAIN,
    GEMINI_API_KEY,
    GEMINI_MODEL_FALLBACK_CHAIN,
)

logger = logging.getLogger(__name__)

def generate(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 1500,
    max_rpm_wait: float = 65.0,
) -> tuple[str, str]:
    """
    Generate an LLM response using the configured provider.
    Returns: (response_text, model_name_used)
    """
    logger.info(f"Generating with provider={LLM_PROVIDER}")

    if LLM_PROVIDER == "openai":
        from openai_client import generate as openai_generate
        return openai_generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_rpm_wait=max_rpm_wait
        )
    else:
        from gemini_client import generate as gemini_generate
        return gemini_generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_rpm_wait=max_rpm_wait
        )

def get_provider_info() -> dict:
    """Return current provider configuration (for display / logging)."""
    if LLM_PROVIDER == "openai":
        return {
            "provider": "OpenAI",
            "primary_model": OPENAI_MODEL_FALLBACK_CHAIN[0] if OPENAI_MODEL_FALLBACK_CHAIN else "gpt-4o",
            "fallback_chain": OPENAI_MODEL_FALLBACK_CHAIN,
            "key_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-")),
        }
    return {
        "provider": "Google Gemini",
        "primary_model": GEMINI_MODEL_FALLBACK_CHAIN[0] if GEMINI_MODEL_FALLBACK_CHAIN else "gemini-2.5-flash",
        "fallback_chain": GEMINI_MODEL_FALLBACK_CHAIN,
        "key_configured": bool(GEMINI_API_KEY),
    }
