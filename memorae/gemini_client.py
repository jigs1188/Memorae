"""
gemini_client.py — Gemini API wrapper with automatic model fallback.
The fallback chain tries each model in order (most capable → cheapest).
On quota-exhaustion or model-unavailable errors it drops to the next model
transparently, so the caller never needs to handle retries.
"""
from __future__ import annotations
import re
import time
import logging
from typing import Optional
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACK_CHAIN
logger = logging.getLogger(__name__)
# Configure the SDK once
genai.configure(api_key=GEMINI_API_KEY)
def _make_model(model_name: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(model_name)
def _extract_retry_delay(err_str: str, default: float = 5.0) -> float:
    """Try to parse the 'retry_delay { seconds: N }' from API error message."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_str)
    if m:
        return float(m.group(1))
    # Also try 'please retry in Xs'
    m2 = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
    if m2:
        return float(m2.group(1))
    return default
def generate(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    retries_per_model: int = 1,
    max_rpm_wait: float = 65.0,   # max seconds to wait for a per-minute rate limit
) -> tuple[str, str]:
    """
    Generate text with automatic model fallback.
    Strategy:
    - 404 / not-found: skip to next model immediately
    - Daily quota exhausted: skip to next model immediately
    - Per-minute rate limit: wait up to max_rpm_wait, then skip to next model
    - Other errors: retry once, then skip to next model
    Returns
    -------
    (response_text, model_used)
    """
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    last_error: Optional[Exception] = None
    for model_name in GEMINI_MODEL_FALLBACK_CHAIN:
        model = _make_model(model_name)
        for attempt in range(retries_per_model + 1):
            try:
                logger.debug(f"Trying model={model_name} attempt={attempt + 1}")
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                text = response.text.strip()
                logger.info(f"Success with model={model_name}")
                return text, model_name
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                # 404 / model-not-found: skip immediately
                if any(kw in err_str for kw in ("not found", "404", "unsupported", "invalid argument")):
                    logger.warning(f"Model {model_name} not available (404), skipping.")
                    break  # next model
                # Daily quota: skip immediately
                is_daily = any(kw in err_str for kw in (
                    "per_day", "perday", "permodelperday", "per model per day",
                    "generatecontentinputtokenspermodelperday",
                    "generaterequestsperday",
                ))
                if is_daily:
                    logger.warning(f"Daily quota exhausted on {model_name}, skipping.")
                    break
                # Per-minute / per-request rate limit: wait if short, else skip
                is_quota = any(kw in err_str for kw in ("quota", "429", "resource_exhausted"))
                if is_quota:
                    wait = _extract_retry_delay(err_str, default=10.0)
                    if wait <= max_rpm_wait:
                        logger.info(
                            f"RPM limit on {model_name}; waiting {wait:.0f}s then retrying..."
                        )
                        time.sleep(wait)
                        continue  # retry same model after wait
                    else:
                        logger.warning(
                            f"RPM wait {wait:.0f}s > {max_rpm_wait}s budget; skipping {model_name}."
                        )
                        break
                # Permission / other errors: retry once then move on
                logger.warning(f"Error on {model_name} attempt {attempt + 1}: {exc}")
                if attempt < retries_per_model:
                    time.sleep(2.0)
                else:
                    break
    raise RuntimeError(
        f"All Gemini models exhausted. Last error: {last_error}"
    ) from last_error
