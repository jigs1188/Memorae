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
from google import genai
from google.genai import types
from core.config import GEMINI_API_KEY
logger = logging.getLogger(__name__)

# Initialize the new SDK client
client = genai.Client(api_key=GEMINI_API_KEY)

_LIVE_MODELS = None
def _get_dynamic_gemini_models() -> list[str]:
    global _LIVE_MODELS
    if _LIVE_MODELS is not None:
        return _LIVE_MODELS
        
    models = []
    try:
        for m in client.models.list():
            # In google-genai, the model object doesn't have supported_generation_methods
            # Just collect the names and filter via the rank function.
            name = m.name.replace("models/", "")
            models.append(name)
    except Exception as exc:
        logger.warning(f"Could not fetch live models: {exc}")
        _LIVE_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
        return _LIVE_MODELS

    def rank(name: str) -> int:
        # Heavily penalize non-text models
        if any(bad in name for bad in ("tts", "image", "vision",
                                        "robotics", "computer-use", "deep-research", "embedding", "aqa", "imagen", "veo")):
            return -1000
            
        score = 0
        # Version tier: higher is better
        if "3.5" in name:   score += 600
        elif "3.1" in name: score += 500
        elif "3.0" in name or (name.count(".") == 0 and "3" in name): score += 400
        elif "2.5" in name: score += 200
        elif "2.0" in name: score += 100
        elif "1.5" in name: score += 50
        
        # Model size tier
        if "pro" in name:   score += 30
        elif "flash" in name: score += 15
        
        # Lite variants are cheaper quota-wise — rank them BELOW flash but keep them in list
        if "lite" in name:  score -= 10
        
        # Previews are okay, just slightly penalize them so stable versions are preferred
        if "preview" in name: score -= 5
        if "experimental" in name: score -= 5
        
        return score

    models.sort(key=rank, reverse=True)
    models = [m for m in models if rank(m) >= 0]

    if not models:
        models = ["gemini-3.5-flash"]
    _LIVE_MODELS = models
    logger.info(f"Live model list (priority order): {models[:6]}")
    return _LIVE_MODELS

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
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    last_error: Optional[Exception] = None
    for model_name in _get_dynamic_gemini_models():
        for attempt in range(retries_per_model + 1):
            try:
                logger.debug(f"Trying model={model_name} attempt={attempt + 1}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generation_config,
                )
                text = response.text.strip() if response.text else ""
                logger.info(f"Success with model={model_name}")
                return text, model_name
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                # 404 / model-not-found: skip immediately
                if any(kw in err_str for kw in ("not found", "404", "unsupported", "invalid argument")):
                    logger.warning(f"Model {model_name} not available (404), skipping.")
                    break  # next model
                # Daily quota: skip immediately AND invalidate the cache so
                # the next request rebuilds the live list — different model families
                # have separate daily quota buckets.
                is_daily = any(kw in err_str for kw in (
                    "per_day", "perday", "permodelperday", "per model per day",
                    "generatecontentinputtokenspermodelperday",
                    "generaterequestsperday",
                ))
                if is_daily:
                    logger.warning(f"Daily quota exhausted on {model_name}, skipping and refreshing model list.")
                    global _LIVE_MODELS
                    _LIVE_MODELS = None  # force re-fetch next call
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
