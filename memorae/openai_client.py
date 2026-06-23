"""
openai_client.py — OpenAI API wrapper with automatic model fallback.
"""
from __future__ import annotations
import re
import time
import logging
from typing import Optional
from config import OPENAI_API_KEY, OPENAI_MODEL_FALLBACK_CHAIN
logger = logging.getLogger(__name__)

def _extract_retry_delay(err_str: str, default: float = 5.0) -> float:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_str)
    if m:
        return float(m.group(1))
    m2 = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", err_str)
    if m2:
        return float(m2.group(1))
    return default

def _is_daily_quota(err_str: str) -> bool:
    return any(kw in err_str for kw in (
        "per_day", "perday", "permodelperday", "per model per day",
        "generatecontentinputtokenspermodelperday", "generaterequestsperday",
        "insufficient_quota", "exceeded your current quota",
    ))

def _is_rate_limit(err_str: str) -> bool:
    return any(kw in err_str for kw in (
        "quota", "429", "resource_exhausted", "rate_limit_exceeded",
        "too many requests", "rate limit",
    ))

def _is_model_unavailable(err_str: str) -> bool:
    return any(kw in err_str for kw in (
        "not found", "404", "unsupported", "invalid argument",
        "model_not_found", "does not exist",
    ))

def generate(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 1500,
    max_rpm_wait: float = 65.0,
) -> tuple[str, str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
        
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is not set or invalid.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    last_error: Optional[Exception] = None

    for model_name in OPENAI_MODEL_FALLBACK_CHAIN:
        for attempt in range(2):
            try:
                logger.debug(f"[OpenAI] Trying model={model_name} attempt={attempt + 1}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                text = response.choices[0].message.content.strip()
                logger.info(f"[OpenAI] Success with model={model_name}")
                return text, model_name
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                
                if _is_model_unavailable(err_str):
                    logger.warning(f"[OpenAI] Model {model_name} not found, skipping.")
                    break
                    
                if _is_daily_quota(err_str):
                    logger.warning(f"[OpenAI] Quota exhausted for {model_name}, skipping.")
                    break
                    
                if _is_rate_limit(err_str):
                    wait = _extract_retry_delay(err_str, default=10.0)
                    if wait <= max_rpm_wait:
                        logger.info(f"[OpenAI] Rate limit on {model_name}; waiting {wait:.0f}s...")
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(f"[OpenAI] Wait {wait:.0f}s too long, skipping {model_name}.")
                        break
                        
                logger.warning(f"[OpenAI] Error on {model_name} attempt {attempt + 1}: {exc}")
                if attempt == 0:
                    time.sleep(2.0)
                else:
                    break

    raise RuntimeError(f"[OpenAI] All models exhausted. Last error: {last_error}") from last_error
