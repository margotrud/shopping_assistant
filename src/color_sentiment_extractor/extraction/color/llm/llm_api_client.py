"""
llm_api_client.py
=================

Does:
    Handles prompt construction, request formatting, header setup, response parsing,
    and RGB value retrieval from a language model for descriptive color phrases.

Returns:
    RGB tuples extracted from model responses or None if failed. Includes retry,
    caching, and debug logging support.

Used by:
    - Descriptive color resolution pipeline
    - RGB extraction and caching logic
"""

# 1) 🔧 Imports & Constants
# ------------------------
import logging
import os
import time
import random
import re
from typing import Optional, Tuple, Protocol

import requests

from color_sentiment_extractor.extraction.color.utils import _parse_rgb_tuple

logger = logging.getLogger(__name__)

# Prefer env overrides, keep safe defaults
LLM_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
LLM_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "100"))
LLM_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.4"))
LLM_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "10"))  # seconds

# Backoff config
BACKOFF_BASE = 1.0        # base seconds added each attempt
BACKOFF_MIN = 1.2         # min multiplier
BACKOFF_SPREAD = 0.6      # random spread added to multiplier

# Single session for connection reuse
_session = requests.Session()


# 2) 💾 Optional cache protocol (for typing)
# -----------------------------------------
class RGBCache(Protocol):
    def get_rgb(self, key: str) -> Optional[Tuple[int, int, int]]: ...
    def store_rgb(self, key: str, value: Tuple[int, int, int]) -> None: ...


# 3) 🧰 Minimal client for simplify()
# -----------------------------------
class OpenRouterClient:
    """
    Minimal client that exposes .simplify(phrase: str) used by llm_recovery.
    Shares the same API key and endpoint as query_llm_for_rgb.
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.2, max_tokens: int = 20):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        # (Optionally) add Referer/X-Title for OpenRouter analytics
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # "HTTP-Referer": "http://localhost",
            # "X-Title": "shopping_assistant_V6",
        }

    def _post(self, messages: list[dict]) -> Optional[str]:
        try:
            resp = _session.post(
                LLM_API_URL,
                headers=self._headers(),
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                },
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("[LLM simplify] status=%s body=%s", resp.status_code, resp.text)
                return None
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            return content or None
        except Exception as e:
            logger.error("[simplify POST] %s", e)
            return None

    def simplify(self, phrase: str) -> str:
        """
        Returns 0–2 words (modifier+tone or tone only), lowercase.
        Strips non alpha/-/space characters.
        """
        prompt = (
            "You normalize makeup color phrases. "
            "Return a minimal base color or modifier+tone (max 2 words), lowercase. "
            "No punctuation. If uncertain, return an empty string.\n"
            f"Phrase: '{phrase}'\n"
            "Answer with only the simplified phrase."
        )
        out = self._post([{"role": "user", "content": prompt}])
        if not out:
            return ""

        # sanitize: keep only letters, spaces, hyphens; collapse spaces; max 2 words
        out = out.strip().strip("'\"").lower()
        out = re.sub(r"[^a-z\- ]+", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        parts = out.split()
        return " ".join(parts[:2])


def has_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def get_llm_client(debug: bool = False) -> Optional[OpenRouterClient]:
    ok = has_api_key()
    if debug:
        logger.debug("OpenRouter API key present: %s", ok)
    return OpenRouterClient() if ok else None


# 4) 🧠 Prompt Construction
# -------------------------
def build_color_prompt(color_phrase: str) -> str:
    """
    Does:
        Constructs a descriptive prompt asking the LLM for an RGB tuple
        matching the given color phrase, with strict formatting instructions.
    Returns:
        A formatted natural language prompt string.
    """
    return (
        f"What is the RGB color code for the descriptive phrase: '{color_phrase}'?\n"
        "Respond ONLY with an RGB tuple in the form (R, G, B), without any explanation.\n"
        "Examples:\n"
        "- 'warm beige' → (245, 222, 179)\n"
        "- 'deep lavender' → (150, 123, 182)\n"
        "- 'rosy nude' → (231, 180, 188)\n"
        f"Now: '{color_phrase}' →"
    )


# 5) 📦 Payload & Headers
# -----------------------
def build_llm_request_payload(color_phrase: str) -> dict:
    """
    Does:
        Wraps a color prompt into a JSON payload suitable for OpenRouter (OpenAI-compatible) APIs.
    Returns:
        A dictionary containing model, temperature, and message data.
    """
    prompt = build_color_prompt(color_phrase)
    return {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }


def build_llm_headers(api_key: str) -> dict:
    """
    Does:
        Builds authorization headers for the LLM API call using the provided key.
    Returns:
        A dictionary of HTTP headers.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# 6) 🚀 LLM Query Interface
# -------------------------
def _backoff_sleep(attempt: int) -> None:
    """Sleep with exponential backoff + jitter."""
    sleep_s = (BACKOFF_BASE + attempt) * (BACKOFF_MIN + random.random() * BACKOFF_SPREAD)
    time.sleep(sleep_s)


def query_llm_for_rgb(
    color_phrase: str,
    llm_client=None,
    cache: Optional[RGBCache] = None,
    retries: int = 2,
    debug: bool = False,
) -> Optional[Tuple[int, int, int]]:
    """
    Does:
        Sends a descriptive color phrase to an LLM, parses the RGB result,
        and caches the response if successful. Handles retries and errors.
    Returns:
        A tuple of (R, G, B) if successful, or None if parsing or network fails.
    """
    # normalize input
    color_phrase = (color_phrase or "").strip()
    if not color_phrase:
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("[NO API KEY] OPENROUTER_API_KEY not found in environment.")
        return None

    if cache:
        cached = cache.get_rgb(color_phrase)
        if cached:
            if debug:
                logger.info("[CACHE HIT] '%s' → %s", color_phrase, cached)
            return cached

    payload = build_llm_request_payload(color_phrase)
    headers = build_llm_headers(api_key)

    for attempt in range(retries + 1):
        try:
            if debug:
                logger.info("[LLM QUERY] Attempt %d: '%s'", attempt + 1, color_phrase)

            response = _session.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)

            # Rate-limit explicit
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "0") or 0)
                if debug:
                    logger.warning("[RATE LIMITED] 429. Retry-After=%s", retry_after)
                time.sleep(retry_after if retry_after > 0 else 1.0)
                _backoff_sleep(attempt)
                continue

            if response.status_code != 200:
                logger.warning("[LLM FAILURE] Status %s: %s", response.status_code, response.text)
                _backoff_sleep(attempt)
                continue

            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")

            if not content:
                logger.warning("[LLM] Empty content in response: %s", data)
                _backoff_sleep(attempt)
                continue

            rgb = _parse_rgb_tuple(content, debug=debug)
            if rgb:
                if cache:
                    cache.store_rgb(color_phrase, rgb)
                return rgb

            # parsing failed → retry
            if debug:
                logger.info("[PARSE FAIL] reply=%r", content)

        except Exception as e:
            logger.error("[EXCEPTION] LLM request failed on attempt %d: %s", attempt + 1, e)

        # backoff before next attempt
        _backoff_sleep(attempt)

    if debug:
        logger.warning("[TOTAL FAILURE] '%s' → No valid RGB response.", color_phrase)
    return None
