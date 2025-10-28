"""
llm_api_client.py.
=================

Does: Build prompts, HTTP payloads and headers, call the OpenRouter chat API,
      parse RGB tuples from replies, and optionally cache results (with retries).
Returns: RGB tuples (R, G, B) or None on failure.
Used by: Descriptive color resolution and RGB extraction/caching pipelines.
"""

from __future__ import annotations

# ── Imports & Typing ─────────────────────────────────────────────────────────
import logging
import os
import random
import re
import time
from typing import Protocol

import requests  # type: ignore[import-untyped]

from color_sentiment_extractor.extraction.color.utils import _parse_rgb_tuple

# ── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Config (env-overridable) ─────────────────────────────────────────────────
LLM_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
LLM_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "100"))
LLM_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.4"))
LLM_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "10"))  # seconds

# Backoff config
BACKOFF_BASE = 1.0  # base seconds added each attempt
BACKOFF_MIN = 1.2  # min multiplier
BACKOFF_SPREAD = 0.6  # random spread added to multiplier

# Single session for connection reuse
_session = requests.Session()

__all__ = [
    "OpenRouterClient",
    "has_api_key",
    "get_llm_client",
    "build_color_prompt",
    "query_llm_for_rgb",
]


# ── Optional cache protocol (typing only) ────────────────────────────────────
class RGBCache(Protocol):
    """Does: Provide typed get/set for RGB cache used by query_llm_for_rgb.
    Args: key: str for lookup; value: (int,int,int) to store.
    Returns: get_rgb → Optional[Tuple[int,int,int]]; store_rgb → None.
    """

    def get_rgb(self, key: str) -> tuple[int, int, int] | None: ...
    def store_rgb(self, key: str, value: tuple[int, int, int]) -> None: ...


# ── Minimal client for phrase simplify() ─────────────────────────────────────
class OpenRouterClient:
    """Does: Minimal OpenRouter client exposing simplify() for phrase cleanup.
    Args: model: str; temperature: float; max_tokens: int.
    Returns: Instance usable for simplify(); raises if API key is missing.
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.2, max_tokens: int = 20):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        """Does: Build authorization/content headers for OpenRouter calls.
        Args: None.
        Returns: Dict of HTTP headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter analytics:
            # "HTTP-Referer": "http://localhost",
            # "X-Title": "shopping_assistant_V6",
        }

    def _post(self, messages: list[dict]) -> str | None:
        """Does: POST a chat request and return the assistant content string.
        Args: messages: list of chat dicts [{role, content}, ...].
        Returns: Content string or None on error/empty.
        """
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
        """Does: Normalize a color phrase to ≤2 words (modifier+tone or tone).
        Args: phrase: raw descriptive color phrase.
        Returns: Simplified lowercase phrase or empty string if uncertain.
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


# ── Public helpers ───────────────────────────────────────────────────────────
def has_api_key() -> bool:
    """Does: Check presence of OPENROUTER_API_KEY in environment.
    Args: None.
    Returns: True if available, else False.
    """
    return bool(os.getenv("OPENROUTER_API_KEY"))


def get_llm_client(debug: bool = False) -> OpenRouterClient | None:
    """Does: Factory returning OpenRouterClient if API key is present.
    Args: debug: if True, logs presence status.
    Returns: OpenRouterClient or None.
    """
    ok = has_api_key()
    if debug:
        logger.debug("OpenRouter API key present: %s", ok)
    return OpenRouterClient() if ok else None


# ── Prompt construction ──────────────────────────────────────────────────────
def build_color_prompt(color_phrase: str) -> str:
    """Does: Build an instruction asking the LLM for a strict RGB tuple.
    Args: color_phrase: descriptive phrase (e.g., 'dusty rose').
    Returns: Formatted prompt ending with an arrow for the answer.
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


# ── Payload & Headers (RGB flow) ─────────────────────────────────────────────
def _build_llm_request_payload(color_phrase: str) -> dict:
    """Does: Wrap the color prompt into the JSON payload for OpenRouter.
    Args: color_phrase: descriptive phrase.
    Returns: Dict payload {model, max_tokens, temperature, messages}.
    """
    prompt = build_color_prompt(color_phrase)
    return {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }


def _build_llm_headers(api_key: str) -> dict:
    """Does: Build auth/content headers for the RGB OpenRouter request.
    Args: api_key: token string.
    Returns: Dict of HTTP headers.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ── RGB Query Interface ──────────────────────────────────────────────────────
def _backoff_sleep(attempt: int) -> None:
    """Does: Sleep with exponential backoff + jitter based on attempt index.
    Args: attempt: 0-based attempt number.
    Returns: None.
    """
    sleep_s = (BACKOFF_BASE + attempt) * (BACKOFF_MIN + random.random() * BACKOFF_SPREAD)
    time.sleep(sleep_s)


def query_llm_for_rgb(
    color_phrase: str,
    llm_client=None,  # kept for API compatibility
    cache: RGBCache | None = None,
    retries: int = 2,
    debug: bool = False,
) -> tuple[int, int, int] | None:
    """Does: Query OpenRouter for an RGB tuple, parse, retry, and cache result.
    Args: color_phrase: phrase; cache: RGBCache; retries: int; debug: bool.
    Returns: (R, G, B) on success or None on failure.
    """
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

    payload = _build_llm_request_payload(color_phrase)
    headers = _build_llm_headers(api_key)

    for attempt in range(retries + 1):
        try:
            if debug:
                logger.info("[LLM QUERY] Attempt %d: '%s'", attempt + 1, color_phrase)

            response = _session.post(
                LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT
            )

            # Rate-limit handling
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

            if debug:
                logger.info("[PARSE FAIL] reply=%r", content)

        except Exception as e:
            logger.error("[EXCEPTION] LLM request failed on attempt %d: %s", attempt + 1, e)

        _backoff_sleep(attempt)

    if debug:
        logger.warning("[TOTAL FAILURE] '%s' → No valid RGB response.", color_phrase)
    return None
