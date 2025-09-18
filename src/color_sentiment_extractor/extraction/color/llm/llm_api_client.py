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

# 1. 🔧 Imports and Constants
# ---------------------------
import logging
import os
import time
from typing import Optional, Tuple
import requests

from color_sentiment_extractor.extraction.color.utils.rgb_distance import _parse_rgb_tuple

LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "mistralai/mistral-7b-instruct"
LLM_MAX_TOKENS = 100
LLM_TEMPERATURE = 0.4

logger = logging.getLogger(__name__)

import re
class OpenRouterClient:
    """
    Client minimal pour fournir .simplify(phrase: str) attendu par llm_recovery.
    Utilise la même clé et l'endpoint OpenRouter que query_llm_for_rgb.
    """
    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.2, max_tokens: int = 20):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        # (Optionnel mais conseillé) OpenRouter apprécie Referer/X-Title
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # headers["HTTP-Referer"] = "http://localhost"
        # headers["X-Title"] = "shopping_assistant_V6"
        return headers

    def _post(self, messages: list[dict]) -> Optional[str]:
        try:
            resp = requests.post(
                LLM_API_URL,
                headers=self._headers(),
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"[⚠️ LLM simplify] status={resp.status_code} body={resp.text}")
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[💥 simplify POST] {e}")
            return None

    def simplify(self, phrase: str) -> str:
        """
        Retourne 0–2 mots (modificateur+ton ou ton seul), en minuscules.
        Supprime tout caractère non alpha/-/espace.
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
        out = re.sub(r"[^a-z\- ]+", " ", out)  # ⬅️ enlève ### & co
        out = re.sub(r"\s+", " ", out).strip()
        parts = out.split()
        return " ".join(parts[:2])

def has_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))

def get_llm_client(debug: bool = False):
    ok = has_api_key()
    if debug:
        print(f"API Key found: {ok}")
    return OpenRouterClient() if ok else None


# 2. 🧠 Prompt Construction
# --------------------------
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


# 3. 📦 Payload & Headers
# ------------------------
def build_llm_request_payload(color_phrase: str) -> dict:
    """
    Does:
        Wraps a color prompt into a JSON payload suitable for OpenAI-style
        LLM API calls.

    Returns:
        A dictionary containing model, temperature, and message data.
    """
    prompt = build_color_prompt(color_phrase)
    return {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "messages": [
            {"role": "user", "content": prompt}
        ]
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
        "Content-Type": "application/json"
    }


# 4. 🎨 Response Parsing
# -----------------------


# 5. 🚀 LLM Query Interface
# --------------------------
def query_llm_for_rgb(
        color_phrase: str,
        llm_client=None,
        cache=None,
        retries: int = 2,
        debug: bool = False
) -> Optional[Tuple[int, int, int]]:
    """
    Does:
        Sends a descriptive color phrase to an LLM, parses the RGB result,
        and caches the response if successful. Handles retries and errors.

    Returns:
        A tuple of (R, G, B) if successful, or None if parsing or network fails.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("[⛔ NO API KEY] OPENROUTER_API_KEY not found in environment.")
        return None

    if cache:
        cached = cache.get_rgb(color_phrase)
        if cached:
            if debug:
                logger.info(f"[🗃️ CACHE HIT] '{color_phrase}' → {cached}")
            return cached

    payload = build_llm_request_payload(color_phrase)
    headers = build_llm_headers(api_key)

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=10
            )

            if debug:
                logger.info(f"[📡 LLM QUERY] Attempt {attempt + 1}: '{color_phrase}'")

            if response.status_code != 200:
                logger.warning(f"[⚠️ LLM FAILURE] Status {response.status_code}: {response.text}")
                time.sleep(1.5 * (attempt + 1))
                continue

            reply = response.json()["choices"][0]["message"]["content"]
            rgb = _parse_rgb_tuple(reply, debug=debug)

            if rgb and cache:
                cache.store_rgb(color_phrase, rgb)

            return rgb

        except Exception as e:
            logger.error(f"[💥 EXCEPTION] LLM request failed on attempt {attempt + 1}: {e}")
            time.sleep(1.5 * (attempt + 1))

    if debug:
        logger.warning(f"[🚫 TOTAL FAILURE] '{color_phrase}' → No valid RGB response.")
    return None
