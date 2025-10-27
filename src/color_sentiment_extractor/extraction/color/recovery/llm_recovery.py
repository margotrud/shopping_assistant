# color/recovery/llm_recovery.py
"""
llm_recovery.
============

Does:
    LLM-aided simplification of color phrases/tokens with safety filters
    (cosmetic nouns, connectors, autonomous-tone ban) and base recovery.

Returns:
    simplify_phrase_if_needed(...)->str|None, simplify_color_description_with_llm(...)->str,
    _attempt_simplify_token(...)->str|None, _extract_filtered_tokens(...)->set[str]
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Protocol

from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.general.token import normalize_token, recover_base
from color_sentiment_extractor.extraction.general.types import TokenLike

from .modifier_resolution import resolve_modifier_token

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "simplify_phrase_if_needed",
    "simplify_color_description_with_llm",
    "_attempt_simplify_token",
    "_extract_filtered_tokens",
]

from ..constants import AUTONOMOUS_TONE_BAN

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────
class LLMClient(Protocol):
    def simplify(self, phrase: str) -> str: ...


# ── Constants / Regex ─────────────────────────────────────────────────────────
_PAIR_RE = re.compile(r"^\s*([a-z\-]+)\s+([a-z][a-z\-\s]*)\s*$")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _preserve_surface_mod_when_valid_pair(
    text: str, known_modifiers: set[str], known_tones: set[str], debug: bool = False
) -> str:
    """
    Does: If text is 'left right' where right∈known_tones and left ends with -y/-ish
          whose base∈known_modifiers, keep the surface form (e.g., 'dusty rose').
    """
    if not text:
        return text

    m = _PAIR_RE.match(text.strip().lower())
    if not m:
        return text

    left, right = m.group(1), m.group(2)
    if right not in known_tones:
        return text

    if left.endswith(("y", "ish")):
        base = recover_base(
            left,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            fuzzy_fallback=False,
            debug=False,
        )
        if base and base in known_modifiers:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[PRESERVE SURFACE] '%s %s' (base='%s')", left, right, base)
            return f"{left} {right}"

    return text


# ── Core ──────────────────────────────────────────────────────────────────────
def _attempt_simplify_token(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    llm_client: LLMClient | None,
    role: str = "modifier",
    debug: bool = True,
) -> str | None:
    """
    Does: Use LLM to simplify a noisy token into a known modifier/tone with guarded fallbacks.
    Returns: Normalized token or None.
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("─────── _attempt_simplify_token ───────")
        logger.debug("[INPUT] token=%r | role=%r", token, role)

    simplified = simplify_phrase_if_needed(
        token, known_modifiers, known_tones, llm_client, debug=debug
    )

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SIMPLIFIED] LLM result: %r", simplified)

    if not simplified:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[REJECT] No simplification result for %r", token)
        return None

    words = simplified.strip().split()
    idx = 0 if role == "modifier" else -1  # first for modifier, last for tone
    raw_result = words[idx] if words else simplified.strip()
    result = normalize_token(raw_result, keep_hyphens=True)

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[PARSE] raw=%r → normalized=%r | in_mod=%s | in_tone=%s",
            raw_result,
            result,
            result in known_modifiers,
            result in known_tones,
        )

    # Banlist for standalone tones
    if role == "tone" and result in AUTONOMOUS_TONE_BAN:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[BANLIST] %r disallowed as standalone tone", result)
        return None

    if (
        (role == "modifier" and result in known_modifiers)
        or (role == "tone" and result in known_tones)
        or result in known_modifiers
        or result in known_tones
    ):
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ACCEPT] %r", result)
        return result

    # Fallback: base recovery (strict+fuzzy)
    recovered = recover_base(
        result,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        fuzzy_fallback=True,
        fuzzy_threshold=78,
        use_cache=False,
        debug=debug,
        depth=0,
    )

    if role == "tone" and recovered in AUTONOMOUS_TONE_BAN:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[BANLIST] recovered %r disallowed as standalone tone", recovered)
        return None

    if recovered:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FALLBACK RECOVERY] %r → %r", result, recovered)
        return recovered

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FINAL] Returning: None")
    return None


def _extract_filtered_tokens(
    tokens: Iterable[TokenLike],
    known_modifiers: set[str],
    known_tones: set[str],
    llm_client: LLMClient | None,
    debug: bool,
) -> set[str]:
    """
    Does: Extract modifier/tone tokens from a token stream with rules, LLM fallback,
     and safety filters.
    Returns: A set of resolved tokens.
    """
    result: set[str] = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TOKEN] %r → %r (POS=%s) | cosmetic=%s",
                tok.text,
                raw,
                tok.pos_,
                raw in COSMETIC_NOUNS,
            )

        # Block cosmetic nouns
        if raw in COSMETIC_NOUNS:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[SKIP] Cosmetic noun %r", raw)
            continue

        # Skip connectors via POS tag
        if tok.pos_ == "CCONJ":
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[SKIP] Connector %r (POS=CCONJ)", raw)
            continue

        # Rule-based resolver first
        resolved = resolve_modifier_token(raw, known_modifiers, known_tones)

        # Fallback to LLM simplifier
        if not resolved and llm_client is not None:
            simplified = simplify_phrase_if_needed(
                raw, known_modifiers, known_tones, llm_client, debug=debug
            )
            if simplified:
                candidate = simplified.strip().split()[0]
                if candidate in known_modifiers or candidate in known_tones:
                    resolved = candidate
                    if debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("[SIMPLIFIED FALLBACK] %r → %r", raw, resolved)

        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[RESOLVED] raw=%r → %r | raw∈tones=%s | res∈tones=%s",
                raw,
                resolved,
                raw in known_tones,
                (resolved in known_tones) if resolved else "—",
            )

        # Safety filters
        if (
            len(raw) <= 3
            and resolved != raw
            and resolved not in known_modifiers
            and resolved not in known_tones
        ):
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[REJECT] %r too short for safe fuzzy → %r", raw, resolved)
            continue

        if resolved and "-" in resolved and not resolved.startswith(raw):
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[REJECT] compound mismatch: %r → %r", raw, resolved)
            continue

        if resolved and " " in resolved and " " not in raw:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[REJECT] multi-word from single token: %r → %r", raw, resolved)
            continue

        if len(result) >= 3 and resolved and resolved != raw:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[REJECT] already 3+ matches, skipping fuzzy %r → %r", raw, resolved)
            continue

        if resolved:
            result.add(resolved)
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[MATCH] %r → %r", raw, resolved)

    return result


# ── LLM wrappers ─────────────────────────────────────────────────────────────
def build_prompt(phrase: str) -> str:
    """Deprecated: kept for compatibility; OpenRouterClient.simplify() builds its own prompt."""
    return f"What is the simplified base color or tone implied by: '{phrase}'?"


def simplify_color_description_with_llm(
    phrase: str,
    llm_client: LLMClient,
    cache=None,
    debug: bool = False,
) -> str:
    """
    Does: Ask the LLM client to simplify a color description (client builds the prompt).
    Returns: Simplified string (may equal input).
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[LLM SIMPLIFY] phrase=%r", phrase)

    if cache and hasattr(cache, "get_simplified"):
        cached = cache.get_simplified(phrase)
        if cached:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[CACHE HIT] %r → %r", phrase, cached)
            return cached

    simplified = llm_client.simplify(phrase)

    if cache and hasattr(cache, "store_simplified"):
        cache.store_simplified(phrase, simplified)

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[LLM RESPONSE] %r → %r", phrase, simplified)
    return simplified


def simplify_phrase_if_needed(
    phrase: str,
    known_modifiers: set[str],
    known_tones: set[str],
    llm_client: LLMClient | None,
    cache=None,
    debug: bool = False,
) -> str | None:
    """
    Does:
        Simplify a descriptive phrase only if needed.
        - Preserve '-y/-ish' surface when there's already a valid (modifier, tone) pair.
        - If phrase is already a known tone, return as-is.

    Returns:
        Simplified or preserved phrase; None if no LLM client.
    """
    if llm_client is None:
        return None

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SIMPLIFY] Checking phrase: %r", phrase)

    # 1) Preserve valid surface pairs (avoid flattening 'dusty rose', etc.)
    preserved = _preserve_surface_mod_when_valid_pair(
        phrase, known_modifiers, known_tones, debug=debug
    )
    if preserved != phrase:
        return preserved

    normalized = phrase.lower().strip()
    if normalized in known_tones:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[EXACT MATCH] %r is a known tone (no fallback)", phrase)
        return phrase

    simplified = simplify_color_description_with_llm(
        phrase=phrase, llm_client=llm_client, cache=cache, debug=debug
    )

    if simplified and simplified != phrase:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[LLM SIMPLIFIED] %r → %r", phrase, simplified)
        # re-apply preservation on the LLM output as well
        simplified = _preserve_surface_mod_when_valid_pair(
            simplified, known_modifiers, known_tones, debug=debug
        )
        return simplified

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[UNSIMPLIFIED] No simplification applied, returning raw phrase")
    return phrase
