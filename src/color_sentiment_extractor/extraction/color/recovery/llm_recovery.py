# extraction/color/recovery/llm_recovery.py
from __future__ import annotations

import logging
import re
from typing import Optional, Set

from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.recovery.modifier_resolution import (
    resolve_modifier_token,
)
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token

logger = logging.getLogger(__name__)

# 🔒 Teintes interdites en autonome (on ne veut pas les promouvoir en "tone" seules)
AUTONOMOUS_TONE_BAN: Set[str] = {"dust", "glow"}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _preserve_surface_mod_when_valid_pair(
    text: str, known_modifiers: Set[str], known_tones: Set[str], debug: bool = False
) -> str:
    """
    Does:
        If text is 'left right' with right ∈ known_tones, and left is a surface with
        '-y'/'-ish' whose base ∈ known_modifiers, keep the surface (e.g., 'dusty rose').
    """
    if not text:
        return text

    normalized = text.strip().lower()
    m = re.match(r"^\s*([a-z\-]+)\s+([a-z][a-z\-\s]*)\s*$", normalized)
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
            if debug:
                logger.debug("[PRESERVE SURFACE] '%s %s' (base='%s')", left, right, base)
            return f"{left} {right}"

    return text


# ------------------------------------------------------------
# Core
# ------------------------------------------------------------
def _attempt_simplify_token(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client,
    role: str = "modifier",
    debug: bool = True,
) -> Optional[str]:
    """
    Does:
        Use LLM to simplify a noisy token into a known tone or modifier.
    Returns:
        A valid normalized token or None.
    """
    if debug:
        logger.debug("─────── _attempt_simplify_token ───────")
        logger.debug("[INPUT] token=%r | role=%r", token, role)

    simplified = simplify_phrase_if_needed(
        token, known_modifiers, known_tones, llm_client, debug=debug
    )

    if debug:
        logger.debug("[SIMPLIFIED] LLM result: %r", simplified)

    if simplified:
        words = simplified.strip().split()
        # rôle: on prend le 1er mot pour 'modifier', le dernier pour 'tone'
        idx = 0 if role == "modifier" else -1
        raw_result = words[idx] if words else simplified.strip()
        result = normalize_token(raw_result, keep_hyphens=True)

        if debug:
            logger.debug(
                "[PARSE] raw=%r → normalized=%r | in_mod=%s | in_tone=%s",
                raw_result,
                result,
                result in known_modifiers,
                result in known_tones,
            )

        # 🚫 banlist pour tones autonomes indésirables
        if role == "tone" and result in AUTONOMOUS_TONE_BAN:
            if debug:
                logger.debug("[BANLIST] %r disallowed as standalone tone", result)
            return None

        if (
            (role == "modifier" and result in known_modifiers)
            or (role == "tone" and result in known_tones)
            or result in known_modifiers
            or result in known_tones
        ):
            if debug:
                logger.debug("[ACCEPT] %r", result)
            return result

        # 🔁 Fallback: base recovery (strict)
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
            if debug:
                logger.debug("[BANLIST] recovered %r disallowed as standalone tone", recovered)
            return None

        if recovered:
            if debug:
                logger.debug("[FALLBACK RECOVERY] %r → %r", result, recovered)
            return recovered

    else:
        if debug:
            logger.debug("[REJECT] No simplification result for %r", token)

    if debug:
        logger.debug("[FINAL] Returning: None")
    return None


def _extract_filtered_tokens(tokens, known_modifiers, known_tones, llm_client, debug: bool):
    """
    Does:
        Extract modifier or tone tokens from a token stream using resolution logic,
        with fallback to LLM simplification and several safety filters.
    Returns:
        A set of resolved tokens (modifiers/tones).
    """
    result: Set[str] = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if debug:
            logger.debug("[TOKEN] %r → %r (POS=%s) | cosmetic=%s", tok.text, raw, tok.pos_, raw in COSMETIC_NOUNS)

        # Block known cosmetic nouns
        if raw in COSMETIC_NOUNS:
            if debug:
                logger.debug("[SKIP] Cosmetic noun %r", raw)
            continue

        # Skip connectors via POS tag
        if tok.pos_ == "CCONJ":
            if debug:
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
                    if debug:
                        logger.debug("[SIMPLIFIED FALLBACK] %r → %r", raw, resolved)

        if debug:
            logger.debug(
                "[RESOLVED] raw=%r → %r | raw∈tones=%s | res∈tones=%s",
                raw,
                resolved,
                raw in known_tones,
                (resolved in known_tones) if resolved else "—",
            )

        # Safety filters
        if len(raw) <= 3 and resolved != raw and resolved not in known_modifiers and resolved not in known_tones:
            if debug:
                logger.debug("[REJECT] %r too short for safe fuzzy → %r", raw, resolved)
            continue

        if resolved and "-" in resolved and not resolved.startswith(raw):
            if debug:
                logger.debug("[REJECT] compound mismatch: %r → %r", raw, resolved)
            continue

        if resolved and " " in resolved and " " not in raw:
            if debug:
                logger.debug("[REJECT] multi-word from single token: %r → %r", raw, resolved)
            continue

        if len(result) >= 3 and resolved and resolved != raw:
            if debug:
                logger.debug("[REJECT] already 3+ matches, skipping fuzzy %r → %r", raw, resolved)
            continue

        if resolved:
            result.add(resolved)
            if debug:
                logger.debug("[MATCH] %r → %r", raw, resolved)

    return result


# ------------------------------------------------------------
# LLM wrappers
# ------------------------------------------------------------
def build_prompt(phrase: str) -> str:
    """Kept for compatibility; NOT used by OpenRouterClient.simplify()."""
    return f"What is the simplified base color or tone implied by: '{phrase}'?"


def simplify_color_description_with_llm(
    phrase: str,
    llm_client,
    cache=None,
    debug: bool = False,
) -> str:
    """
    Does:
        Ask the LLM client to simplify a color description.
        NOTE: OpenRouterClient.simplify() expects the raw phrase (it builds the prompt itself).
    """
    if debug:
        logger.debug("[LLM SIMPLIFY] phrase=%r", phrase)

    if cache and hasattr(cache, "get_simplified"):
        cached = cache.get_simplified(phrase)
        if cached:
            if debug:
                logger.debug("[CACHE HIT] %r → %r", phrase, cached)
            return cached

    # ✅ pass the PHRASE directly; the client builds its prompt
    simplified = llm_client.simplify(phrase)

    if cache and hasattr(cache, "store_simplified"):
        cache.store_simplified(phrase, simplified)

    if debug:
        logger.debug("[LLM RESPONSE] %r → %r", phrase, simplified)
    return simplified


def simplify_phrase_if_needed(
    phrase: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client,
    cache=None,
    debug: bool = False,
) -> Optional[str]:
    """
    Does:
        Simplify a descriptive phrase only if needed.
        - Preserve '-y/-ish' surface when we already have a valid (modifier, tone) pair.
        - If phrase is already a known tone, return as-is.
    """
    if llm_client is None:
        return None

    if debug:
        logger.debug("[SIMPLIFY] Checking phrase: %r", phrase)

    # 1) preservation avant toute chose (ne pas aplatir 'dusty rose')
    preserved = _preserve_surface_mod_when_valid_pair(
        phrase, known_modifiers, known_tones, debug=debug
    )
    if preserved != phrase:
        return preserved  # valid surface pair → keep as-is

    normalized = phrase.lower().strip()
    if normalized in known_tones:
        if debug:
            logger.debug("[EXACT MATCH] %r is a known tone (no fallback)", phrase)
        return phrase

    simplified = simplify_color_description_with_llm(
        phrase=phrase, llm_client=llm_client, cache=cache, debug=debug
    )

    if simplified and simplified != phrase:
        if debug:
            logger.debug("[LLM SIMPLIFIED] %r → %r", phrase, simplified)
        # re-apply preservation on the LLM output as well
        simplified = _preserve_surface_mod_when_valid_pair(
            simplified, known_modifiers, known_tones, debug=debug
        )
        return simplified

    if debug:
        logger.debug("[UNSIMPLIFIED] No simplification applied, returning raw phrase")
    return phrase
