# extraction/color/recovery/modifier_resolution.py

"""
modifier_resolution.py
======================

Handles the resolution of descriptive modifier tokens in color phrases.
Covers direct match, suffix stripping, compound fallback, and fuzzy logic.
Designed to support modular, multi-step modifier normalization across domains.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

from color_sentiment_extractor.extraction.color import (
    BLOCKED_TOKENS,
    RECOVER_BASE_OVERRIDES,
)
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import (
    singularize,
    normalize_token,
)
from color_sentiment_extractor.extraction.general.utils.load_config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers to lazily get vocabs (avoid strong dependency on color.vocab)
# ---------------------------------------------------------------------
# APRÈS
from color_sentiment_extractor.extraction.color import get_known_tones

def _get_known_tones() -> Set[str]:
    # léger et stable : passe par l'API du package color
    try:
        return set(get_known_tones())
    except Exception:
        return set()


# =============================================================================
# 1. TONE AND COLOR VALIDATION HELPERS
# =============================================================================
def is_known_tone(word: str, known_tones: Set[str], all_webcolor_names: Set[str]) -> bool:
    """
    Does: Checks whether a normalized token is a recognized tone or a standard web color.
    Returns: True if the token is in either the known tone set or web color set.
    """
    norm = normalize_token(word, keep_hyphens=True)
    return norm in known_tones or norm in all_webcolor_names


def is_valid_tone(phrase: str, known_tones: Set[str], debug: bool = False) -> bool:
    """
    Does: Validates whether a phrase resolves to a known tone using normalization
          and unified base recovery (strict by default).
    """
    norm = normalize_token(phrase, keep_hyphens=True)
    if norm in known_tones:
        return True

    base = recover_base(
        norm,
        known_modifiers=set(),
        known_tones=known_tones,
        fuzzy_fallback=False,
        debug=debug,
    )
    return bool(base and base in known_tones)


# =============================================================================
# 2. MODIFIER TOKEN RESOLUTION HELPERS
# =============================================================================
def match_direct_modifier(
    token: str,
    known_modifiers: Set[str],
    known_tones: Optional[Set[str]] = None,
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Resolve a token to a known modifier using direct match, base recovery and simple fallbacks.
    Returns: A matching modifier or None if no match is found.
    """
    raw = token
    token = normalize_token(token, keep_hyphens=True)

    # Step 1: Direct match
    if token in known_modifiers:
        return token

    # Step 2: Use unified recovery
    base = recover_base(
        token,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or _get_known_tones()),
        fuzzy_fallback=True,
        debug=debug,
    )
    if base:
        # 2a) exact base is a known modifier
        if base in known_modifiers:
            if debug:
                logger.debug("[BASE MATCH] %r → %r (in modifiers)", raw, base)
            return base

        # 2b) If base ends with 'y' and its root is a known modifier, prefer the root (glossy → gloss)
        if base.endswith("y"):
            root = base[:-1]
            if root in known_modifiers:
                if debug:
                    logger.debug("[Y→ROOT] %r → %r → %r", raw, base, root)
                return root

        # 2c) Chained overrides like 'rosier' → 'rosy' → 'rose'
        if token.endswith("ier"):
            y_form = token[:-3] + "y"
            override = RECOVER_BASE_OVERRIDES.get(y_form)
            if override and override in known_modifiers:
                if debug:
                    logger.debug("[IER→Y→OVERRIDE] %r → %r → %r", token, y_form, override)
                return override

    # Step 3: Singularize
    singular = singularize(token)
    if singular in known_modifiers:
        if debug:
            logger.debug("[SINGULAR MATCH] %r → %r", raw, singular)
        return singular

    # Step 4: Compound fallback
    if " " in token:
        for part in token.split():
            if part in known_modifiers:
                if debug:
                    logger.debug("[COMPOUND PART] %r → %r", raw, part)
                return part

    if debug:
        logger.debug("[NO MATCH] %r", raw)
    return None


def match_suffix_fallback(
    token: str,
    known_modifiers: Set[str],
    known_tones: Optional[Set[str]],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Attempts to resolve a noisy or suffixed modifier token using recover_base(),
          including variants like 'smoky' → 'smoke'. Accepts space-separated forms too.
    Returns: A valid root (modifier or tone) if found, else None.
    """
    norm = normalize_token(token, keep_hyphens=True)
    raw = norm.lower()
    if debug:
        logger.debug("[SUFFIX FALLBACK] %r → %r", token, raw)

    # Handle spaced variant like "soft y"
    collapsed = raw.replace(" ", "")
    if collapsed in known_modifiers or (known_tones is not None and collapsed in known_tones):
        if debug:
            logger.debug("[COLLAPSED MATCH] %r", collapsed)
        return collapsed

    base = recover_base(
        raw,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or _get_known_tones()),
        fuzzy_fallback=True,
        debug=debug,
    )
    if debug:
        logger.debug("[FINAL BASE] %r → %r", token, base)

    if base and (base in known_modifiers or (known_tones and base in known_tones)):
        if debug:
            logger.debug("[VALID BASE] %r", base)
        return base

    if debug:
        logger.debug("[NO VALID MATCH] %r", token)
    return None


def recover_y_with_fallback(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Resolve '-y' (and friends) to a canonical base via the unified recover_base().
    Returns: base if it’s a known modifier/tone, else None.
    """
    norm = normalize_token(token, keep_hyphens=True)
    base = recover_base(
        norm,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        fuzzy_fallback=True,
        debug=debug,
    )
    return base if base and (base in known_modifiers or base in known_tones) else None


# =============================================================================
# 3. MODIFIER RESOLUTION ORCHESTRATOR
# =============================================================================
def resolve_modifier_token(
    raw_token: str,
    known_modifiers: Set[str],
    known_tones: Optional[Set[str]] = None,
    *,
    fuzzy: bool = True,
    debug: bool = False,
) -> Optional[str]:
    """
    Does:
        Resolve a modifier via normalize + unified recover_base.
        If the input is already a known tone (when `known_tones` is provided),
        returns it as-is (compat behavior).
    """
    if not raw_token:
        return None

    token = normalize_token(raw_token, keep_hyphens=True)

    # Compat: if it’s already a known tone, keep it
    if known_tones and token in known_tones:
        if debug:
            logger.debug("[KNOWN TONE SHORTCUT] %r is a tone → keep as-is", raw_token)
        return token

    base = recover_base(
        token,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or _get_known_tones()),
        fuzzy_fallback=fuzzy,
        debug=debug,
    )
    return base if (base in known_modifiers) else None


# =============================================================================
# 4. MODIFIER CONFLICT & FILTER HELPERS
# =============================================================================
def should_suppress_compound(mod: str, tone: str) -> bool:
    """
    Does: Returns True if mod and tone are semantically redundant
          (equality or prefix containment; e.g., 'soft soft-pink').
    """
    mod_n = normalize_token(mod, keep_hyphens=True)
    tone_n = normalize_token(tone, keep_hyphens=True)
    return mod_n == tone_n or tone_n.startswith(mod_n) or mod_n.startswith(tone_n)


def is_blocked_modifier_tone_pair(
    modifier: str,
    tone: str,
    blocked_pairs: Set[tuple[str, str]] = BLOCKED_TOKENS,
) -> bool:
    """
    Does: Checks whether a modifier-tone pair is explicitly blocked using a blocklist.
    """
    m = normalize_token(modifier, keep_hyphens=True)
    t = normalize_token(tone, keep_hyphens=True)
    return (m, t) in blocked_pairs or (t, m) in blocked_pairs


def is_modifier_compound_conflict(expression: str, modifier_tokens: Set[str]) -> bool:
    """
    Does: Determines whether the expression token semantically overlaps with known modifiers
          by resolving the expression and checking against the modifier token set.
    """
    resolved = resolve_modifier_token(
        expression,
        modifier_tokens,
        known_tones=set(),  # explicit empty set (not None)
        fuzzy=True,
        debug=False,
    )
    return bool(resolved and resolved in modifier_tokens)


def resolve_fallback_tokens(
    tokens,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Set[str]:
    """
    Does: Recovers missed tone or modifier tokens after extraction using direct match or modifier resolution.
    Returns: Set of resolved modifier or tone tokens.
    """
    resolved: Set[str] = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if raw in known_tones:
            resolved.add(raw)
            continue

        mod = resolve_modifier_token(raw, known_modifiers, known_tones, fuzzy=True, debug=debug)
        if mod:
            resolved.add(mod)
            if debug:
                logger.debug("[FALLBACK TOKEN] %r → %r", raw, mod)

    return resolved
