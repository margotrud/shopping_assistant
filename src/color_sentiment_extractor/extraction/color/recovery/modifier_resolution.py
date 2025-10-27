# extraction/color/recovery/modifier_resolution.py
"""
modifier_resolution.
===================

Does:
    Resolve descriptive modifier tokens via normalize → base recovery,
    direct/compound matches, suffix handling, and guarded fuzzy fallback.

Returns:
    Public helpers to validate tones, resolve modifiers, and filter conflicts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Set

from color_sentiment_extractor.extraction.color import (
    BLOCKED_TOKENS,
    RECOVER_BASE_OVERRIDES,
    get_known_tones,  # light API access to tones
)
from color_sentiment_extractor.extraction.general.token import (
    normalize_token,
    recover_base,
    singularize,
)
from color_sentiment_extractor.extraction.general.types import TokenLike

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "is_known_tone",
    "is_valid_tone",
    "match_direct_modifier",
    "match_suffix_fallback",
    "recover_y_with_fallback",
    "resolve_modifier_token",
    "should_suppress_compound",
    "is_blocked_modifier_tone_pair",
    "is_modifier_compound_conflict",
    "resolve_fallback_tokens",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Lazy vocab fetch (avoid strong deps) ──────────────────────────────────────
def _get_known_tones() -> set[str]:
    """Does: Get tones via public color API; empty set on failure."""
    try:
        return set(get_known_tones())
    except Exception:
        return set()


# =============================================================================
# 1) TONE AND COLOR VALIDATION HELPERS
# =============================================================================
def is_known_tone(word: str, known_tones: set[str], all_webcolor_names: set[str]) -> bool:
    """Does: Check if a normalized token is a tone or standard web color. Returns: bool."""
    norm = normalize_token(word, keep_hyphens=True)
    return norm in known_tones or norm in all_webcolor_names


def is_valid_tone(phrase: str, known_tones: set[str], debug: bool = False) -> bool:
    """Does: Validate a phrase as a known tone via normalize + strict base recovery.
    Returns: bool.
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
# 2) MODIFIER TOKEN RESOLUTION HELPERS
# =============================================================================
def match_direct_modifier(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str] | None = None,
    debug: bool = False,
) -> str | None:
    """Does: Resolve to a known modifier via direct match → base recovery → light fallbacks.
    Returns: str|None.
    """
    raw = token
    token = normalize_token(token, keep_hyphens=True)

    # 1) Direct
    if token in known_modifiers:
        return token

    # 2) Unified recovery (strict+fuzzy guarded)
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
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[BASE MATCH] %r → %r (in modifiers)", raw, base)
            return base

        # 2b) glossy→gloss : prefer root if base endswith 'y'
        if base.endswith("y"):
            root = base[:-1]
            if root in known_modifiers:
                if debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[Y→ROOT] %r → %r → %r", raw, base, root)
                return root

        # 2c) rosier→rosy→rose via overrides
        if token.endswith("ier"):
            y_form = token[:-3] + "y"
            override = RECOVER_BASE_OVERRIDES.get(y_form)
            if override and override in known_modifiers:
                if debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[IER→Y→OVERRIDE] %r → %r → %r", token, y_form, override)
                return override

    # 3) Singularize
    singular = singularize(token)
    if singular in known_modifiers:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[SINGULAR MATCH] %r → %r", raw, singular)
        return singular

    # 4) Compound part fallback (space-split)
    if " " in token:
        for part in token.split():
            if part in known_modifiers:
                if debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[COMPOUND PART] %r → %r", raw, part)
                return part

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[NO MATCH] %r", raw)
    return None


def match_suffix_fallback(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str] | None = None,
    debug: bool = False,
) -> str | None:
    """Does: Resolve noisy/suffixed modifiers via recovery; accepts spaced forms.
    Returns: str|None.
    """
    norm = normalize_token(token, keep_hyphens=True)
    raw = norm.lower()
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SUFFIX FALLBACK] %r → %r", token, raw)

    # Allow collapsed space variants like "soft y" → "softy"
    collapsed = raw.replace(" ", "")
    if collapsed in known_modifiers or (known_tones is not None and collapsed in known_tones):
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[COLLAPSED MATCH] %r", collapsed)
        return collapsed

    base = recover_base(
        raw,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or _get_known_tones()),
        fuzzy_fallback=True,
        debug=debug,
    )
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FINAL BASE] %r → %r", token, base)

    if base and (base in known_modifiers or (known_tones and base in known_tones)):
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[VALID BASE] %r", base)
        return base

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[NO VALID MATCH] %r", token)
    return None


def recover_y_with_fallback(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """Does: Resolve '-y' family to a canonical base via unified recovery. Returns: str|None."""
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
# 3) MODIFIER RESOLUTION ORCHESTRATOR
# =============================================================================
def resolve_modifier_token(
    raw_token: str,
    known_modifiers: set[str],
    known_tones: set[str] | None = None,
    *,
    fuzzy: bool = True,
    debug: bool = False,
) -> str | None:
    """Does: Normalize + recover_base to resolve a modifier. Keeps tones as-is if provided.
    Returns: str|None.
    """
    if not raw_token:
        return None

    token = normalize_token(raw_token, keep_hyphens=True)

    # Compat: keep tones when provided
    if known_tones and token in known_tones:
        if debug and logger.isEnabledFor(logging.DEBUG):
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
# 4) MODIFIER CONFLICT & FILTER HELPERS
# =============================================================================
def should_suppress_compound(mod: str, tone: str) -> bool:
    """Does: Detect redundant (modifier, tone) by equality/prefix-overlap. Returns: bool."""
    mod_n = normalize_token(mod, keep_hyphens=True)
    tone_n = normalize_token(tone, keep_hyphens=True)
    return mod_n == tone_n or tone_n.startswith(mod_n) or mod_n.startswith(tone_n)


def is_blocked_modifier_tone_pair(
    modifier: str,
    tone: str,
    blocked_pairs: Set[tuple[str, str]] = BLOCKED_TOKENS,
) -> bool:
    """Does: Check if (modifier, tone) is explicitly blocked (symmetric). Returns: bool."""
    m = normalize_token(modifier, keep_hyphens=True)
    t = normalize_token(tone, keep_hyphens=True)
    return (m, t) in blocked_pairs or (t, m) in blocked_pairs


def is_modifier_compound_conflict(expression: str, modifier_tokens: set[str]) -> bool:
    """Does: Resolve expression and test overlap with known modifiers. Returns: bool."""
    resolved = resolve_modifier_token(
        expression,
        modifier_tokens,
        known_tones=set(),  # explicit empty set (not None)
        fuzzy=True,
        debug=False,
    )
    return bool(resolved and resolved in modifier_tokens)


def resolve_fallback_tokens(
    tokens: Iterable[TokenLike],
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> set[str]:
    """Does: Recover missed modifier/tone tokens from a stream (rule-first, LLM-free).
    Returns: set[str].
    """
    resolved: set[str] = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if raw in known_tones:
            resolved.add(raw)
            continue

        mod = resolve_modifier_token(raw, known_modifiers, known_tones, fuzzy=True, debug=debug)
        if mod:
            resolved.add(mod)
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[FALLBACK TOKEN] %r → %r", raw, mod)

    return resolved
