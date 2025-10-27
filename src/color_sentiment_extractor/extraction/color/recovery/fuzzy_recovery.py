# color/recovery/fuzzy_recovery.py
"""
fuzzy_recovery
==============

Does:
    Verify whether two tokens are suffix/root variants that reduce to the same
    known base, while enforcing semantic and rhyming conflict rules.
Returns:
    is_suffix_root_match(alias, token, ...) -> bool
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import AbstractSet, FrozenSet, Optional

# ── Imports ──────────────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.general.fuzzy import rhyming_conflict
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token

from color_sentiment_extractor.extraction.general.utils import load_config

# ── Public API ───────────────────────────────────────────────────────────────
__all__ = ["is_suffix_root_match"]

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Internal helpers ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_known_modifiers() -> FrozenSet[str]:
    """Does: Load and cache the known modifiers vocabulary."""
    return frozenset(load_config("known_modifiers", mode="set"))


@lru_cache(maxsize=1)
def _get_known_tones() -> FrozenSet[str]:
    """Does: Load and cache the known tones vocabulary."""
    return frozenset(load_config("known_tones", mode="set"))


def _common_base(a_norm: str, t_norm: str, *, km: AbstractSet[str], kt: AbstractSet[str]) -> Optional[str]:
    """Does: Return shared base if both reduce to the same known base, else None."""
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    base_a = recover_base(a_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False)
    base_t = recover_base(t_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False)
    if base_a and base_t and (base_a == base_t) and (base_a in km or base_a in kt):
        return base_a
    return None


# ── Core API ─────────────────────────────────────────────────────────────────
def is_suffix_root_match(
    alias: str,
    token: str,
    *,
    known_modifiers: Optional[AbstractSet[str]] = None,
    known_tones: Optional[AbstractSet[str]] = None,
    debug: bool = False,
) -> bool:
    """
    Does:
        Check if `alias` and `token` are suffix/root variants resolving to the
        same known base (e.g., 'rosy'↔'rose', 'beigey'↔'beige'), without
        semantic or rhyming conflicts.
    Returns:
        True iff they share a known base and at least one was transformed from its base.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
    if not alias or not token:
        return False

    km: AbstractSet[str] = known_modifiers or _get_known_modifiers()
    kt: AbstractSet[str] = known_tones or _get_known_tones()

    a_norm = normalize_token(alias, keep_hyphens=True)
    t_norm = normalize_token(token, keep_hyphens=True)
    if not a_norm or not t_norm:
        return False

    # Require an actual transformation on at least one side
    base_a_noff = recover_base(a_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False)
    base_t_noff = recover_base(t_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False)
    if a_norm == base_a_noff and t_norm == base_t_noff:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[NO-OP] both unchanged → reject | a=%r t=%r", a_norm, t_norm)
        return False

    base = _common_base(a_norm, t_norm, km=km, kt=kt)
    if base is None:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FAIL] no common base | a=%r(%r) t=%r(%r)", a_norm, base_a_noff, t_norm, base_t_noff)
        return False

    pair = frozenset({a_norm, t_norm})
    if pair in SEMANTIC_CONFLICTS:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[CONFLICT] semantic conflict for pair=%r", pair)
        return False

    if rhyming_conflict(a_norm, t_norm):
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[CONFLICT] rhyming conflict for a=%r t=%r", a_norm, t_norm)
        return False

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[OK] common base=%r; a=%r t=%r", base, a_norm, t_norm)
    return True
