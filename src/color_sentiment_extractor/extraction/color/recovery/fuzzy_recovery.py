# color/recovery/fuzzy_recovery.py

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Set

# ✅ simplified import via color/__init__.py
from color_sentiment_extractor.extraction.color import SEMANTIC_CONFLICTS

from color_sentiment_extractor.extraction.general.fuzzy.scoring import rhyming_conflict
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token
from color_sentiment_extractor.extraction.general.utils.load_config import load_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_known_modifiers() -> Set[str]:
    return frozenset(load_config("known_modifiers", mode="set"))


@lru_cache(maxsize=1)
def _get_known_tones() -> Set[str]:
    # avoid strong import on color.vocab; single source of truth = config
    return frozenset(load_config("known_tones", mode="set"))


def is_suffix_root_match(
    alias: str,
    token: str,
    *,
    known_modifiers: Optional[Set[str]] = None,
    known_tones: Optional[Set[str]] = None,
    debug: bool = False,
) -> bool:
    """
    Does:
        Check if `alias` and `token` are suffix/root variants that recover to the same
        known base (e.g., 'rosy' ↔ 'rose', 'beigey' ↔ 'beige'), without semantic/rhyming conflicts.
    Returns:
        True iff both recover to the same base in vocab AND at least one was transformed.
    """
    if not alias or not token:
        return False

    km = known_modifiers or _get_known_modifiers()
    kt = known_tones or _get_known_tones()

    a_norm = normalize_token(alias, keep_hyphens=True)
    t_norm = normalize_token(token, keep_hyphens=True)

    # strict bases (no fuzzy to avoid overreach)
    base_alias = recover_base(
        a_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False
    )
    base_token = recover_base(
        t_norm, known_modifiers=km, known_tones=kt, debug=False, fuzzy_fallback=False
    )

    if debug:
        logger.debug("[SUFFIX ROOT] %r→%r | %r→%r", t_norm, base_token, a_norm, base_alias)

    # Require an actual transformation (at least one changed)
    if (a_norm == base_alias) and (t_norm == base_token):
        if debug:
            logger.debug("[NO-OP] both unchanged → reject")
        return False

    # Both must resolve to the same known base (modifier or tone)
    if base_alias and base_token and (base_alias == base_token):
        base = base_alias
        if (base in km) or (base in kt):
            # no semantic or rhyming conflicts
            pair = frozenset({a_norm, t_norm})
            if (pair not in SEMANTIC_CONFLICTS) and not rhyming_conflict(a_norm, t_norm):
                if debug:
                    logger.debug("[OK] common base=%r; no conflict", base)
                return True

    if debug:
        logger.debug("[FAIL] alias=%r token=%r baseA=%r baseT=%r", a_norm, t_norm, base_alias, base_token)
    return False
