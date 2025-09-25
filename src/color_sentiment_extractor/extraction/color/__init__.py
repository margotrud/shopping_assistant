"""
color
=====

Core color-domain definitions: constants, vocabularies, and base rules
shared across extraction and normalization logic.
"""

# ── Constants ────────────────────────────────────────────────────────────────
from .constants import (
    SEMANTIC_CONFLICTS,
    BLOCKED_TOKENS,
    EXPRESSION_SUPPRESSION_RULES,
    Y_SUFFIX_ALLOWLIST,
    ED_SUFFIX_ALLOWLIST,
    NON_SUFFIXABLE_MODIFIERS,
    RECOVER_BASE_OVERRIDES,
    Y_SUFFIX_OVERRIDE_FORMS,
    COSMETIC_NOUNS,
)

# ── Vocabulary ───────────────────────────────────────────────────────────────
from .vocab import (
    WEB_ONLY_COLOR_NAMES,
    COSMETIC_FALLBACK_TONES,
    all_webcolor_names,        # CSS-only alias (backward compat)
    known_tones,               # full tone set (CSS+XKCD+fallbacks)
    get_xkcd_names,
    get_web_named_color_names,
    get_known_tones,
)

__all__ = [
    # constants
    "SEMANTIC_CONFLICTS",
    "BLOCKED_TOKENS",
    "EXPRESSION_SUPPRESSION_RULES",
    "Y_SUFFIX_ALLOWLIST",
    "ED_SUFFIX_ALLOWLIST",
    "NON_SUFFIXABLE_MODIFIERS",
    "RECOVER_BASE_OVERRIDES",
    "Y_SUFFIX_OVERRIDE_FORMS",
    "COSMETIC_NOUNS",
    # vocab
    "WEB_ONLY_COLOR_NAMES",
    "COSMETIC_FALLBACK_TONES",
    "all_webcolor_names",
    "known_tones",
    "get_xkcd_names",
    "get_web_named_color_names",
    "get_known_tones",
]
