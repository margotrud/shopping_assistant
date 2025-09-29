"""
color
=====

Does: Aggregate core color-domain definitions (constants & vocabularies) shared
      across extraction, recovery, and normalization logic.
Used By: Token/base recovery, suffix handling, alias validation, phrase extraction,
         RGB pipelines.
Returns: Pure data structures and accessor functions; no side effects beyond
         lightweight set construction.
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
