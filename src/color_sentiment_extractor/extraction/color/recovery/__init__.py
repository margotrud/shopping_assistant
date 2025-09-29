"""
recovery package
================

Handles recovery strategies for noisy or ambiguous tokens.

Submodules:
-----------
- fuzzy_recovery     : suffix/root equivalence & fuzzy matches
- llm_recovery       : simplification using LLM assistance
- modifier_resolution: resolution & normalization of modifier tokens
"""

from .fuzzy_recovery import is_suffix_root_match
from .llm_recovery import (simplify_phrase_if_needed, simplify_color_description_with_llm,
    _attempt_simplify_token, _extract_filtered_tokens)
from .modifier_resolution import (
    resolve_modifier_token,
    match_direct_modifier,
    match_suffix_fallback,
    is_blocked_modifier_tone_pair,
    is_known_tone

)

__all__ = [
    "is_suffix_root_match",
    "simplify_phrase_if_needed",
    "simplify_color_description_with_llm",
    "resolve_modifier_token",
    "match_direct_modifier",
    "match_suffix_fallback",
    "_attempt_simplify_token",
    "extract_filtered_tokens",
]
