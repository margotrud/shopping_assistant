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
from .llm_recovery import simplify_phrase_if_needed, simplify_color_description_with_llm
from .modifier_resolution import (
    resolve_modifier_token,
    match_direct_modifier,
    match_suffix_fallback,
)

__all__ = [
    "is_suffix_root_match",
    "simplify_phrase_if_needed",
    "simplify_color_description_with_llm",
    "resolve_modifier_token",
    "match_direct_modifier",
    "match_suffix_fallback",
]
