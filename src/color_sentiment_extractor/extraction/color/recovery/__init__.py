# extraction/color/recovery/__init__.py
"""
recovery
========

Does:
    Provide recovery strategies for noisy or ambiguous tokens in color phrases.
    Includes suffix/root equivalence, LLM-based simplification, and modifier resolution.
Submodules:
    - fuzzy_recovery      : suffix/root equivalence & fuzzy matches
    - llm_recovery        : simplification using LLM assistance
    - modifier_resolution : resolution & normalization of modifier tokens
"""

from __future__ import annotations

# ── Submodule imports ─────────────────────────────────────────────────────────
from .fuzzy_recovery import is_suffix_root_match
from .llm_recovery import (
    simplify_phrase_if_needed,
    simplify_color_description_with_llm,
)
from .modifier_resolution import (
    resolve_modifier_token,
    match_direct_modifier,
    match_suffix_fallback,
    is_blocked_modifier_tone_pair,
    is_known_tone,
    recover_y_with_fallback,
    is_modifier_compound_conflict
)

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "is_suffix_root_match",
    "simplify_phrase_if_needed",
    "simplify_color_description_with_llm",
    "resolve_modifier_token",
    "match_direct_modifier",
    "match_suffix_fallback",
    "is_blocked_modifier_tone_pair",
    "is_known_tone",
    "recover_y_with_fallback",
    "is_modifier_compound_conflict"
]
