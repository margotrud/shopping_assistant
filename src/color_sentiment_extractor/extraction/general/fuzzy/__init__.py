# src/color_sentiment_extractor/extraction/general/fuzzy/__init__.py
"""
fuzzy
=====

Does: Thin facade exposing stable fuzzy utilities.
Exports: is_exact_match, is_strong_fuzzy_match, fuzzy_match_token_safe,
         fuzzy_token_score, rhyming_conflict, fuzzy_token_overlap_count
"""

from __future__ import annotations

# Re-exports (petits modules, imports légers)
from .fuzzy_core import (
    is_exact_match,
    is_strong_fuzzy_match,
    fuzzy_match_token_safe,
    fuzzy_token_match,           # utile si tu veux le score brut
    collapse_duplicates,
    is_single_transposition,
    is_single_substitution,
)

from .scoring import (
    fuzzy_token_score,
    rhyming_conflict,
    fuzzy_token_overlap_count,
)

from .conflict_rules import (
   is_negation_conflict
)

from .alias_validation import (
   _handle_multiword_alias, is_valid_singleword_alias
)
__all__ = [
    "is_exact_match",
    "is_strong_fuzzy_match",
    "fuzzy_match_token_safe",
    "fuzzy_token_match",
    "collapse_duplicates",
    "is_single_transposition",
    "is_single_substitution",
    "fuzzy_token_score",
    "rhyming_conflict",
    "fuzzy_token_overlap_count",
    "is_negation_conflict",
    "_handle_multiword_alias",
    "is_valid_singleword_alias"
]
