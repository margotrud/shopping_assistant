# src/color_sentiment_extractor/extraction/general/fuzzy/__init__.py
"""
fuzzy

Does: Facade exposing stable fuzzy utilities across core scoring, conflicts, alias validation, and expression matching.
Returns: Public API for fuzzy token scoring, conflict detection, safe match, and expression alias resolution.
Used by: General extraction pipelines needing robust fuzzy matching.
"""

from __future__ import annotations

# ── Core ─────────────────────────────────────────────────────────────────────
from .fuzzy_core import (
    is_exact_match,
    is_strong_fuzzy_match,
    fuzzy_match_token_safe,
    fuzzy_token_match,           # expose raw score
    collapse_duplicates,
    is_single_transposition,
    is_single_substitution,
)

# ── Scoring ──────────────────────────────────────────────────────────────────
from .scoring import (
    fuzzy_token_score,
    rhyming_conflict,
    fuzzy_token_overlap_count,
)

# ── Conflict rules ──────────────────────────────────────────────────────────
from .conflict_rules import (
    is_negation_conflict,
)

# ── Alias validation ────────────────────────────────────────────────────────
from .alias_validation import (
    _handle_multiword_alias,
    is_valid_singleword_alias,
)

# ── Expression matching ─────────────────────────────────────────────────────
from .expression_match import (
    cached_match_expression_aliases,
    match_expression_aliases,
)

__all__ = [
    # Core
    "is_exact_match",
    "is_strong_fuzzy_match",
    "fuzzy_match_token_safe",
    "fuzzy_token_match",
    "collapse_duplicates",
    "is_single_transposition",
    "is_single_substitution",
    # Scoring
    "fuzzy_token_score",
    "rhyming_conflict",
    "fuzzy_token_overlap_count",
    # Conflict
    "is_negation_conflict",
    # Alias
    "_handle_multiword_alias",
    "is_valid_singleword_alias",
    # Expressions
    "cached_match_expression_aliases",
    "match_expression_aliases",
]

__docformat__ = "google"
