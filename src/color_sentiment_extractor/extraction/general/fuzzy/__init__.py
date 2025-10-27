# src/color_sentiment_extractor/extraction/general/fuzzy/__init__.py
"""
fuzzy.

Does: Facade exposing stable fuzzy utilities across core scoring, conflicts,
alias validation, and expression matching.

Returns: Public API for fuzzy token scoring, conflict detection, safe match,
and expression alias resolution.
Used by: General extraction pipelines needing robust fuzzy matching.
"""

from __future__ import annotations

# ── Conflict rules ──────────────────────────────────────────────────────────
from .conflict_rules import (
    is_negation_conflict,
)

# ── Core ─────────────────────────────────────────────────────────────────────
from .fuzzy_core import (
    collapse_duplicates,
    fuzzy_match_token_safe,
    fuzzy_token_match,  # expose raw score
    is_exact_match,
    is_single_substitution,
    is_single_transposition,
    is_strong_fuzzy_match,
)

# ── Scoring ──────────────────────────────────────────────────────────────────
from .scoring import (
    fuzzy_token_overlap_count,
    fuzzy_token_score,
    rhyming_conflict,
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
]

__docformat__ = "google"
