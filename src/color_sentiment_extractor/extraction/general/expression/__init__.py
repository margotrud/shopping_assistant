# src/color_sentiment_extractor/extraction/general/expression/__init__.py

"""
expression package.
==================

Does: Public interface for expression matching and tone mapping.
Returns: Curated helpers for alias matching, expression→tone mapping, and glued-token vocab.
Used by: Color expression detection pipelines and compound/standalone extractors.
"""

from __future__ import annotations

from .expression_helpers import (
    _inject_expression_modifiers as inject_expression_modifiers,  # noqa: F401
)
from .expression_helpers import (
    # Rules
    apply_expression_context_rules,
    apply_expression_suppression_rules,
    # Exact alias scan helper
    extract_exact_alias_tokens,
    get_all_alias_tokens,
    # Trigger vocab
    get_all_trigger_tokens,
    # Glued-token vocab
    get_glued_token_vocabulary,
    get_matching_expression_tags_cached,
    # Matching & mapping
    map_expressions_to_tones,
)

__docformat__ = "google"


__all__ = [
    # Matching & mapping
    "map_expressions_to_tones",
    "get_matching_expression_tags_cached",
    # Trigger vocab
    "get_all_trigger_tokens",
    "get_all_alias_tokens",
    # Exact alias scan helper
    "extract_exact_alias_tokens",
    # Rules
    "apply_expression_context_rules",
    "apply_expression_suppression_rules",
    # Glued-token vocab
    "get_glued_token_vocabulary",
    # Optional (drop if you want a stricter API)
    "inject_expression_modifiers",
]
