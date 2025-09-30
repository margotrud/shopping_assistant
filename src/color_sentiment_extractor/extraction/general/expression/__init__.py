# src/color_sentiment_extractor/extraction/general/expression/__init__.py
from __future__ import annotations

"""
expression package
==================

Does: Public interface for expression matching and tone mapping.
Returns: Curated helpers for alias matching, expression→tone mapping, and glued-token vocab.
Used by: Color expression detection pipelines and compound/standalone extractors.
"""

__docformat__ = "google"

# Public API re-exports (curated)
from .expression_helpers import (
    # Matching & mapping
    map_expressions_to_tones,
    get_matching_expression_tags_cached,
    # Trigger vocab
    get_all_trigger_tokens,
    get_all_alias_tokens,
    # Exact alias scan helper
    extract_exact_alias_tokens,
    # Rules
    apply_expression_context_rules,
    apply_expression_suppression_rules,
    # Glued-token vocab
    get_glued_token_vocabulary,
)

# If you truly need the injector outside this package, expose a non-private alias.
# Otherwise, leave it unexported to keep the API clean.
from .expression_helpers import _inject_expression_modifiers as inject_expression_modifiers  # noqa: F401

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
