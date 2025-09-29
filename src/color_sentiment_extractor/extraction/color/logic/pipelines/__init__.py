"""
pipelines
=========

Does: High-level orchestration for color phrase extraction and RGB resolution.
Returns: Public API for phrase→tone extraction and phrase→RGB resolution.
Used By: sentiment analysis, modifier–tone mapping, and end-user parsing flows.
"""

from __future__ import annotations

# Public API re-exports
from .phrase_pipeline import (
    extract_all_descriptive_color_phrases,
    extract_phrases_from_segment,
    process_segment_colors,
    aggregate_color_phrase_results,
)
from .rgb_pipeline import (
    get_rgb_from_descriptive_color_llm_first,
    resolve_rgb_with_llm,
    process_color_phrase,
)

__all__ = [
    # phrase_pipeline
    "extract_all_descriptive_color_phrases",
    "extract_phrases_from_segment",
    "process_segment_colors",
    "aggregate_color_phrase_results",
    # rgb_pipeline
    "get_rgb_from_descriptive_color_llm_first",
    "resolve_rgb_with_llm",
    "process_color_phrase",
]

# Optional: consistent docstring style for tooling
__docformat__ = "google"
