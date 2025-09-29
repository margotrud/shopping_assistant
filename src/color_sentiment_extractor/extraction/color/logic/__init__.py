"""
logic
=====

Does: Aggregate high-level color logic: phrase extraction/validation, tone–modifier classification, and RGB orchestration.
Returns: Public API re-exports for pipelines (phrase/RGB) and classification (mappings/formatting).
Used By: app entrypoints, sentiment module, modifier–tone mapping, and end-user parsing flows.
"""

from __future__ import annotations

# ── Public API re-exports ─────────────────────────────────────────────────────
# Pipelines (phrase extraction & RGB resolution)
from .pipelines import (
    extract_all_descriptive_color_phrases,
    extract_phrases_from_segment,
    process_segment_colors,
    aggregate_color_phrase_results,
    get_rgb_from_descriptive_color_llm_first,
    resolve_rgb_with_llm,
    process_color_phrase,
)

# Classification (tone↔modifier mappings)
from .classification import (
    build_tone_modifier_mappings,
    format_tone_modifier_mappings,
)

__all__ = [
    # Pipelines
    "extract_all_descriptive_color_phrases",
    "extract_phrases_from_segment",
    "process_segment_colors",
    "aggregate_color_phrase_results",
    "get_rgb_from_descriptive_color_llm_first",
    "resolve_rgb_with_llm",
    "process_color_phrase",
    # Classification
    "build_tone_modifier_mappings",
    "format_tone_modifier_mappings",
]

# Optional: enforce a consistent docstring style for tooling
__docformat__ = "google"
