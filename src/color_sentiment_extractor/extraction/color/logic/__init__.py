"""
logic package
=============

High-level orchestration layer for color sentiment extraction.

Submodules:
-----------
- classification/ : categorization utilities (tone–modifier mapping, etc.)
- pipelines/      : end-to-end pipelines (phrase extraction, RGB resolution)
"""

from .pipelines.rgb_pipeline import process_color_phrase
from .pipelines.phrase_pipeline import extract_phrases_from_segment, aggregate_color_phrase_results
from .classification.categorizer import build_tone_modifier_mappings, format_tone_modifier_mappings

__all__ = [
    "process_color_phrase",
    "extract_phrases_from_segment",
    "aggregate_color_phrase_results",
    "build_tone_modifier_mappings",
    "format_tone_modifier_mappings",
]
