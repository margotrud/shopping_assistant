"""
logic package
=============

High-level orchestration layer for color sentiment extraction.

Public API:
-----------
- process_color_phrase()         → normalize & resolve RGB for a phrase
- extract_phrases_from_segment() → validated phrase extraction
- aggregate_color_phrase_results() → aggregate tones, phrases, RGBs
- build_tone_modifier_mappings() → tone–modifier bidirectional mapping
- format_tone_modifier_mappings() → formatted dict view
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
