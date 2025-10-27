"""
logic
=====

Thin namespace for classification/ and pipelines/.
- Avoid eager imports to prevent circular dependencies at runtime.
- Provide TYPE_CHECKING stubs so IDEs/static analyzers see symbols.

Public API:
- classification: build_tone_modifier_mappings, format_tone_modifier_mappings
- pipelines    : extract_all_descriptive_color_phrases, extract_phrases_from_segment,
                 process_segment_colors, aggregate_color_phrase_results,
                 get_rgb_from_descriptive_color_llm_first, resolve_rgb_with_llm,
                 process_color_phrase
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# ---- Static typing / IDE stubs (do NOT run at runtime) ----------------------
if TYPE_CHECKING:
    # classification
    from .classification.categorizer import (
        build_tone_modifier_mappings,
        format_tone_modifier_mappings,
    )
    # pipelines
    from .pipelines.phrase_pipeline import (
        extract_all_descriptive_color_phrases,
        extract_phrases_from_segment,
        process_segment_colors,
        aggregate_color_phrase_results,
    )
    from .pipelines.rgb_pipeline import (
        get_rgb_from_descriptive_color_llm_first,
        resolve_rgb_with_llm,
        process_color_phrase,
    )

# ---- Lazy runtime exports (PEP 562) -----------------------------------------
def __getattr__(name: str):
    # classification (no dependency to pipelines)
    if name in ("build_tone_modifier_mappings", "format_tone_modifier_mappings"):
        from .classification.categorizer import (
            build_tone_modifier_mappings as _btmm,
            format_tone_modifier_mappings as _ftmm,
        )
        return {
            "build_tone_modifier_mappings": _btmm,
            "format_tone_modifier_mappings": _ftmm,
        }[name]

    # pipelines (loaded on demand only)
    if name in (
        "extract_all_descriptive_color_phrases",
        "extract_phrases_from_segment",
        "process_segment_colors",
        "aggregate_color_phrase_results",
        "get_rgb_from_descriptive_color_llm_first",
        "resolve_rgb_with_llm",
        "process_color_phrase",
    ):
        if name in (
            "extract_all_descriptive_color_phrases",
            "extract_phrases_from_segment",
            "process_segment_colors",
            "aggregate_color_phrase_results",
        ):
            from .pipelines.phrase_pipeline import (
                extract_all_descriptive_color_phrases as _eacp,
                extract_phrases_from_segment as _epfs,
                process_segment_colors as _psc,
                aggregate_color_phrase_results as _acpr,
            )
            return {
                "extract_all_descriptive_color_phrases": _eacp,
                "extract_phrases_from_segment": _epfs,
                "process_segment_colors": _psc,
                "aggregate_color_phrase_results": _acpr,
            }[name]

        from .pipelines.rgb_pipeline import (
            get_rgb_from_descriptive_color_llm_first as _grdclf,
            resolve_rgb_with_llm as _rrwl,
            process_color_phrase as _pcp,
        )
        return {
            "get_rgb_from_descriptive_color_llm_first": _grdclf,
            "resolve_rgb_with_llm": _rrwl,
            "process_color_phrase": _pcp,
        }[name]

    raise AttributeError(name)


__all__ = [
    # classification
    "build_tone_modifier_mappings",
    "format_tone_modifier_mappings",
    # pipelines
    "extract_all_descriptive_color_phrases",
    "extract_phrases_from_segment",
    "process_segment_colors",
    "aggregate_color_phrase_results",
    "get_rgb_from_descriptive_color_llm_first",
    "resolve_rgb_with_llm",
    "process_color_phrase",
]

__docformat__ = "google"
