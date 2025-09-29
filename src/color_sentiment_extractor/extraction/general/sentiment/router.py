"""
sentiment_router.py
===================

Builds a color response summary for a given sentiment category.
Each sentiment may yield distinct representative tones, phrases, and RGB mappings.

Outputs:
- matched_color_names: Sorted list of resolved phrases
- base_rgb: Representative RGB triplet for this sentiment (or None)
- threshold: RGB margin for inclusion
"""
from __future__ import annotations

from typing import List, Set, Tuple, Dict, Optional, Union, Mapping, TypedDict
from statistics import median

from color_sentiment_extractor.extraction.color.logic import (
    aggregate_color_phrase_results, format_tone_modifier_mappings
)
from color_sentiment_extractor.extraction.color.utils import (
    choose_representative_rgb,
    rgb_distance,
)

# Fallback threshold if dispersion-based estimate is not applicable
RGB_THRESHOLD: float = 60.0

class ColorSentimentSummary(TypedDict):
    matched_color_names: List[str]
    base_rgb: Optional[Tuple[int, int, int]]
    threshold: float


def _stable_sorted_phrases(items: List[str]) -> List[str]:
    # Case-insensitive, accent/locale-agnostic enough for our tokens
    return sorted({s.strip(): None for s in items if s and s.strip()}.keys(), key=lambda s: s.casefold())


def _estimate_threshold(base_rgb: Optional[Tuple[int, int, int]], rgb_map: Mapping[str, Tuple[int, int, int]]) -> float:
    """
    Heuristic: if we have ≥2 colors, use median distance to base as threshold,
    clamped to a reasonable band. Else fall back to RGB_THRESHOLD.
    """
    if not base_rgb:
        return RGB_THRESHOLD
    if not rgb_map:
        return RGB_THRESHOLD

    distances: List[float] = []
    for name, rgb in rgb_map.items():
        try:
            distances.append(float(rgb_distance(base_rgb, rgb)))
        except Exception:
            # ignore malformed rgb tuples
            continue

    if len(distances) < 2:
        return RGB_THRESHOLD

    med = median(distances)
    # Clamp between 35 and 75 to avoid overly tight/loose thresholds
    return max(35.0, min(75.0, med))


def build_color_sentiment_summary(
    sentiment: str,
    segments: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
    rgb_map: Dict[str, Tuple[int, int, int]],
    base_rgb_by_sentiment: Dict[str, Optional[Tuple[int, int, int]]],
    *,
    debug: bool = False,
) -> ColorSentimentSummary:
    """
    Aggregates all color phrases and RGBs associated with a sentiment-labeled group of segments.

    Args:
        sentiment: Label (e.g., 'romantic', 'edgy') for grouping.
        segments: All related user utterances or queries.
        known_tones: Tone vocabulary.
        known_modifiers: Modifier vocabulary.
        rgb_map: Cache of known phrase → RGB mappings (will be updated in-place).
        base_rgb_by_sentiment: Storage for selected base RGB per sentiment (updated in-place).
        debug: When True, also formats tone/modifier mappings for logs.

    Returns:
        {
          "matched_color_names": [...],
          "base_rgb": (r,g,b) | None,
          "threshold": float
        }
    """
    # Run phrase pipeline (list(...) to avoid passing dict_keys views downstream)
    matched_tones, matched_phrases, local_rgb_map = aggregate_color_phrase_results(
        segments=segments,
        known_modifiers=known_modifiers,
        all_webcolor_names=list(rgb_map.keys()),
        llm_client=None,
        cache=None,
        debug=debug,
    )

    # Update global RGB map defensively
    if local_rgb_map:
        rgb_map.update(local_rgb_map)

    if debug:
        try:
            # Optionally format for logs or downstream inspection
            format_tone_modifier_mappings(matched_phrases, known_tones, known_modifiers)
        except Exception:
            # Never let logging/introspection break the route
            pass

    # Choose base RGB from local extraction first
    base_rgb = choose_representative_rgb(local_rgb_map) if local_rgb_map else None

    # If nothing local, preserve any existing base set for this sentiment
    if base_rgb is None:
        base_rgb = base_rgb_by_sentiment.get(sentiment)

    # Persist selection for this sentiment (can be None)
    base_rgb_by_sentiment[sentiment] = base_rgb

    # Derive a threshold from dispersion when possible; else fallback constant
    threshold = _estimate_threshold(base_rgb, local_rgb_map or {})

    return ColorSentimentSummary(
        matched_color_names=_stable_sorted_phrases(matched_phrases),
        base_rgb=base_rgb,
        threshold=threshold,
    )
