# src/color_sentiment_extractor/extraction/general/sentiment/router.py
from __future__ import annotations

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

from typing import List, Set, Tuple, Dict, Optional, Mapping, TypedDict, Iterable, Callable
from statistics import median
import os
import logging

from color_sentiment_extractor.extraction.color.logic import (
    aggregate_color_phrase_results,
    format_tone_modifier_mappings,
)
from color_sentiment_extractor.extraction.color.utils import (
    choose_representative_rgb,
    rgb_distance,
)

__all__ = ["build_color_sentiment_summary"]

logger = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────────────────────
RGB = Tuple[int, int, int]

class ColorSentimentSummary(TypedDict):
    matched_color_names: List[str]
    base_rgb: Optional[RGB]
    threshold: float

# ── ENV Config (tunable without code change) ─────────────────────────────────
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

RGB_THRESHOLD_DEFAULT: float = _env_float("RGB_THRESHOLD_DEFAULT", 60.0)
RGB_THRESHOLD_MIN: float = _env_float("RGB_THRESHOLD_MIN", 35.0)
RGB_THRESHOLD_MAX: float = _env_float("RGB_THRESHOLD_MAX", 75.0)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _stable_sorted_phrases(items: Iterable[str]) -> List[str]:
    """
    Case-insensitive, whitespace-trimmed, duplicate-free ordering.
    Dedup is done with casefold(); original casing of first occurrence is preserved.
    """
    canon: dict[str, str] = {}
    for s in items:
        if not s:
            continue
        t = s.strip()
        if not t:
            continue
        key = t.casefold()
        # preserve first-seen original casing
        canon.setdefault(key, t)
    # sort by case-insensitive key to keep deterministic order
    return [canon[k] for k in sorted(canon.keys())]

def _estimate_threshold(base_rgb: Optional[RGB], rgb_map: Mapping[str, RGB]) -> float:
    """
    Heuristic: if we have ≥2 colors, use median distance to base as threshold,
    clamped to a reasonable band. Else fall back to RGB_THRESHOLD_DEFAULT.
    """
    if not base_rgb or not rgb_map:
        return RGB_THRESHOLD_DEFAULT

    distances: List[float] = []
    for rgb in rgb_map.values():
        try:
            distances.append(float(rgb_distance(base_rgb, rgb)))
        except Exception:
            # Ignore malformed rgb tuples
            continue

    if len(distances) < 2:
        return RGB_THRESHOLD_DEFAULT

    med = median(distances)
    return max(RGB_THRESHOLD_MIN, min(RGB_THRESHOLD_MAX, med))

# ── Public API ────────────────────────────────────────────────────────────────
def build_color_sentiment_summary(
    sentiment: str,
    segments: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
    rgb_map: Dict[str, RGB],
    base_rgb_by_sentiment: Dict[str, Optional[RGB]],
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

    # Run phrase pipeline (materialize keys list to avoid view issues downstream)
    _matched_tones, matched_phrases, local_rgb_map = aggregate_color_phrase_results(
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
            # Optionally format for logs or downstream inspection (side-effect only)
            format_tone_modifier_mappings(matched_phrases, known_tones, known_modifiers)
        except Exception:
            # Never let logging/introspection break the route
            logger.debug("format_tone_modifier_mappings failed; continuing.", exc_info=True)

    # Choose base RGB from local extraction first; else keep prior for this sentiment
    base_rgb: Optional[RGB] = choose_representative_rgb(local_rgb_map) if local_rgb_map else None
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
