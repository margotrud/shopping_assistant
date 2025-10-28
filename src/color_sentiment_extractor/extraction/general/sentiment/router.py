# src/color_sentiment_extractor/extraction/general/sentiment/router.py

"""
sentiment_router.py.
===================

Builds a color response summary for a given sentiment category.
Each sentiment may yield distinct representative tones, phrases, and RGB mappings.

Outputs:
- matched_color_names: Sorted list of resolved phrases
- base_rgb: Representative RGB triplet for this sentiment (or None)
- threshold: RGB margin for inclusion
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from statistics import median
from typing import (
    TypedDict,
)

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
RGB = tuple[int, int, int]


class ColorSentimentSummary(TypedDict):
    matched_color_names: list[str]
    base_rgb: RGB | None
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
def _stable_sorted_phrases(items: Iterable[str]) -> list[str]:
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


def _estimate_threshold(base_rgb: RGB | None, rgb_map: Mapping[str, RGB]) -> float:
    """
    Heuristic: if we have ≥2 colors, use median distance to base as threshold,
    clamped to a reasonable band. Else fall back to RGB_THRESHOLD_DEFAULT.
    """
    if not base_rgb or not rgb_map:
        return RGB_THRESHOLD_DEFAULT

    distances: list[float] = []
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
    segments: list[str],
    known_tones: set[str],
    known_modifiers: set[str],
    expression_map: dict[str, dict[str, list[str]]],
    rgb_map: dict[str, RGB],
    base_rgb_by_sentiment: dict[str, RGB | None],
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
        expression_map: Canonical expression/alias map used by phrase aggregation.
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
    # 1. Run phrase pipeline once for this sentiment cluster
    _matched_tones, matched_phrases, local_rgb_map = aggregate_color_phrase_results(
        # NOTE: we switch to kwargs so mypy can check names+types
        segments=segments,
        known_tones=known_tones,
        known_modifiers=known_modifiers,
        expression_map=expression_map,
        all_webcolor_names=set(rgb_map.keys()),  # must be a set[str] not list[str]
        llm_client=None,
        cache=None,
        debug=debug,
    )

    # 2. Update global RGB map defensively with any new local discoveries
    if local_rgb_map:
        rgb_map.update(local_rgb_map)

    # 3. Debug / logging helpers (optional, never break the flow)
    if debug:
        try:
            # format_tone_modifier_mappings is side-effect/logging only
            format_tone_modifier_mappings(
                matched_phrases,
                known_tones,
                known_modifiers,
            )
        except Exception:
            logger.debug(
                "format_tone_modifier_mappings failed; continuing.",
                exc_info=True,
            )

    # 4. Derive / persist base color for this sentiment
    # Prefer local extraction; else fall back to a previously chosen base.
    base_rgb: RGB | None = choose_representative_rgb(local_rgb_map) if local_rgb_map else None
    if base_rgb is None:
        base_rgb = base_rgb_by_sentiment.get(sentiment)

    # Persist selection for this sentiment (can stay None)
    base_rgb_by_sentiment[sentiment] = base_rgb

    # 5. Threshold ~ how far variants may drift from base_rgb in this cluster
    threshold = _estimate_threshold(base_rgb, local_rgb_map or {})

    # 6. Stable output
    return ColorSentimentSummary(
        matched_color_names=_stable_sorted_phrases(matched_phrases),
        base_rgb=base_rgb,
        threshold=threshold,
    )
