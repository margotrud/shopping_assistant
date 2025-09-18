"""
sentiment_router.py
===================

Builds a color response summary for a given sentiment category.
Each sentiment may yield distinct representative tones, phrases, and RGB mappings.

Used By:
--------
- Sentiment-level breakdown of user style preferences.

Key Outputs:
------------
- matched_color_names: Sorted list of resolved phrases
- base_rgb: Representative RGB triplet for this sentiment
- threshold: RGB margin for inclusion
"""
from typing import List, Set, Tuple, Dict, Optional, Union

from src.color_sentiment_extractor.extraction.color.logic.color_pipeline import aggregate_color_phrase_results
from extraction.color.utils.rgb_distance import choose_representative_rgb
from extraction.color.logic.color_categorizer import format_tone_modifier_mappings

RGB_THRESHOLD = 60.0

def build_color_sentiment_summary(
    sentiment: str,
    segments: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
    rgb_map: Dict[str, Tuple[int, int, int]],
    base_rgb_by_sentiment: Dict[str, Optional[Tuple[int, int, int]]]
) -> Dict[str, Union[List[str], Optional[Tuple[int, int, int]], float]]:
    """
    Aggregates all color phrases and RGBs associated with a sentiment-labeled group of segments.

    Args:
        sentiment (str): Label (e.g., 'romantic', 'edgy') for grouping.
        segments (List[str]): All related user utterances or queries.
        known_tones (Set[str]): Tone vocabulary.
        known_modifiers (Set[str]): Modifier vocabulary.
        rgb_map (Dict[str, Tuple[int, int, int]]): Cache of known phrase → RGB mappings.
        base_rgb_by_sentiment (Dict[str, Optional[Tuple[int, int, int]]]): Where base RGB is stored per sentiment.

    Returns:
        Dict with:
            - "matched_color_names" (List[str])
            - "base_rgb" (Tuple[int, int, int] or None)
            - "threshold" (float)
    """
    matched_tones, matched_phrases, local_rgb_map = aggregate_color_phrase_results(
        segments=segments,
        known_modifiers=known_modifiers,
        all_webcolor_names=rgb_map.keys(),
        llm_client=None,
        cache=None,
        debug=False
    )

    # Update global RGB map
    rgb_map.update(local_rgb_map)

    # Optionally format for logs or downstream use
    format_tone_modifier_mappings(matched_phrases, known_tones, known_modifiers)

    base_rgb = choose_representative_rgb(local_rgb_map)
    base_rgb_by_sentiment[sentiment] = base_rgb

    return {
        "matched_color_names": sorted(matched_phrases),
        "base_rgb": base_rgb,
        "threshold": RGB_THRESHOLD
    }

