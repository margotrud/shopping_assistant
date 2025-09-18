"""
rgb_distance.py
===============

Functions for RGB distance comparison, similarity clustering,
and fallback name matching based on perceptual closeness.

Used By:
--------
- LLM color resolution
- Sentiment RGB clustering
- Fallback phrase simplification
"""
import logging
from typing import Tuple, Dict, Optional, List
import re

from matplotlib.colors import CSS4_COLORS, XKCD_COLORS
from webcolors import hex_to_rgb

from extraction.color.vocab import all_webcolor_names
from extraction.general.token.normalize import normalize_token
logger = logging.getLogger(__name__)

# =============================================================================
# TITLE 1. RGB DISTANCE METRICS
# =============================================================================

def rgb_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """
    Does: Computes the Euclidean distance between two RGB color tuples.
    Return: Float distance value representing perceptual difference.
    """
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5


def is_within_rgb_margin(
    rgb1: Tuple[int, int, int],
    rgb2: Tuple[int, int, int],
    margin: float = 60.0
) -> bool:
    """
    Does: Checks if two RGB values are within a given distance margin.
    Return: True if distance ≤ margin; otherwise False.
    """
    return rgb_distance(rgb1, rgb2) <= margin


# =============================================================================
# TITLE 2. RGB CLUSTERING & REPRESENTATION
# =============================================================================

def choose_representative_rgb(
    rgb_map: Dict[str, Tuple[int, int, int]]
) -> Optional[Tuple[int, int, int]]:
    """
    Does: Selects the RGB tuple most central to all others in a set.
    Return: RGB tuple that minimizes total distance to all others; None if input is empty.
    """
    if not rgb_map:
        return None

    candidates = list(rgb_map.values())
    min_total = float("inf")
    best_rgb = None

    for candidate in candidates:
        total = sum(rgb_distance(candidate, other) for other in candidates)
        if total < min_total:
            min_total = total
            best_rgb = candidate

    return best_rgb


# =============================================================================
# TITLE 3. COLOR NAME LOOKUP BY DISTANCE
# =============================================================================

def find_similar_color_names(
    base_rgb: Tuple[int, int, int],
    known_rgb_map: Dict[str, Tuple[int, int, int]],
    threshold: float = 60.0
) -> List[str]:
    """
    Does: Returns all color names from a known RGB map that are close to the given RGB.
    Return: List of matching color names within the threshold margin.
    """
    return sorted([
        name for name, rgb in known_rgb_map.items()
        if is_within_rgb_margin(rgb, base_rgb, margin=threshold)
    ])


# =============================================================================
# TITLE 4. FUZZY NAME MATCHING FROM PHRASE
# =============================================================================

def fuzzy_match_rgb_from_known_colors(
    phrase: str,
) -> Optional[str]:
    """
    Does: Uses difflib to match a color phrase to a known named RGB color string.
    Return: Closest matching color name if found; otherwise None.
    """
    import difflib

    candidates = difflib.get_close_matches(phrase, all_webcolor_names, n=1, cutoff=0.75)
    if candidates:
        return candidates[0]
    return None

# =============================================================================
# TITLE 5. EXACT COLOR LOOKUP (CSS/XKCD)
# =============================================================================

def _try_simplified_match(name: str, debug=False) -> Optional[Tuple[int, int, int]]:
    """
    Does: Attempts exact match of normalized color name against CSS4 and XKCD color maps.
    Return: RGB tuple if found; otherwise None.
    """
    name = normalize_token(name, keep_hyphens=True).replace("-", " ")

    if name in CSS4_COLORS:
        hex_code = CSS4_COLORS[name]
        if debug:
            print(f"[🎨 CSS4 MATCH] '{name}' → {hex_code}")
        return hex_to_rgb(hex_code)

    xkcd_key = f"xkcd:{name}"
    if xkcd_key in XKCD_COLORS:
        hex_code = XKCD_COLORS[xkcd_key]
        if debug:
            print(f"[🎨 XKCD MATCH] '{name}' → {hex_code}")
        return hex_to_rgb(hex_code)

    if debug:
        print(f"[🕵️‍♀️ NOT FOUND] '{name}' not in XKCD or CSS4")
    return None

def _parse_rgb_tuple(response: str, debug=False) -> Optional[Tuple[int, int, int]]:
    """
    Does:
        Extracts an RGB tuple from a string using regex.
        Performs range validation and logs errors if debug is enabled.

    Returns:
        A valid (R, G, B) tuple if found and in range, else None.
    """
    match = re.search(r"\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)", response)
    if not match:
        if debug:
            logger.warning(f"[❌ PARSE FAIL] Could not extract RGB from response: {response}")
        return None

    r, g, b = map(int, match.groups())
    if all(0 <= val <= 255 for val in (r, g, b)):
        return (r, g, b)

    if debug:
        logger.warning(f"[❌ OUT-OF-RANGE] RGB out of bounds: {r}, {g}, {b}")
    return None
