"""
utils
=====

Shared utilities for color distance, clustering, and name resolution.
Re-exports main functions from `rgb_distance` for convenient imports.
"""

from .rgb_distance import (
    rgb_distance,
    lab_distance,
    is_within_rgb_margin,
    choose_representative_rgb,
    find_similar_color_names,
    nearest_color_name,
    fuzzy_match_rgb_from_known_colors,
    _try_simplified_match,
    _parse_rgb_tuple,
)

__all__ = [
    "rgb_distance",
    "lab_distance",
    "is_within_rgb_margin",
    "choose_representative_rgb",
    "find_similar_color_names",
    "nearest_color_name",
    "fuzzy_match_rgb_from_known_colors",
    "_try_simplified_match",
    "_parse_rgb_tuple",
]
