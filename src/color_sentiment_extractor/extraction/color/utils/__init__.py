"""
utils package.
=============

Does: Provide general utilities for normalization, distance calculations,
      config loading, and shared helpers across extraction modules.
"""

from .rgb_distance import (
    _parse_rgb_tuple,
    _try_simplified_match,
    choose_representative_rgb,
    find_similar_color_names,
    fuzzy_match_rgb_from_known_colors,
    is_within_rgb_margin,
    lab_distance,
    nearest_color_name,
    rgb_distance,
)

__all__ = [
    "rgb_distance",
    "lab_distance",
    "is_within_rgb_margin",
    "choose_representative_rgb",
    "find_similar_color_names",
    "nearest_color_name",
    "fuzzy_match_rgb_from_known_colors",
    "_parse_rgb_tuple",
    "_try_simplified_match",
]

__docformat__ = "google"
