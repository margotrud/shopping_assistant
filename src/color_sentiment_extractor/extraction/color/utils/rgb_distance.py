"""
rgb_distance.py
===============

Functions for RGB distance comparison, similarity clustering,
and fallback name matching based on perceptual closeness.
"""

from __future__ import annotations
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable, Iterable

# ✔ pull webcolor names via color package (re-exported by color/__init__.py)
from color_sentiment_extractor.extraction.color import all_webcolor_names
from color_sentiment_extractor.extraction.general.token import normalize_token

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────
RGB = Tuple[int, int, int]

# Helper: normalize access to webcolor name list (supports var or callable)
def _get_all_webcolor_names() -> List[str]:
    try:
        # If it's a function (callable), call it.
        if callable(all_webcolor_names):
            names = all_webcolor_names()
        else:
            names = all_webcolor_names  # type: ignore[assignment]
        # Ensure list[str] with normalized spacing like elsewhere
        if isinstance(names, Iterable):
            return list(names)
    except Exception:
        pass
    return []

# =============================================================================
# 1) CORE DISTANCES
# =============================================================================

def rgb_distance(rgb1: RGB, rgb2: RGB) -> float:
    """Does: Compute Euclidean distance in sRGB space."""
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5


def _srgb_to_linear(v: float) -> float:
    v = v / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return x, y, z


def _f_lab(t: float) -> float:
    d = 6 / 29
    return t ** (1 / 3) if t > d ** 3 else (t / (3 * d * d) + 4 / 29)


def _rgb_to_lab(rgb: RGB) -> Tuple[float, float, float]:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # D65 white
    x, y, z = _rgb_to_xyz(rgb)
    fx, fy, fz = _f_lab(x / Xn), _f_lab(y / Yn), _f_lab(z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def lab_distance(rgb1: RGB, rgb2: RGB) -> float:
    """Does: Compute ΔE76 (Lab distance) between two RGB colors."""
    L1, a1, b1 = _rgb_to_lab(rgb1)
    L2, a2, b2 = _rgb_to_lab(rgb2)
    return ((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) ** 0.5


def is_within_rgb_margin(rgb1: RGB, rgb2: RGB, margin: float = 60.0) -> bool:
    """Does: Check if two RGB colors are within Euclidean margin."""
    return rgb_distance(rgb1, rgb2) <= margin


# =============================================================================
# 2) CLUSTERING / REPRESENTATION
# =============================================================================

def choose_representative_rgb(
    rgb_map: Dict[str, RGB],
    metric: Callable[[RGB, RGB], float] = lab_distance
) -> Optional[RGB]:
    """Does: Pick the RGB most central among candidates by chosen metric."""
    if not rgb_map:
        return None
    values = list(rgb_map.values())
    best, best_sum = None, float("inf")
    for c in values:
        s = sum(metric(c, o) for o in values)
        if s < best_sum:
            best, best_sum = c, s
    return best


# =============================================================================
# 3) NAMED COLOR MAPS (lazy import)
# =============================================================================

def _get_named_color_map() -> Dict[str, RGB]:
    """Does: Merge CSS4 and XKCD color dicts into {name: RGB} with lazy import."""
    from matplotlib.colors import CSS4_COLORS, XKCD_COLORS
    from webcolors import hex_to_rgb

    named: Dict[str, RGB] = {}
    for name, hx in CSS4_COLORS.items():
        named[normalize_token(name).replace("-", " ")] = tuple(hex_to_rgb(hx))  # type: ignore
    for name, hx in XKCD_COLORS.items():
        key = normalize_token(name.replace("xkcd:", "")).replace("-", " ")
        named.setdefault(key, tuple(hex_to_rgb(hx)))  # type: ignore
    return named


# =============================================================================
# 4) LOOKUPS & MATCHING
# =============================================================================

def find_similar_color_names(
    base_rgb: RGB,
    known_rgb_map: Dict[str, RGB],
    threshold: float = 30.0,
    metric: Callable[[RGB, RGB], float] = lab_distance
) -> List[str]:
    """Does: Return names within threshold distance of base_rgb."""
    return sorted([n for n, rgb in known_rgb_map.items() if metric(rgb, base_rgb) <= threshold])


def nearest_color_name(
    rgb: RGB,
    known_rgb_map: Optional[Dict[str, RGB]] = None,
    metric: Callable[[RGB, RGB], float] = lab_distance
) -> Optional[str]:
    """Does: Find nearest color name in map by chosen metric."""
    known = known_rgb_map or _get_named_color_map()
    best_name, best_d = None, float("inf")
    for name, ref in known.items():
        d = metric(rgb, ref)
        if d < best_d:
            best_name, best_d = name, d
    return best_name


def fuzzy_match_rgb_from_known_colors(
    phrase: str,
    n: int = 1,
    cutoff: float = 0.75
) -> Optional[str]:
    """Does: Fuzzy match phrase to a known webcolor name."""
    import difflib
    q = normalize_token(phrase).replace("-", " ")
    names = _get_all_webcolor_names()
    candidates = difflib.get_close_matches(q, names, n=n, cutoff=cutoff)
    return candidates[0] if candidates else None


# =============================================================================
# 5) EXACT LOOKUP (CSS/XKCD)
# =============================================================================

def _try_simplified_match(name: str, debug: bool = False) -> Optional[RGB]:
    """Does: Try exact normalized name against CSS4/XKCD maps."""
    from matplotlib.colors import CSS4_COLORS, XKCD_COLORS
    from webcolors import hex_to_rgb

    key = normalize_token(name, keep_hyphens=True).replace("-", " ")

    if key in CSS4_COLORS:
        hx = CSS4_COLORS[key]
        if debug:
            print(f"[🎨 CSS4 MATCH] '{key}' → {hx}")
        return tuple(hex_to_rgb(hx))  # type: ignore

    xkcd_key = f"xkcd:{key}"
    if xkcd_key in XKCD_COLORS:
        hx = XKCD_COLORS[xkcd_key]
        if debug:
            print(f"[🎨 XKCD MATCH] '{key}' → {hx}")
        return tuple(hex_to_rgb(hx))  # type: ignore

    if debug:
        print(f"[🕵️‍♀️ NOT FOUND] '{key}' not in XKCD or CSS4")
    return None


# =============================================================================
# 6) RGB PARSER
# =============================================================================

_RGB_PATTERNS = [
    r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
    r"\[\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\]",
    r"\b(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\b",
]

def _parse_rgb_tuple(response: str, debug: bool = False) -> Optional[RGB]:
    """Does: Extract an RGB triple from text; validates 0–255 range."""
    for pat in _RGB_PATTERNS:
        m = re.search(pat, response)
        if m:
            r, g, b = map(int, m.groups())
            if all(0 <= v <= 255 for v in (r, g, b)):
                return (r, g, b)
            if debug:
                logger.warning(f"[❌ OUT-OF-RANGE] RGB out of bounds: {r}, {g}, {b}")
            return None
    if debug:
        logger.warning(f"[❌ PARSE FAIL] Could not extract RGB from: {response!r}")
    return None
