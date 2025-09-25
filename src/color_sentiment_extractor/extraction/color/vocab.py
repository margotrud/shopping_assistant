# src/color_sentiment_extractor/extraction/color/vocab.py

"""
Does: Define canonical color vocabularies (CSS, XKCD, project fallbacks).
Return: Pure frozen sets for safe import across the project.
"""

from __future__ import annotations

from typing import FrozenSet
import webcolors

# ── Load CSS names (lightweight) ─────────────────────────────────────────────
_CSS3_NAMES = frozenset(n.lower() for n in webcolors.CSS3_NAMES_TO_HEX.keys())
_CSS21_NAMES = frozenset(n.lower() for n in webcolors.CSS21_NAMES_TO_HEX.keys())

# CSS-only names (strict web standards)
WEB_ONLY_COLOR_NAMES: FrozenSet[str] = frozenset(_CSS3_NAMES | _CSS21_NAMES)

# ── XKCD names (lazy to avoid heavy import at module load) ───────────────────
def _load_xkcd_names() -> FrozenSet[str]:
    from matplotlib.colors import XKCD_COLORS  # lazy import
    return frozenset(k.replace("xkcd:", "").lower() for k in XKCD_COLORS.keys())

# Expose as a cached singleton to avoid repeated imports
_XKCD_NAMES: FrozenSet[str] | None = None

def get_xkcd_names() -> FrozenSet[str]:
    global _XKCD_NAMES
    if _XKCD_NAMES is None:
        _XKCD_NAMES = _load_xkcd_names()
    return _XKCD_NAMES

# ── Cosmetic fallbacks (project-specific) ────────────────────────────────────
COSMETIC_FALLBACK_TONES: FrozenSet[str] = frozenset({
    "nude", "ash", "ink", "almond", "champagne",
})

# ── Public aggregates ────────────────────────────────────────────────────────
# All web *named* colors (CSS + XKCD)
def get_web_named_color_names() -> FrozenSet[str]:
    return frozenset(WEB_ONLY_COLOR_NAMES | get_xkcd_names())

# Full tone universe used for tone matching (web + project fallbacks)
def get_known_tones() -> FrozenSet[str]:
    return frozenset(get_web_named_color_names() | COSMETIC_FALLBACK_TONES)

# Backward-compat aliases (if other modules expect these names)
# - Keep them as *functions* or *materialize once* depending on your needs.
all_webcolor_names = WEB_ONLY_COLOR_NAMES          # CSS-only (strict)
known_tones = get_known_tones()                    # full set at import-time
