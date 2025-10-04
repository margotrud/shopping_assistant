"""
vocab
=====

Does: Define canonical color vocabularies (CSS, CSS2.1, XKCD) plus project fallbacks,
      and expose convenient accessors for web named colors and known tones.
Used By: Tone matching, alias validation, RGB/name resolution across extraction pipelines.
Returns: Pure frozen sets and getter functions (no side effects beyond lazy caching).
"""

from __future__ import annotations
from typing import FrozenSet
import logging

log = logging.getLogger(__name__)

try:
    import webcolors
except Exception:  # pragma: no cover
    webcolors = None  # optional in some test envs

# ── Load CSS names (lightweight) ─────────────────────────────────────────────
if webcolors is not None:
    _CSS3_NAMES = frozenset(n.lower() for n in webcolors.CSS3_NAMES_TO_HEX.keys())
    _CSS21_NAMES = frozenset(n.lower() for n in webcolors.CSS21_NAMES_TO_HEX.keys())
else:  # safe fallback
    _CSS3_NAMES = frozenset()
    _CSS21_NAMES = frozenset()

# CSS-only names (strict web standards)
WEB_ONLY_COLOR_NAMES: FrozenSet[str] = frozenset(_CSS3_NAMES | _CSS21_NAMES)

# ── XKCD names (lazy to avoid heavy import at module load) ───────────────────
def _load_xkcd_names() -> FrozenSet[str]:
    """Does: Load and normalize XKCD color names once (lazy import)."""
    try:
        from matplotlib.colors import XKCD_COLORS  # lazy import
        return frozenset(k.replace("xkcd:", "").lower() for k in XKCD_COLORS.keys())
    except Exception:
        log.debug("XKCD_COLORS not available; returning empty set", exc_info=True)
        return frozenset()

# Cached singleton
_XKCD_NAMES: FrozenSet[str] | None = None

def get_xkcd_names() -> FrozenSet[str]:
    """Does: Return cached XKCD color names, loading them on first call."""
    global _XKCD_NAMES
    if _XKCD_NAMES is None:
        _XKCD_NAMES = _load_xkcd_names()
    return _XKCD_NAMES

# ── Cosmetic fallbacks (project-specific) ────────────────────────────────────
COSMETIC_FALLBACK_TONES: FrozenSet[str] = frozenset({
    "nude", "ash", "ink", "almond", "champagne",
})

# ── Public aggregates ────────────────────────────────────────────────────────
def get_web_named_color_names() -> FrozenSet[str]:
    """Does: Return all web *named* colors (CSS + XKCD) as a frozen set."""
    return frozenset(WEB_ONLY_COLOR_NAMES | get_xkcd_names())

def get_known_tones() -> FrozenSet[str]:
    """Does: Return the full tone universe (web named colors + cosmetic fallbacks)."""
    return frozenset(get_web_named_color_names() | COSMETIC_FALLBACK_TONES)

# ── Canonical single accessor (API recommandée) ──────────────────────────────
def get_all_webcolor_names() -> FrozenSet[str]:
    """
    Does: Canonical accessor for all web color names used by the project.
    Returns: FrozenSet[str] (CSS + XKCD). Stable, test-friendly, single source of truth.
    """
    return get_web_named_color_names()

# ── Backward-compat (si des modules consomment encore ces symboles) ─────────
#   - Évite la duplication de logique : on redirige vers l’API canonique.
all_webcolor_names = get_all_webcolor_names()  # CSS+XKCD à l'import (si besoin)
known_tones = get_known_tones()
