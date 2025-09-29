# src/color_sentiment_extractor/extraction/color/suffix/rules.py
from __future__ import annotations

"""
suffix.rules
============

Does: Utilities for suffix handling on color tokens: check '-y' eligibility, detect CVC endings,
      build '-y'/'-ey' variants (overrides/allowlists/rules), and apply reverse overrides.
Used By: Suffix vocab builders, base-recovery flows, and compound/standalone extractors.
Returns: Public helpers for suffix generation/recovery used across extraction pipelines.

Notes importantes:
- EY_SUFFIX_ALLOWLIST peut être absent de `constants`; on bascule alors sur un ensemble vide.
"""

from functools import lru_cache
from typing import Optional, Set
import logging

# ── Public surface ───────────────────────────────────────────────────────────
__all__ = [
    "is_y_suffix_allowed",
    "is_cvc_ending",
    "build_y_variant",
    "build_ey_variant",
    "_apply_reverse_override",
    "_collapse_repeated_consonant",
]
__docformat__ = "google"

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Domain imports ───────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color.constants import (
    Y_SUFFIX_ALLOWLIST,
    Y_SUFFIX_OVERRIDE_FORMS,
    RECOVER_BASE_OVERRIDES,
)

# Optionnel: EY_SUFFIX_ALLOWLIST peut ne pas exister dans constants.
# On fournit un fallback neutre (ensemble vide) pour rester import-safe.
try:  # ImportError si la constante est absente
    from color_sentiment_extractor.extraction.color.constants import EY_SUFFIX_ALLOWLIST  # type: ignore
except ImportError:
    EY_SUFFIX_ALLOWLIST = frozenset()  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Heuristics / small tables
# ─────────────────────────────────────────────────────────────────────────────

# Consonant endings that naturally accept a '-y' coloristic form.
ALLOW_ENDS_FOR_Y = (
    "sh", "ch", "ss", "m", "n", "r", "l",
    "t", "d", "k", "p", "b", "f", "v",
    "s", "z", "c", "g", "h", "j",
)

# =============================================================================
# Core rules
# =============================================================================

def is_y_suffix_allowed(base: str) -> bool:
    """Does: Return True if base can accept '-y' via allowlist + heuristic rules. Returns: bool."""
    if base in Y_SUFFIX_ALLOWLIST:
        return True
    if len(base) < 3 or base.endswith(("y", "e")) or base[-1] in "aeiou":
        return False
    return base.endswith(ALLOW_ENDS_FOR_Y)


def is_cvc_ending(base: str) -> bool:
    """Does: Detect final CVC (consonant–vowel–consonant) excluding final w/x/y. Returns: bool."""
    if len(base) < 3:
        return False
    c1, v, c2 = base[-3], base[-2], base[-1]
    if not (c1.isalpha() and v.isalpha() and c2.isalpha()):
        return False
    if c1 in "aeiou" or v not in "aeiou" or c2 in "aeiouwxy":
        return False
    return True


def build_y_variant(base: str, debug: bool = False) -> Optional[str]:
    """Does: Build a '-y' variant using overrides, allowlist, then rules. Returns: token or None."""
    if base in Y_SUFFIX_OVERRIDE_FORMS:
        if debug:
            log.debug("[override -y] %s -> %s", base, Y_SUFFIX_OVERRIDE_FORMS[base])
        return Y_SUFFIX_OVERRIDE_FORMS[base]
    if base in Y_SUFFIX_ALLOWLIST or is_y_suffix_allowed(base):
        if debug:
            log.debug("[rule -y] %s -> %sy", base, base)
        return base + "y"
    if debug:
        log.debug("[deny -y] %s", base)
    return None


def build_ey_variant(base: str, raw: str, debug: bool = False) -> Optional[str]:
    """Does: Build an '-ey' variant via dedicated allowlist or strict sibilant rule. Returns: token/None."""
    def _ey(stem: str) -> str:
        return (stem[:-1] if stem.endswith("e") else stem) + "ey"

    # 1) dedicated allowlist first (si présente)
    if raw in EY_SUFFIX_ALLOWLIST or base in EY_SUFFIX_ALLOWLIST:
        if debug:
            log.debug("[allowlist -ey] %s/%s -> %s", raw, base, _ey(base))
        return _ey(base)
    # 2) constrained sibilant endings
    if len(base) > 2 and base.endswith(("ge", "ce", "ze", "se")):
        if debug:
            log.debug("[rule -ey] %s -> %s", base, _ey(base))
        return _ey(base)
    if debug:
        log.debug("[deny -ey] %s (raw=%s)", base, raw)
    return None

# =============================================================================
# Overrides (reverse) and minor normalizations
# =============================================================================

@lru_cache(maxsize=1)
def _stripped_override_map() -> dict[str, str]:
    """Does: Map stripped override tokens (trim 'y'/'ed') to normalized bases. Returns: dict."""
    out: dict[str, str] = {}
    for k, v in RECOVER_BASE_OVERRIDES.items():
        stripped = k[:-1] if k.endswith("y") else k[:-2] if k.endswith("ed") else k
        out[stripped] = v
    return out


def _apply_reverse_override(base: str, token: str, debug: bool = False) -> str:
    """Does: Use stripped token to look up an override base, falling back to provided base. Returns: str."""
    m = _stripped_override_map()
    key = token[:-1] if token.endswith("y") else token[:-2] if token.endswith("ed") else token
    out = m.get(key, base)
    if debug and out != base:
        log.debug("[reverse override] %s (from %s) -> %s", base, token, out)
    return out


def _collapse_repeated_consonant(
    base: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> str:
    """Does: Collapse doubled final consonant if collapsed form exists in known sets. Returns: str."""
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed = base[:-1]
        if collapsed in known_modifiers or collapsed in known_tones:
            if debug:
                log.debug("[collapse double] %s -> %s", base, collapsed)
            return collapsed
    return base
