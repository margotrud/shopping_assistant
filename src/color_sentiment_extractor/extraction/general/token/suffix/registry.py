# extraction/general/token/suffix/registry.py
from __future__ import annotations

"""
suffix.registry

Does: Centralize suffix recovery functions and provide a suffix-aware dispatcher.
Returns: RecoverFn type, SUFFIX_RECOVERY_FUNCS tuple, and recover_with_registry().
Used by: Base-recovery flows that need fast, deterministic suffix handling.
"""

from typing import Callable, Optional

from color_sentiment_extractor.extraction.general.token.suffix.recovery import (
    recover_y,
    recover_ed,
    recover_ee_to_y,
    recover_ing,
    recover_ish,
    recover_ly,
    recover_en,
    recover_ness,
    recover_ier,
    recover_er,
    recover_ied,
    recover_ey,
)

# Public surface
__all__ = ["RecoverFn", "SUFFIX_RECOVERY_FUNCS", "recover_with_registry"]

# Signature commune des fonctions de recovery
RecoverFn = Callable[[str, set[str], set[str], bool], Optional[str]]

# Ordre global (fallback): du plus spécifique au plus générique
SUFFIX_RECOVERY_FUNCS: tuple[RecoverFn, ...] = (
    recover_ier,     # trendier → trendy
    recover_ied,     # tried → try
    recover_ey,      # bronzey → bronze, beigey → beige
    recover_ee_to_y, # ivoree → ivory
    recover_ing,     # glowing → glow/e
    recover_ed,      # paled/muted/tapped → pale/mute/tap
    recover_ish,     # greenish → green, ivorish → ivory
    recover_en,      # golden → gold
    recover_ness,    # softness → soft
    recover_ly,      # softly → soft
    recover_er,      # darker → dark
    recover_y,       # shiny/rosy/creamy → shine/rose/cream
)

# Dispatcher suffix-aware: map suffix → fonctions candidates
_SUFFIX_MAP: dict[str, tuple[RecoverFn, ...]] = {
    "ness": (recover_ness,),
    "ing":  (recover_ing,),
    "ier":  (recover_ier,),
    "ied":  (recover_ied,),
    "ish":  (recover_ish,),
    "ey":   (recover_ey,),
    "ee":   (recover_ee_to_y,),
    "ly":   (recover_ly,),
    "en":   (recover_en,),
    "er":   (recover_er,),
    "ed":   (recover_ed,),
    "y":    (recover_y,),
}

# Recherche du suffixe le plus long d'abord (déterministe)
_SUFFIX_ORDER: tuple[str, ...] = ("ness", "ing", "ier", "ied", "ish", "ey", "ee", "ly", "en", "er", "ed", "y")


def _candidate_funcs_for(token: str) -> tuple[RecoverFn, ...]:
    """Does: Pick the minimal recovery set based on the token’s longest matching suffix."""
    t = (token or "").lower()
    for suf in _SUFFIX_ORDER:
        if t.endswith(suf):
            return _SUFFIX_MAP[suf]
    # Fallback global (ne devrait quasi jamais arriver)
    return SUFFIX_RECOVERY_FUNCS


def recover_with_registry(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Try all relevant suffix recoveries (suffix-aware) and return the first base found.
    Returns: Base string if recovered, else None.
    """
    for fn in _candidate_funcs_for(token):
        base = fn(token, known_modifiers, known_tones, debug=debug)
        if base:
            return base
    return None
