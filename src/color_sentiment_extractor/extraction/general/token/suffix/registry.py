# src/color_sentiment_extractor/extraction/general/token/suffix/registry.py
from __future__ import annotations

"""
suffix.registry

Does: Centralize suffix recovery functions and provide a suffix-aware dispatcher.
Returns: RecoverFn type, SUFFIX_RECOVERY_FUNCS tuple, and recover_with_registry().
Used by: Base-recovery flows that need fast, deterministic suffix handling.
"""

from typing import Callable, AbstractSet, Optional, Tuple, Dict, cast

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

# Signature commune des fonctions de recovery (unique, pas de double définition)
RecoverFn = Callable[[str, AbstractSet[str], AbstractSet[str], bool], Optional[str]]

# Ordre global (fallback): du plus spécifique au plus générique
SUFFIX_RECOVERY_FUNCS: Tuple[RecoverFn, ...] = (
    cast(RecoverFn, recover_ier),     # trendier → trendy
    cast(RecoverFn, recover_ied),     # tried → try
    cast(RecoverFn, recover_ey),      # bronzey → bronze, beigey → beige
    cast(RecoverFn, recover_ee_to_y), # ivoree → ivory
    cast(RecoverFn, recover_ing),     # glowing → glow/e
    cast(RecoverFn, recover_ed),      # paled/muted/tapped → pale/mute/tap
    cast(RecoverFn, recover_ish),     # greenish → green, ivorish → ivory
    cast(RecoverFn, recover_en),      # golden → gold
    cast(RecoverFn, recover_ness),    # softness → soft
    cast(RecoverFn, recover_ly),      # softly → soft
    cast(RecoverFn, recover_er),      # darker → dark
    cast(RecoverFn, recover_y),       # shiny/rosy/creamy → shine/rose/cream
)

# Dispatcher suffix-aware: map suffix → fonctions candidates
_SUFFIX_MAP: Dict[str, Tuple[RecoverFn, ...]] = {
    "ness": (cast(RecoverFn, recover_ness),),
    "ing":  (cast(RecoverFn, recover_ing),),
    "ier":  (cast(RecoverFn, recover_ier),),
    "ied":  (cast(RecoverFn, recover_ied),),
    "ish":  (cast(RecoverFn, recover_ish),),
    "ey":   (cast(RecoverFn, recover_ey),),
    "ee":   (cast(RecoverFn, recover_ee_to_y),),
    "ly":   (cast(RecoverFn, recover_ly),),
    "en":   (cast(RecoverFn, recover_en),),
    "er":   (cast(RecoverFn, recover_er),),
    "ed":   (cast(RecoverFn, recover_ed),),
    "y":    (cast(RecoverFn, recover_y),),
}

# Recherche du suffixe le plus long d'abord (déterministe)
_SUFFIX_ORDER: Tuple[str, ...] = ("ness", "ing", "ier", "ied", "ish", "ey", "ee", "ly", "en", "er", "ed", "y")


def _candidate_funcs_for(token: str) -> Tuple[RecoverFn, ...]:
    """Does: Pick the minimal recovery set based on the token’s longest matching suffix."""
    t = (token or "").lower()
    for suf in _SUFFIX_ORDER:
        if t.endswith(suf):
            return _SUFFIX_MAP[suf]
    # Fallback global (ne devrait quasi jamais arriver)
    return SUFFIX_RECOVERY_FUNCS


def recover_with_registry(
    token: str,
    known_modifiers: AbstractSet[str],
    known_tones: AbstractSet[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Try all relevant suffix recoveries (suffix-aware) and return the first base found.
    Returns: Base string if recovered, else None.
    """
    for fn in _candidate_funcs_for(token):
        base = fn(token, known_modifiers, known_tones, debug)
        if base:
            return base
    return None
