# token/suffix/registry.py
from __future__ import annotations
from typing import Callable, Optional, Set, Tuple, List, Dict

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
    recover_ey,   # ← manquait dans ta version
)

# Signature commune des fonctions de recovery
RecoverFn = Callable[[str, Set[str], Set[str], bool], Optional[str]]

# Ordre: plus spécifiques → plus génériques
SUFFIX_RECOVERY_FUNCS: Tuple[RecoverFn, ...] = (
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

__all__ = ["RecoverFn", "SUFFIX_RECOVERY_FUNCS", "recover_with_registry"]

# --- Optionnel: dispatcher suffix-aware (efficient) ---
_SUFFIX_MAP: Dict[str, Tuple[RecoverFn, ...]] = {
    "ier": (recover_ier,),
    "ied": (recover_ied,),
    "ey":  (recover_ey,),
    "ee":  (recover_ee_to_y,),
    "ing": (recover_ing,),
    "ed":  (recover_ed,),
    "ish": (recover_ish,),
    "en":  (recover_en,),
    "ness": (recover_ness,),
    "ly":  (recover_ly,),
    "er":  (recover_er,),
    "y":   (recover_y,),
}

def _candidate_funcs_for(token: str) -> Tuple[RecoverFn, ...]:
    """Does: choose the minimal set of recovery fns based on token’s suffix."""
    t = token.lower()
    # Tester d’abord les suffixes à 4/3/2/1 lettres pour attraper le plus long
    for suf in ("ness", "ing", "ier", "ied", "ish", "ey", "ee", "ly", "en", "er", "ed", "y"):
        if t.endswith(suf):
            return (
                _SUFFIX_MAP[suf])
    return SUFFIX_RECOVERY_FUNCS  # fallback (ne devrait presque jamais arriver)

def recover_with_registry(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
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
