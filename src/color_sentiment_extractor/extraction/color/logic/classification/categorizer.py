"""
categorizer.py

Does:
    Build and format modifier–tone mappings from descriptive phrases.
    - Robust parsing of "modifier tone" across spaced, hyphenated, or glued forms.
    - Strict base recovery for modifiers (no fuzzy).
    - Filters cosmetic nouns from tones.

Returns:
    - build_tone_modifier_mappings(): (tones, modifiers, mod_to_tone, tone_to_mod)
    - format_tone_modifier_mappings(): {"modifiers": {...}, "tones": {...}}
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Iterable

from color_sentiment_extractor.extraction.general.token import recover_base, normalize_token
from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS  # via color/__init__.py


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_token(t: str) -> str:
    """Lowercase + trim; keep hyphens (meaningful for glued tokens)."""
    return normalize_token(t or "", keep_hyphens=True)

def _norm_set(values: Iterable[str]) -> Set[str]:
    """Normalize all items (lowercase, trimmed, hyphen-safe)."""
    return {_norm_token(v) for v in values if v}

def _split_into_candidates(phrase: str) -> List[Tuple[str, str]]:
    """
    Generate (mod, tone) candidates from a phrase:
    - "dusty rose"            → ("dusty", "rose")
    - "dusty-rose"/"dusty_rose" → ("dusty", "rose")
    - For longer phrases, slide a 2-token window to find the first plausible pair.
    """
    p = _norm_token(phrase)
    if not p:
        return []

    # Single glued token with hyphen/underscore → split
    if " " not in p:
        glue = p.replace("_", "-")
        if "-" in glue:
            parts = [s for s in glue.split("-") if s]
            if len(parts) == 2:
                return [(parts[0], parts[1])]
        return []

    tokens = [t for t in p.replace("_", " ").split() if t]
    if len(tokens) < 2:
        return []

    # Sliding 2-gram windows: (t0,t1), (t1,t2), ...
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def build_tone_modifier_mappings(
    phrases: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Does:
        Build bidirectional mappings between modifiers and tones from phrases.
        - Accepts spaced, hyphenated, or glued forms.
        - Recovers modifier base strictly (no fuzzy).
        - Filters cosmetic nouns from tones.

    Returns:
        tones, modifiers, mod_to_tone, tone_to_mod
    """
    known_tones = _norm_set(known_tones)
    known_modifiers = _norm_set(known_modifiers)
    cosmetic_nouns = set(COSMETIC_NOUNS)  # already normalized upstream

    tones: Set[str] = set()
    modifiers: Set[str] = set()
    mod_to_tone: Dict[str, Set[str]] = defaultdict(set)
    tone_to_mod: Dict[str, Set[str]] = defaultdict(set)

    for phrase in phrases:
        if not phrase:
            continue

        candidates = _split_into_candidates(phrase)
        if not candidates:
            continue

        chosen: Tuple[str, str] | None = None

        for mod_raw, tone_raw in candidates:
            tone = tone_raw
            # Skip tones that are cosmetic product nouns (e.g., "lipstick", "mascara")
            if tone in cosmetic_nouns:
                continue

            # Strict base recovery (no fuzzy) for the modifier
            base = recover_base(
                mod_raw,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
                fuzzy_threshold=90,
            )

            modifier = (
                mod_raw if mod_raw in known_modifiers
                else (base if base and base in known_modifiers else None)
            )

            if modifier and tone in known_tones:
                chosen = (modifier, tone)
                break  # take the first valid window

        if not chosen:
            continue

        modifier, tone = chosen
        tones.add(tone)
        modifiers.add(modifier)
        mod_to_tone[modifier].add(tone)
        tone_to_mod[tone].add(modifier)

    return tones, modifiers, mod_to_tone, tone_to_mod


def format_tone_modifier_mappings(
    phrases: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Does:
        Format mappings into sorted dicts for display or export.
    """
    tones, modifiers, mod_to_tone, tone_to_mod = build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )
    return {
        "modifiers": {m: sorted(ts) for m, ts in mod_to_tone.items()},
        "tones":     {t: sorted(ms) for t, ms in tone_to_mod.items()},
    }
