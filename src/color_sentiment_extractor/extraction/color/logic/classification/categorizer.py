"""
color_categorizer.py

Does:
    Provides utility functions to analyze and map color modifier–tone relationships from user-facing phrases.
    Includes:
        - Mapping builder that extracts (modifier, tone) pairs from descriptive phrases
        - Formatter to convert mappings into a display-friendly dictionary format

Returns:
    - Raw mappings (sets and dicts) of modifiers and tones via build_tone_modifier_mappings()
    - Sorted, nested JSON-like structure via format_tone_modifier_mappings()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token


# =============================================
# 0. Helpers
# =============================================
def _norm_token(t: str) -> str:
    """Lowercase+strip normalization wrapper with hyphen support."""
    return normalize_token(t or "", keep_hyphens=True)


def _norm_set(values: Set[str]) -> Set[str]:
    """Normalize all items in a set (lowercase, trimmed, hyphen-safe)."""
    return { _norm_token(v) for v in values if v }


# =============================================
# 1. Core Logic Function
# =============================================
def build_tone_modifier_mappings(
    phrases: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Does:
        Builds bidirectional mappings between modifiers and tones from descriptive two-word phrases.
        Normalizes modifier tokens using recover_base() to account for suffix variants like 'glowy' → 'glow'.

    Returns:
        - tones (Set[str]): All matched tone tokens.
        - modifiers (Set[str]): All matched modifier tokens (including suffix variants).
        - mod_to_tone (Dict[str, Set[str]]): Mapping from each modifier to its associated tones.
        - tone_to_mod (Dict[str, Set[str]]): Mapping from each tone to its associated modifiers.
    """
    # Normalize vocabularies for robust membership checks
    known_tones = _norm_set(known_tones)
    known_modifiers = _norm_set(known_modifiers)

    tones: Set[str] = set()
    modifiers: Set[str] = set()
    mod_to_tone: Dict[str, Set[str]] = defaultdict(set)
    tone_to_mod: Dict[str, Set[str]] = defaultdict(set)

    for phrase in phrases:
        if not phrase:
            continue

        # Normalize and split → expect 2 tokens (modifier tone)
        parts = [_norm_token(t) for t in phrase.split()]
        if len(parts) != 2:
            continue

        mod_raw, tone = parts[0], parts[1]
        if not mod_raw or not tone:
            continue

        # Try to recover a base form for the modifier (no fuzzy here to stay strict)
        base = recover_base(
            mod_raw,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            debug=False,
            fuzzy_fallback=False,
            fuzzy_threshold=90,
        )

        # Accept raw modifier if known; else accept recovered base if it is known
        modifier = (
            mod_raw if mod_raw in known_modifiers
            else (base if base and base in known_modifiers else None)
        )

        # Only register valid (modifier, tone) pairs
        if modifier and tone in known_tones:
            tones.add(tone)
            modifiers.add(modifier)
            mod_to_tone[modifier].add(tone)
            tone_to_mod[tone].add(modifier)

    return tones, modifiers, mod_to_tone, tone_to_mod


# =============================================
# 2. Formatter Function
# =============================================
def format_tone_modifier_mappings(
    phrases: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Does:
        Formats tone–modifier mappings into a sorted dictionary structure,
        suitable for display, export, or user-friendly representation.

    Returns:
        {
            "modifiers": { modifier: [sorted tone list] },
            "tones": { tone: [sorted modifier list] }
        }
    """
    tones, modifiers, mod_to_tone, tone_to_mod = build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )

    # Produce deterministic (sorted) output for stable diffs/UI
    return {
        "modifiers": { mod: sorted(tones_set) for mod, tones_set in mod_to_tone.items() },
        "tones":     { tone: sorted(mods_set) for tone, mods_set in tone_to_mod.items() },
    }
