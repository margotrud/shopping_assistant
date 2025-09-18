"""
color_categorizer.py

Does:
    Provides utility functions to analyze and map color modifier-tone relationships from user-facing phrases.
    Includes:
        - Mapping builder that extracts (modifier, tone) pairs from descriptive phrases
        - Formatter to convert mappings into a display-friendly dictionary format

Returns:
    - Raw mappings (sets and dicts) of modifiers and tones via build_tone_modifier_mappings()
    - Sorted, nested JSON-like structure via format_tone_modifier_mappings()
"""

from collections import defaultdict
from typing import List, Set, Dict, Tuple

from extraction.general.token.base_recovery import recover_base
from extraction.general.token.normalize import normalize_token


# =============================================
# 1. Core Logic Function
# =============================================
def build_tone_modifier_mappings(
    phrases: List[str],
    known_tones: Set[str],
    known_modifiers: Set[str]
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

    tones = set()
    modifiers = set()
    mod_to_tone = defaultdict(set)
    tone_to_mod = defaultdict(set)

    for phrase in phrases:
        parts = [normalize_token(t, keep_hyphens=True) for t in phrase.split()]
        if len(parts) != 2:
            continue

        mod_raw, tone = parts
        base = recover_base(mod_raw)

        # Accept if the raw or recovered modifier is in known_modifiers
        modifier = mod_raw if mod_raw in known_modifiers else (
            base if base in known_modifiers else None
        )

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
    known_modifiers: Set[str]
) -> Dict[str, Dict[str, List[str]]]:
    """
       Does:
           Formats tone-modifier mappings into a sorted dictionary structure,
           suitable for display, export, or user-friendly representation.

       Returns:
           A dictionary with two top-level keys:
           {
               "modifiers": { modifier: [sorted tone list] },
               "tones": { tone: [sorted modifier list] }
           }
       """
    tones, modifiers, mod_to_tone, tone_to_mod = build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )

    return {
        "modifiers": {
            mod: sorted(list(tones)) for mod, tones in mod_to_tone.items()
        },
        "tones": {
            tone: sorted(list(mods)) for tone, mods in tone_to_mod.items()
        }
    }

