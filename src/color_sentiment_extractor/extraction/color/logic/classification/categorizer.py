"""
categorizer.py
Does: Parse phrases to build modifier↔tone mappings (spaced/hyphen/glued). Enforces strict base recovery for modifiers (no fuzzy) and filters cosmetic nouns from tones.
Returns: build_tone_modifier_mappings() → (tones, modifiers, mod_to_tone, tone_to_mod); format_tone_modifier_mappings() → {"modifiers": {...}, "tones": {...}}.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

# Façades stables (tu as rempli les __init__.py) : on importe via les portails publics.
from color_sentiment_extractor.extraction.general.token import recover_base, normalize_token
from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS

__all__ = ["build_tone_modifier_mappings", "format_tone_modifier_mappings"]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Gérer les tirets Unicode fréquents pour une robustesse aux sources hétérogènes
UNICODE_HYPHENS: Tuple[str, ...] = ("-", "–", "—", "‒", "−")


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
    Generate (modifier, tone) 2-gram candidates from a phrase.
    - Accepts spaced, hyphenated, and glued forms (incl. unicode hyphens, underscores).
    - For N≥2 tokens/segments, returns a sliding 2-gram window.
    """
    p = _norm_token(phrase)
    if not p:
        return []

    # Unifier séparateurs hyphens/underscores
    for h in UNICODE_HYPHENS:
        p = p.replace(h, "-")
    p = p.replace("_", "-")

    # Cas "glued" sans espace : on découpe sur hyphens et fait un 2-gram sliding
    if " " not in p:
        parts = [s for s in p.split("-") if s]
        if len(parts) >= 2:
            return [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
        return []

    # Cas avec espaces : on tolère underscores côté espaces (déjà remplacés ci-dessus)
    tokens = [t for t in p.split() if t]
    if len(tokens) < 2:
        return []

    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────

def build_tone_modifier_mappings(
    phrases: Iterable[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Does: Build bidirectional mappings (modifier↔tone) from phrases with strict base recovery; filters cosmetic nouns.
    Returns: (tones, modifiers, mod_to_tone, tone_to_mod) as sets/dicts with deterministic content.
    """
    known_tones = _norm_set(known_tones)
    known_modifiers = _norm_set(known_modifiers)
    cosmetic_nouns = _norm_set(COSMETIC_NOUNS)

    tones: Set[str] = set()
    modifiers: Set[str] = set()
    mod_to_tone: Dict[str, Set[str]] = defaultdict(set)
    tone_to_mod: Dict[str, Set[str]] = defaultdict(set)

    for phrase in phrases or []:
        if not phrase:
            continue

        candidates = _split_into_candidates(phrase)
        if not candidates:
            continue

        # Mini-heuristique : privilégier un hit "direct" (mod_raw ∈ known_modifiers),
        # sinon un hit "via base" (recover_base) ; toujours tone ∈ known_tones.
        best: Tuple[str, str] | None = None

        for mod_raw, tone_raw in candidates:
            tone = tone_raw
            if tone in cosmetic_nouns:
                continue  # exclure les noms de produits cosmétiques en tones

            # Strict base recovery (no fuzzy)
            base = recover_base(
                mod_raw,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
                fuzzy_threshold=90,
            )

            direct = mod_raw if mod_raw in known_modifiers else None
            via_base = base if base and base in known_modifiers else None

            if (direct or via_base) and tone in known_tones:
                best = (direct or via_base, tone)
                if direct:
                    break  # priorité au match direct sur known_modifiers

        if not best:
            continue

        modifier, tone = best
        tones.add(tone)
        modifiers.add(modifier)
        mod_to_tone[modifier].add(tone)
        tone_to_mod[tone].add(modifier)

    return tones, modifiers, mod_to_tone, tone_to_mod


def format_tone_modifier_mappings(
    phrases: Iterable[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Does: Produce sorted, JSON-ready dicts from mappings with stable keys/values.
    Returns: {"modifiers": {mod: [tones...]}, "tones": {tone: [mods...]}}.
    """
    if not phrases:
        return {"modifiers": {}, "tones": {}}

    tones, modifiers, mod_to_tone, tone_to_mod = build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )

    # Tri des clés et des valeurs pour un rendu totalement déterministe
    return {
        "modifiers": {k: sorted(mod_to_tone[k]) for k in sorted(mod_to_tone)},
        "tones": {k: sorted(tone_to_mod[k]) for k in sorted(tone_to_mod)},
    }


if __name__ == "__main__":
    # Petit exemple manuel (utile en revue rapide ; sans I/O externes)
    _phrases = ["dusty rose", "super-dusty—rose", "dusty lipstick", "soft-beige", "rosy_ nude"]
    _known_tones = {"rose", "beige", "nude"}
    _known_mods = {"dusty", "soft", "rosy"}
    demo = format_tone_modifier_mappings(_phrases, _known_tones, _known_mods)
    # Impression simple pour illustrer l’API ; commenter si non souhaité en prod.
    import json  # local import to keep module footprint minimal
    print(json.dumps(demo, ensure_ascii=False, indent=2))
