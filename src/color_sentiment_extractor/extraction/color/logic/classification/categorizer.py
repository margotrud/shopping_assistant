"""
categorizer.py

Does:
    Build strict modifier↔tone mappings from phrases (spaces/hyphens/glued),
    using strict base recovery for modifiers and filtering cosmetic nouns from tones.
Returns:
    build_tone_modifier_mappings() → (tones, modifiers, mod_to_tone, tone_to_mod);
    format_tone_modifier_mappings() → {"modifiers": {...}, "tones": {...}}.
"""

from __future__ import annotations

# ── Imports & Public API ──────────────────────────────────────────────────────
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

# Import via stable façades (keeps caller deps clean)
from color_sentiment_extractor.extraction.general.token import (
    recover_base,
    normalize_token,
)
from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS

__all__ = ["build_tone_modifier_mappings", "format_tone_modifier_mappings"]

# ── Constants ────────────────────────────────────────────────────────────────
# Common hyphen variants seen in heterogeneous sources
UNICODE_HYPHENS: Tuple[str, ...] = ("-", "–", "—", "‒", "−")
_HYPHEN_TRANS = str.maketrans({h: "-" for h in UNICODE_HYPHENS})

# ── Helpers (private) ────────────────────────────────────────────────────────
def _norm_token(t: str) -> str:
    """Lowercase + trim; keep hyphens (meaningful for glued tokens)."""
    return normalize_token(t or "", keep_hyphens=True)


def _norm_set(values: Iterable[str]) -> Set[str]:
    """Normalize all items (lower/trim, hyphen-safe)."""
    return {_norm_token(v) for v in values if v}


def _split_into_candidates(phrase: str) -> List[Tuple[str, str]]:
    """
    Produce (modifier, tone) bigram candidates from a phrase.
    Accepts spaces, hyphens (incl. Unicode), underscores, and glued forms.
    """
    p = _norm_token(phrase)
    if not p:
        return []

    # Unify separators (all hyphens → '-', underscores → '-')
    p = p.translate(_HYPHEN_TRANS).replace("_", "-")

    # No spaces: split on hyphens and slide window of size 2
    if " " not in p:
        parts = [s for s in p.split("-") if s]
        return [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)] if len(parts) >= 2 else []

    # With spaces: token bigrams
    tokens = [t for t in p.split() if t]
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


# ── Core API (public) ────────────────────────────────────────────────────────
def build_tone_modifier_mappings(
    phrases: Iterable[str],
    known_tones: Set[str],
    known_modifiers: Set[str],
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Does:
        Derive bidirectional mappings (modifier↔tone) using strict base recovery
        for modifiers; excludes cosmetic nouns as tones.
    Returns:
        (tones, modifiers, mod_to_tone, tone_to_mod).
    """
    known_tones = _norm_set(known_tones)
    known_modifiers = _norm_set(known_modifiers)
    cosmetic_nouns = _norm_set(COSMETIC_NOUNS)

    tones: Set[str] = set()
    modifiers: Set[str] = set()
    mod_to_tone: Dict[str, Set[str]] = defaultdict(set)
    tone_to_mod: Dict[str, Set[str]] = defaultdict(set)

    for phrase in phrases or []:
        candidates = _split_into_candidates(phrase)
        if not candidates:
            continue

        best: Tuple[str, str] | None = None

        for mod_raw, tone in candidates:
            # Fast tone screening
            if tone in cosmetic_nouns or tone not in known_tones:
                continue

            # Strict base recovery (no fuzzy fallback)
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
            candidate = direct or via_base
            if candidate is not None:
                best = (candidate, tone)
                if direct is not None:
                    break  # Prefer exact modifier hit

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
    Does:
        Convert mappings to sorted, JSON-ready dicts with stable keys/values.
    Returns:
        {"modifiers": {mod: [tones...]}, "tones": {tone: [mods...]}}.
    """
    if not phrases:
        return {"modifiers": {}, "tones": {}}

    _, _, mod_to_tone, tone_to_mod = build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )

    return {
        "modifiers": {k: sorted(v) for k, v in sorted(mod_to_tone.items())},
        "tones": {k: sorted(v) for k, v in sorted(tone_to_mod.items())},
    }


# ── Demo (optional) ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    _phrases = ["dusty rose", "super-dusty—rose", "dusty lipstick", "soft-beige", "rosy_ nude"]
    _known_tones = {"rose", "beige", "nude"}
    _known_mods = {"dusty", "soft", "rosy"}

    import json

    demo = format_tone_modifier_mappings(_phrases, _known_tones, _known_mods)
    print(json.dumps(demo, ensure_ascii=False, indent=2))
