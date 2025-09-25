from __future__ import annotations

from functools import lru_cache
from typing import Optional, Set

from color_sentiment_extractor.extraction.color.constants import (
    Y_SUFFIX_ALLOWLIST,
    Y_SUFFIX_OVERRIDE_FORMS,
    RECOVER_BASE_OVERRIDES,
)

# Consonant endings that commonly accept -y
ALLOW_ENDS = (
    "sh", "ch", "ss", "m", "n", "r", "l",
    "t", "d", "k", "p", "b", "f", "v",
    "s", "z", "c", "g", "h", "j",
)


def is_y_suffix_allowed(base: str) -> bool:
    """
    Determine if a base token can take the '-y' suffix.

    Rules:
      - Allow if in Y_SUFFIX_ALLOWLIST
      - Block if short (<3) or ends with 'y'/'e' or a vowel
      - Allow typical soft/consonant endings (ALLOW_ENDS)
    """
    if base in Y_SUFFIX_ALLOWLIST:
        return True
    if len(base) < 3:
        return False
    if base.endswith(("y", "e")):
        return False
    if base[-1] in "aeiou":
        return False
    return base.endswith(ALLOW_ENDS)


def is_cvc_ending(base: str) -> bool:
    """
    Return True if base ends with a consonant–vowel–consonant pattern (CVC),
    excluding final 'w','x','y' (e.g., 'dew','tax','sky').
    """
    if len(base) < 3:
        return False
    c1, v, c2 = base[-3], base[-2], base[-1]
    if not (c1.isalpha() and v.isalpha() and c2.isalpha()):
        return False
    if c1 in "aeiou" or v not in "aeiou" or c2 in "aeiouwxy":
        return False
    return True


def build_y_variant(base: str, debug: bool = False) -> Optional[str]:
    """
    Build a valid '-y' form from a base.
      1) override table (e.g., 'rose'→'rosy')
      2) allowlist
      3) rule-based allow
    Returns: suffixed token or None.
    """
    if base in Y_SUFFIX_OVERRIDE_FORMS:
        return Y_SUFFIX_OVERRIDE_FORMS[base]
    if base in Y_SUFFIX_ALLOWLIST:
        return base + "y"
    if is_y_suffix_allowed(base):
        return base + "y"
    return None


def build_ey_variant(base: str, raw: str, debug: bool = False) -> Optional[str]:
    """
    Build a valid '-ey' variant from a base.
      - Accept if base/raw is allowlisted.
      - Also allow sibilant endings (ge/ce/ze/se) by dropping final 'e'.
    """
    def _ey(stem: str) -> str:
        return (stem[:-1] if stem.endswith("e") else stem) + "ey"

    if raw in Y_SUFFIX_ALLOWLIST or base in Y_SUFFIX_ALLOWLIST:
        return _ey(base)
    if base.endswith(("ge", "ce", "ze", "se")) and len(base) > 2:
        return _ey(base)
    return None


def _default_suffix_strip(token: str) -> str:
    """Strip trailing '-y' if present."""
    return token[:-1] if token.endswith("y") else token


@lru_cache(maxsize=1)
def _stripped_override_map() -> dict[str, str]:
    """
    Build a map once: stripped override token (without trailing 'y'/'ed') → override base.
    Lazy to avoid work at import if table is large.
    """
    out: dict[str, str] = {}
    for k, v in RECOVER_BASE_OVERRIDES.items():
        if k.endswith("y"):
            stripped = k[:-1]
        elif k.endswith("ed"):
            stripped = k[:-2]
        else:
            stripped = k
        out[stripped] = v
    return out


def _apply_reverse_override(base: str, token: str, debug: bool = False) -> str:
    """
    Apply reverse override by matching a stripped override token to the given base.
    Recognizes 'y' and 'ed' suffix forms. If match found, return override base.
    """
    m = _stripped_override_map()
    return m.get(base, base)


def _collapse_repeated_consonant(
    base: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> str:
    """
    Remove a final doubled consonant (e.g., 'blurr' → 'blur') if the collapsed form
    exists in known sets; otherwise keep original.
    """
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed = base[:-1]
        if collapsed in known_modifiers or collapsed in known_tones:
            return collapsed
    return base
