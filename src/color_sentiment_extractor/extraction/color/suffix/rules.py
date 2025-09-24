from typing import Set

from color_sentiment_extractor.extraction.color.constants import (
    Y_SUFFIX_ALLOWLIST,
    Y_SUFFIX_OVERRIDE_FORMS,
    RECOVER_BASE_OVERRIDES,
)


def is_y_suffix_allowed(base: str) -> bool:
    """
    Does: Determines if a base token can take the '-y' suffix by rule-based conditions.
    - Allows if in Y_SUFFIX_ALLOWLIST.
    - Blocks short words or tokens ending in 'y', 'e', or a vowel.
    - Allows soft or consonant endings likely to produce valid '-y' forms.
    """
    if base in Y_SUFFIX_ALLOWLIST:
        return True
    if len(base) < 3:
        return False
    if base.endswith(("y", "e")):
        return False
    if base[-1] in "aeiou":
        return False

    ALLOW_ENDS = (
        "sh", "ch", "ss", "m", "n", "r", "l",
        "t", "d", "k", "p", "b", "f", "v",
        "s", "z", "c", "g", "h", "j"
    )
    if base.endswith(ALLOW_ENDS):
        return True
    return False


def is_cvc_ending(base: str) -> bool:
    """
    Does: Returns True if `base` ends in a consonant-vowel-consonant (CVC) pattern,
    allowing consonant doubling (e.g., 'blur' → 'blurry').
    - Blocks final consonants 'w', 'x', 'y' (e.g., 'dew', 'tax', 'sky').
    - Requires last 3 letters to follow CVC structure.
    """
    if len(base) < 3:
        return False

    c1, v, c2 = base[-3], base[-2], base[-1]

    return (
        c1.isalpha() and c2.isalpha()
        and c1 not in "aeiou"
        and v in "aeiou"
        and c2 not in "aeiou"
        and c2 not in "wxy"
    )


def build_y_variant(base: str, debug: bool = False) -> str | None:
    """
    Does: Builds a valid '-y' form from a base token if allowed.
    - Uses override forms first (e.g., 'rose' → 'rosy').
    - Then checks allowlist and suffix eligibility rules.
    Returns: Suffixed token or None if not allowed.
    """
    result = None

    if base in Y_SUFFIX_OVERRIDE_FORMS:
        result = Y_SUFFIX_OVERRIDE_FORMS[base]
        label = "OVERRIDE -y"
    elif base in Y_SUFFIX_ALLOWLIST:
        result = base + "y"
        label = "ALLOWLIST -y FIXED"
    elif is_y_suffix_allowed(base):
        result = base + "y"
        label = "RULED -y"
    else:
        if debug:
            print(f"⛔ [NO -y] Not allowed for '{base}'")
        return None

    if debug:
        print(f"✅ [{label}] '{base}' → '{result}'")
    return result


def build_ey_variant(base: str, raw: str, debug: bool = False) -> str | None:
    """
    Does: Builds a valid '-ey' variant from a base token.
    - Accepts if base or raw is in allowlist.
    - Allows for sibilant endings (ge, ce, ze, se) by dropping final 'e'.
    Returns: Suffixed token or None if not allowed.
    """

    def _ey(stem: str) -> str:
        return (stem[:-1] if stem.endswith("e") else stem) + "ey"

    if raw in Y_SUFFIX_ALLOWLIST or base in Y_SUFFIX_ALLOWLIST:
        result, label = _ey(base), "ALLOWLIST -ey"
    elif base.endswith(("ge", "ce", "ze", "se")) and len(base) > 2:
        result, label = _ey(base), "RULED -ey (sibilant)"
    else:
        if debug:
            print(f"⛔ [NO -ey] Not allowed for '{base}'")
        return None

    if debug:
        print(f"✅ [{label}] '{base}' → '{result}'")
    return result


def _default_suffix_strip(token: str) -> str:
    """
    Does: Strips the default '-y' suffix if present.
    Returns: Base token with '-y' removed, or original if no '-y'.
    """
    return token[:-1] if token.endswith("y") else token


STRIPPED_OVERRIDE_MAP = {
    (k[:-1] if k.endswith("y") else k[:-2] if k.endswith("ed") else k): v
    for k, v in RECOVER_BASE_OVERRIDES.items()
}


def _apply_reverse_override(base: str, token: str, debug: bool = False) -> str:
    """
    Does: Applies reverse override by matching a stripped override token to the given base.
    - Recognizes 'y' and 'ed' suffix forms in override_token.
    - Returns override_base if match found, else returns original base.
    """
    if base in STRIPPED_OVERRIDE_MAP:
        if debug:
            print(f"📌 [REVERSE OVERRIDE] '{token}' → '{STRIPPED_OVERRIDE_MAP[base]}'")
        return STRIPPED_OVERRIDE_MAP[base]
    return base


def _collapse_repeated_consonant(
    base: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False
) -> str:
    """
    Does: Removes final repeated consonant (e.g., 'blurr' → 'blur') if the result is valid.
    Returns: Collapsed base if it exists in known sets, else returns original.
    """
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed_base = base[:-1]
        if collapsed_base in known_modifiers or collapsed_base in known_tones:
            if debug:
                print(f"📌 [COLLAPSE] Repeated final consonant → '{collapsed_base}'")
            return collapsed_base

    return base
