# color/recovery/fuzzy_recovery.py

import warnings
from typing import Optional
from extraction.color.constants import SEMANTIC_CONFLICTS
from extraction.color.vocab import known_tones as KNOWN_TONES
from extraction.general.fuzzy.scoring import rhyming_conflict
from extraction.general.token.base_recovery import recover_base
from extraction.general.utils.load_config import load_config

KNOWN_MODIFIERS = load_config("known_modifiers", mode="set")


def is_suffix_root_match(alias: str, token: str, debug: bool = False) -> bool:
    """
    Does: Checks if alias and token are suffix/root variants from valid suffix recovery.
    Returns: True only if both recover to same known base and a transformation actually occurred.
    """
    base_token = recover_base(token, use_cache=True)
    base_alias = recover_base(alias, use_cache=True)

    if debug:
        print(f"[🔁 SUFFIX ROOT CHECK] token='{token}' → '{base_token}' | alias='{alias}' → '{base_alias}'")

    # exiger une vraie transformation
    if token == base_token and alias == base_alias:
        if debug:
            print(f"[🚫 NO TRANSFORMATION] Both inputs unchanged → Rejecting '{alias}' vs '{token}'")
        return False

    if base_token and base_token == base_alias:
        if (base_token in KNOWN_MODIFIERS) or (base_token in KNOWN_TONES):
            if frozenset({alias, token}) not in SEMANTIC_CONFLICTS and not rhyming_conflict(alias, token):
                if debug:
                    print(f"[✅ ROOT MATCH] via base '{base_token}'")
                return True

    if debug:
        print(f"[❌ SUFFIX MATCH FAIL] '{alias}' vs '{token}'")
    return False


