from typing import Optional, Set
from color_sentiment_extractor.extraction.color.vocab import all_webcolor_names, known_tones
from color_sentiment_extractor.extraction.color.constants import RECOVER_BASE_OVERRIDES, NON_SUFFIXABLE_MODIFIERS, ED_SUFFIX_ALLOWLIST
from color_sentiment_extractor.extraction.color.suffix import build_y_variant, build_ey_variant, is_cvc_ending
def build_augmented_suffix_vocab(
    known_tokens: set[str],
    known_modifiers: set[str],
    debug: bool = False
) -> set[str]:
    """
    Does: Builds a suffix-augmented vocabulary from known tokens and modifiers.
    - Recovers base form via rules, overrides, or fallback recovery.
    - Generates valid '-y', '-ey', and '-ed' variants including CVC doubling.
    Returns: Set of valid base and suffixed tokens.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
    raw_inputs = known_tokens | known_modifiers
    recovery_vocab = known_modifiers | known_tokens | all_webcolor_names
    augmented = set()

    if debug:
        print(f"\n🔍 Starting suffix vocab build from {len(raw_inputs)} tokens")

    for raw in sorted(raw_inputs):
        if debug: print(f"\n🔁 [PROCESS] Raw token: '{raw}'")

        base = RECOVER_BASE_OVERRIDES.get(raw) or recover_base(raw, use_cache=True)

        # Manual retry if base unresolved and raw is not known
        if base == raw and raw not in known_modifiers and raw not in known_tones:
            fallback_base = None
            if raw.endswith("ey"):
                fallback_base = raw[:-2] + "e"
            elif raw.endswith("y"):
                fallback_base = raw[:-1]
            if fallback_base:
                retry = recover_base(
                    fallback_base,
                    known_modifiers=recovery_vocab,
                    known_tones=recovery_vocab,
                    allow_fuzzy=False,
                    debug=False,
                )
                if retry:
                    if debug: print(f"✅ [MANUAL OVERRIDE] '{raw}' → '{retry}' via fallback")
                    base = retry

        if not base:
            if debug: print(f"⛔ [SKIP] No base recovered for: '{raw}'")
            continue

        if len(base) < 3:
            if debug: print(f"⛔ [SKIP] Base too short: '{base}'")
            continue

        if base in NON_SUFFIXABLE_MODIFIERS:
            if debug: print(f"⛔ [BLOCKED] Non-suffixable base: '{base}'")
            augmented.add(raw)
            continue

        forms = {raw, base}

        # --- -y variant ---
        y_form = build_y_variant(base, debug=False)
        if y_form:
            forms.add(y_form)
        if debug:print(f"[🔍 DEBUG] base = '{base}', fallback_y = '{base + 'y'}'")


        # Always check if "base + y" is a valid known modifier, even if build_y_variant() rejects it
        fallback_y = base + "y"
        if fallback_y in known_modifiers:
            forms.add(fallback_y)
            if debug and fallback_y not in forms:
                print(f"✅ [FORCE-INCLUDE -y] '{base}' → '{fallback_y}' (in known_modifiers)")

        # --- -ey variant: ONLY if allowlisted ---
        ey_form = build_ey_variant(base, raw, debug=False)
        if ey_form and ey_form in known_modifiers:
            forms.add(ey_form)

        # --- -ed variants ---
        if base in ED_SUFFIX_ALLOWLIST:
            ed_form = base + "ed"
            forms.add(ed_form)
            if debug: print(f"✅ [ALLOWLIST -ed] '{base}' → '{ed_form}'")
        elif is_cvc_ending(base):
            ed_form = base + base[-1] + "ed"
            forms.add(ed_form)
            if debug: print(f"✅ [CVC -ed] '{base}' → '{ed_form}'")
        elif base.endswith("y"):
            ed_form = base[:-1] + "ed"
            forms.add(ed_form)
            if debug: print(f"✅ [Y -ed] '{base}' → '{ed_form}'")
        elif base.endswith("e"):
            ed_form = base[:-1] + "ed"
            forms.add(ed_form)
            if debug: print(f"✅ [E -ed] '{base}' → '{ed_form}'")
        else:
            ed_form = base + "ed"
            forms.add(ed_form)
            if debug: print(f"✅ [DEFAULT -ed] '{base}' → '{ed_form}'")

        # --- allowlisted raw ED ---
        if raw in ED_SUFFIX_ALLOWLIST:
            ed_form_raw = raw + "ed"
            forms.add(ed_form_raw)
            if debug: print(f"✅ [ALLOWLIST -ed RAW] '{raw}' → '{ed_form_raw}'")

        # --- CVC + y ---
        if is_cvc_ending(base):
            cvc_y = base + base[-1] + "y"
            forms.add(cvc_y)
            if debug: print(f"✅ [CVC -y] '{base}' → '{cvc_y}'")

        if debug:
            print(f"📦 Final forms for '{raw}': {sorted(forms)}")

        augmented.update(forms)

    if debug:
        print(f"\n🧾 Done. Final augmented vocab ({len(augmented)} items):")
        print(sorted(augmented))

    return augmented


from functools import lru_cache

@lru_cache(maxsize=4096)
def is_suffix_variant(
    token: str,
    known_modifiers: frozenset,
    known_tones: frozenset,
    debug: bool = False,
    allow_fuzzy: bool = False
) -> bool:
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
    """
    Does: Checks whether a token is a '-y' or '-ed' suffix variant of a known modifier or tone.
    Returns: True if the base is known and not blocked, else False.
    """
    if not token.endswith(("y", "ed")):
        if debug: print(f"❌ [SKIP] '{token}' → not a suffix form")
        return False

    # If already known and not an override form, it's not a variant
    if token in known_modifiers or token in known_tones:
        if token not in RECOVER_BASE_OVERRIDES:
            if debug: print(f"❌ [SKIP] '{token}' is already known and not an override")
            return False

    base = recover_base(
        token,
        allow_fuzzy=allow_fuzzy,
        debug=debug,
        known_modifiers=known_modifiers,  # legacy kwargs supported by recover_base
        known_tones=known_tones,  # legacy kwargs supported by recover_base
    )

    is_known = base in known_modifiers or base in known_tones
    is_blocked = base in NON_SUFFIXABLE_MODIFIERS
    valid = is_known and not is_blocked

    if debug:
        print(f"{'✅ VALID' if valid else '❌ INVALID'} | token: '{token}' → base: '{base}' "
              f"| known: {is_known} | blocked: {is_blocked}")

    return valid


# ─────────────────────────────────────────────
# 2. Suffix Recovery Functions
# ─────────────────────────────────────────────

def recover_y(token, known_modifiers, known_tones, debug=False):
    token = token.strip().lower()
    if not token.endswith("y") or len(token) < 3:
        if debug: print(f"[SKIP] '{token}' → not a valid '-y' suffix candidate")
        return None

    if debug:
        print(f"[DEBUG🔍] Token received: '{token}'")
        print(f"[DEBUG🔍] Overrides keys: {list(RECOVER_BASE_OVERRIDES.keys())}")
        print(f"[DEBUG🔍] Does token in overrides? → {token in RECOVER_BASE_OVERRIDES}")

    # Direct override
    if token in RECOVER_BASE_OVERRIDES:
        base = RECOVER_BASE_OVERRIDES[token]
        if debug: print(f"📌 [Y OVERRIDE] '{token}' → '{base}' via RECOVER_BASE_OVERRIDES")
        return base

    base = token[:-1]
    candidates = [base]

    # Add collapsed duplicate consonant (e.g., "fuzzy" → "fuz")
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed = base[:-1]
        candidates.append(collapsed)
        candidates.append(collapsed + "e")

    # Add base + "e" (e.g., "creamy" → "creame")
    candidates.append(base + "e")

    # Add second y-strip (e.g., "glossyy" → "gloss")
    if base.endswith("y") and len(base) > 3:
        candidates.append(base[:-1])

    for candidate in candidates:
        if candidate in known_modifiers or candidate in known_tones:
            if debug: print(f"📌 [MATCHED] '{token}' → '{candidate}'")
            return candidate

    if debug: print(f"[RETURN NONE] recover_y() got no base for '{token}'")
    return None

def recover_ed(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ed") or len(token) <= 3:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ed' suffix candidate")
        return None

    base = token[:-2]
    candidates = [base + "e"]  # e.g., "faded" → "fade"

    # CVC collapse: e.g., "tapped" → "tap"
    if len(base) >= 3 and base[-1] == base[-2]:
        candidates.append(base[:-1])

    # Raw base last (e.g., "muted" → "mut")
    candidates.append(base)

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug: print(f"📌 [MATCHED] '{token}' → '{cand}'")
            return cand

    if debug: print(f"[RETURN NONE] recover_ed() got no base for '{token}'")
    return None

def recover_ing(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ing") or len(token) <= 5:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ing' suffix candidate")
        return None

    base = token[:-3]
    candidates = [base + "e", base]  # e.g., "glowing" → "glow", "glowe"

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug: print(f"📌 [MATCHED] '{token}' → '{cand}'")
            return cand

    if debug: print(f"[RETURN NONE] recover_ing() got no base for '{token}'")
    return None

def recover_ied(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ied") or len(token) <= 3:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ied' suffix candidate")
        return None

    candidate = token[:-3] + "y"  # e.g., "tried" → "try"

    if candidate in known_modifiers or candidate in known_tones:
        if debug: print(f"📌 [IED → Y] '{token}' → '{candidate}'")
        return candidate

    if debug: print(f"[RETURN NONE] recover_ied() got no base for '{token}'")
    return None

def recover_er(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("er") or len(token) <= 2:
        if debug: print(f"[SKIP] '{token}' → not a valid '-er' suffix candidate")
        return None

    candidate = token[:-2]  # e.g., "darker" → "dark"

    if candidate in known_modifiers or candidate in known_tones:
        if debug: print(f"📌 [COMPARATIVE -ER] '{token}' → '{candidate}'")
        return candidate

    if debug: print(f"[RETURN NONE] recover_er() got no base for '{token}'")
    return None
def recover_ier(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ier") or len(token) <= 4:
        if debug: print(f"[SKIP] '{token}' is not a valid '-ier' form")
        return None

    stem = token[:-3]
    if debug: print(f"[TRY IER] '{token}' → stem: '{stem}'")

    # Case 1: direct match to stem
    if stem in known_modifiers or stem in known_tones:
        if debug: print(f"[IER STRIP] '{token}' → '{stem}' (direct match)")
        return stem

    # Case 2: stem + "y" is known (e.g. "trendier" → "trendy")
    if stem and stem[-1] not in "aeiou":
        y_form = stem + "y"
        if y_form in known_modifiers or y_form in known_tones:
            if debug: print(f"[IER → Y MATCH] '{token}' → '{y_form}'")
            return y_form

        # Case 3: override exists for y-form (e.g. "fancier" → "fancy" → override)
        if y_form in RECOVER_BASE_OVERRIDES:
            override = RECOVER_BASE_OVERRIDES[y_form]
            if override in known_modifiers or override in known_tones:
                if debug: print(f"[IER → Y → OVERRIDE] '{token}' → '{override}'")
                return override

    if debug: print(f"[RETURN NONE] recover_ier() got no base for '{token}'")
    return None

def recover_ish(token, known_modifiers, known_tones, debug=False):
    if debug:
        print(f"[💥 ENTERED LIVE recover_ish()] token = '{token}'")

    if "ish" not in token or len(token) <= 4:
        return None

    idx = token.rfind("ish")
    if idx < 2:
        return None

    raw_base = token[:idx]
    base = raw_base.strip("-_. ")

    if base in known_modifiers or base in known_tones:
        if debug: print(f"[✅ ISH DIRECT] '{token}' → '{base}'")
        return base

    collapsed = _collapse_double_consonant(base, known_modifiers, known_tones, debug=False)
    if collapsed:
        if debug: print(f"[✅ ISH COLLAPSED] '{token}' → '{collapsed}'")
        return collapsed

    # NEW: Try base + "y" (e.g. "ivor" → "ivory")
    extended_y = base + "y"
    if extended_y in known_modifiers or extended_y in known_tones:
        if debug: print(f"[✅ ISH +Y] '{token}' → '{extended_y}'")
        return extended_y

    # Try base + "e" (e.g. "whit" → "white")
    extended_e = base + "e"
    if extended_e in known_modifiers or extended_e in known_tones:
        if debug: print(f"[✅ ISH +E] '{token}' → '{extended_e}'")
        return extended_e

    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
    recovered = recover_base(
        base,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        fuzzy_fallback=True,
        fuzzy_threshold=75,
        use_cache=False,
        debug=debug,
        depth=1
    )
    if debug: print(f"[🧪 MATCH TEST] recover_base('{base}') → '{recovered}'")

    if recovered:
        if debug: print(f"[🔁 ISH CHAINED] '{token}' → '{base}' → '{recovered}'")
        return recovered

    return None
def recover_ness(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ness") or len(token) <= 5:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ness' suffix candidate")
        return None

    base = token[:-4]  # e.g., "softness" → "soft"
    if debug:
        print(f"[DEBUG🌀] recover_ness('{token}')")
        print(f"[STRIP NESS] → '{base}'")

    # Case 1: ends with "i" → try collapsing (e.g., "happiness" → "happy")
    if base.endswith("i"):
        collapsed = base[:-1]
        if collapsed in known_modifiers or collapsed in known_tones:
            if debug: print(f"📌 [NESS → Y COLLAPSE] '{token}' → '{collapsed}'")
            return collapsed

    # Case 2: try base directly
    if base in known_modifiers or base in known_tones:
        if debug: print(f"📌 [NESS STRIP] '{token}' → '{base}'")
        return base

    if debug: print(f"[RETURN NONE] recover_ness() got no base for '{token}'")
    return None

def recover_ly(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ly") or len(token) <= 3:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ly' suffix candidate")
        return None

    # Handle special case: "ally" → "ic" (e.g., "emotionally" → "emot(ic)")
    if token.endswith("ally"):
        candidate = token[:-4] + "ic"
    else:
        candidate = token[:-2]

    if debug:
        print(f"[DEBUG🌀] recover_ly('{token}')")
        print(f"[CANDIDATE] '{token}' → '{candidate}'")

    if candidate in known_modifiers or candidate in known_tones:
        if debug: print(f"📌 [ADVERB -LY] '{token}' → '{candidate}'")
        return candidate

    if debug: print(f"[RETURN NONE] recover_ly() got no base for '{token}'")
    return None


def recover_en(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("en") or len(token) <= 3:
        if debug: print(f"[SKIP] '{token}' → not a valid '-en' suffix candidate")
        return None

    candidate = token[:-2]  # e.g., "golden" → "gold"

    if debug:
        print(f"[DEBUG🌀] recover_en('{token}')")
        print(f"[CANDIDATE] '{token}' → '{candidate}'")

    if candidate in known_modifiers or candidate in known_tones:
        if debug: print(f"📌 [EN SUFFIX] '{token}' → '{candidate}'")
        return candidate

    if debug: print(f"[RETURN NONE] recover_en() got no base for '{token}'")
    return None


def recover_ey(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ey") or len(token) <= 4:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ey' suffix candidate")
        return None

    base = token[:-2]
    candidate = base + "e"

    if debug:
        print(f"[DEBUG🌀] recover_ey('{token}')")
        print(f"[CHECK] '{token}' → base: '{base}', candidate: '{candidate}'")

    # Case 1: try base + "e" (e.g., "bronzey" → "bronze")
    if candidate in known_modifiers or candidate in known_tones:
        if debug: print(f"📌 [EY +E RESTORE] '{token}' → '{candidate}'")
        return candidate

    # Case 2: check allowlist (e.g., "beigey" → "beige")
    if base in Y_SUFFIX_ALLOWLIST:
        if debug: print(f"📌 [EY ALLOWLIST] '{token}' → '{base}'")
        return base

    if debug: print(f"⛔ [BLOCKED] '-ey' not recoverable: '{base}'")
    return None

def _collapse_double_consonant(base: str, known_modifiers: set, known_tones: set, debug: bool = False) -> Optional[str]:
    """
    If base ends with a double consonant (e.g. 'redd'), and it's invalid,
    try collapsing to a single consonant (e.g. 'red') and return if valid.
    """
    if len(base) < 2 or base[-1] != base[-2]:
        return None  # not double letter

    collapsed = base[:-1]
    if collapsed in known_modifiers or collapsed in known_tones:
        if debug:
            print(f"[🔁 COLLAPSE DOUBLE] '{base}' → '{collapsed}'")
        return collapsed

    return None

def recover_ee_to_y(token, known_modifiers, known_tones, debug=False):
    if not token.endswith("ee") or len(token) < 4:
        if debug: print(f"[SKIP] '{token}' → not a valid '-ee' suffix candidate")
        return None

    base = token[:-2] + "y"  # e.g., "ivoree" → "ivory"

    if debug:
        print(f"[DEBUG🌀] recover_ee_to_y('{token}')")
        print(f"[CANDIDATE] '{token}' → '{base}'")

    if base in known_modifiers or base in known_tones:
        if debug: print(f"📌 [EE→Y] '{token}' → '{base}'")
        return base

    if debug: print(f"[RETURN NONE] recover_ee_to_y() got no base for '{token}'")
    return None
