# token/suffix/recovery.py

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Set

from color_sentiment_extractor.extraction.color.constants import (
    RECOVER_BASE_OVERRIDES,
    NON_SUFFIXABLE_MODIFIERS,
    ED_SUFFIX_ALLOWLIST,
    Y_SUFFIX_ALLOWLIST,
)
from color_sentiment_extractor.extraction.color.suffix import (
    build_y_variant,
    build_ey_variant,
    is_cvc_ending,
)


# ─────────────────────────────────────────────
# 1) Augmented Suffix Vocabulary Builder
# ─────────────────────────────────────────────

def build_augmented_suffix_vocab(
    known_tokens: set[str],
    known_modifiers: set[str],
    known_tones: set[str] | None = None,
    webcolor_names: set[str] | None = None,
    debug: bool = False,
) -> set[str]:
    """
    Does: Build a suffix-augmented vocab from known tokens/modifiers(/tones).
    - Recovers base via rules/overrides; generates valid -y/-ey/-ed (CVC, y→ied, e-drop).
    - Avoids default -ed; restricts -ey to allowlist/known.
    Returns: Set of valid base and suffixed tokens.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    known_tones = known_tones or set()
    webcolor_names = webcolor_names or set()

    # Inclure les tones dans la graine pour permettre p.ex. rose→rosy
    raw_inputs = known_tokens | known_modifiers | known_tones
    # Vocab de référence pour les récupérations “fallback”
    recovery_vocab = known_modifiers | known_tokens | known_tones | webcolor_names

    augmented: set[str] = set()

    if debug:
        print(f"\n🔍 Starting suffix vocab build from {len(raw_inputs)} tokens")

    for raw in sorted(raw_inputs):
        if debug:
            print(f"\n🔁 [PROCESS] Raw token: '{raw}'")

        # 1) Base via override ou recover_base
        base = RECOVER_BASE_OVERRIDES.get(raw) or recover_base(raw, use_cache=True)

        # 1.b) Fallback manuel si “base” non résolue et raw non connu
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
                    if debug:
                        print(f"✅ [MANUAL OVERRIDE] '{raw}' → '{retry}' via fallback")
                    base = retry

        if not base:
            if debug:
                print(f"⛔ [SKIP] No base recovered for: '{raw}'")
            continue

        if len(base) < 3:
            if debug:
                print(f"⛔ [SKIP] Base too short: '{base}'")
            continue

        forms = {raw, base}

        # Si la base est explicitement non suffixable, on garde au moins raw
        if base in NON_SUFFIXABLE_MODIFIERS:
            if debug:
                print(f"⛔ [BLOCKED] Non-suffixable base: '{base}'")
            augmented.update(forms)
            continue

        # 2) Génération -y
        y_form = build_y_variant(base, debug=False)
        if y_form:
            forms.add(y_form)

        # Debug helper
        if debug:
            print(f"[🔍 DEBUG] base = '{base}', fallback_y = '{base + 'y'}'")

        # Force-inclusion “base+y” si déjà présent dans known_modifiers (et pas déjà généré)
        fallback_y = base + "y"
        if fallback_y in known_modifiers and fallback_y not in forms:
            forms.add(fallback_y)
            if debug:
                print(f"✅ [FORCE-INCLUDE -y] '{base}' → '{fallback_y}'")

        # 3) Génération -ey STRICTE: seulement si build_ey_variant + présent dans known_modifiers
        ey_form = build_ey_variant(base, raw, debug=False)
        if ey_form and ey_form in known_modifiers:
            forms.add(ey_form)

        # 4) Génération -ed (uniquement modifiers ; pas de DEFAULT)
        if base in known_modifiers:
            if base in ED_SUFFIX_ALLOWLIST:
                ed_form = base + "ed"
                forms.add(ed_form)
                if debug:
                    print(f"✅ [ALLOWLIST -ed] '{base}' → '{ed_form}'")
            elif is_cvc_ending(base):
                ed_form = base + base[-1] + "ed"
                forms.add(ed_form)
                if debug:
                    print(f"✅ [CVC -ed] '{base}' → '{ed_form}'")
            elif base.endswith("y"):
                ed_form = base[:-1] + "ed"  # flashy → flashed
                forms.add(ed_form)
                if debug:
                    print(f"✅ [Y -ed] '{base}' → '{ed_form}'")
            elif base.endswith("e"):
                ed_form = base[:-1] + "ed"  # pale → paled
                forms.add(ed_form)
                if debug:
                    print(f"✅ [E -ed] '{base}' → '{ed_form}'")

        # 5) Allowlist “raw+ed” au besoin
        if raw in ED_SUFFIX_ALLOWLIST:
            ed_form_raw = raw + "ed"
            forms.add(ed_form_raw)
            if debug:
                print(f"✅ [ALLOWLIST -ed RAW] '{raw}' → '{ed_form_raw}'")

        # 6) CVC + y (p.ex. “mat” → “matty” si la règle CVC est vraie)
        if is_cvc_ending(base):
            cvc_y = base + base[-1] + "y"
            forms.add(cvc_y)
            if debug:
                print(f"✅ [CVC -y] '{base}' → '{cvc_y}'")

        if debug:
            print(f"📦 Final forms for '{raw}': {sorted(forms)}")

        augmented.update(forms)

    if debug:
        print(f"\n🧾 Done. Final augmented vocab ({len(augmented)} items):")
        print(sorted(augmented))

    return augmented


# ─────────────────────────────────────────────
# 2) Variant Predicate
# ─────────────────────────────────────────────

@lru_cache(maxsize=4096)
def is_suffix_variant(
    token: str,
    known_modifiers: frozenset[str],
    known_tones: frozenset[str],
    debug: bool = False,
    allow_fuzzy: bool = False,
) -> bool:
    """
    Does: Tell if token is a -y/-ey/-ed variant whose base is a known modifier/tone.
    - Uses recover_base with optional fuzzy; blocks NON_SUFFIXABLE_MODIFIERS.
    Returns: True if base is known and not blocked.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    if not token.endswith(("y", "ey", "ed")):
        if debug:
            print(f"❌ [SKIP] '{token}' → not a handled suffix form")
        return False

    # Si déjà connu (et pas une forme override), ce n'est pas un *variant*.
    if (token in known_modifiers or token in known_tones) and token not in RECOVER_BASE_OVERRIDES:
        if debug:
            print(f"❌ [SKIP] '{token}' is already known and not an override")
        return False

    base = recover_base(
        token,
        allow_fuzzy=allow_fuzzy,
        debug=debug,
        known_modifiers=known_modifiers,  # kwargs legacy support
        known_tones=known_tones,          # kwargs legacy support
    )

    is_known_base = base in known_modifiers or base in known_tones
    is_blocked = base in NON_SUFFIXABLE_MODIFIERS
    valid = bool(base) and is_known_base and not is_blocked

    if debug:
        print(
            f"{'✅ VALID' if valid else '❌ INVALID'} | token: '{token}' → base: '{base}' "
            f"| known_base: {is_known_base} | blocked: {is_blocked}"
        )

    return valid


# ─────────────────────────────────────────────
# 3) Suffix Recovery Functions
# ─────────────────────────────────────────────

def recover_y(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -y forms (creamy→cream/e, dusty→dust, shiny→shine).
    - Tries base, base+e, double-consonant collapse, second y-strip.
    Returns: Base token if found, else None.
    """
    token = token.strip().lower()
    if not token.endswith("y") or len(token) < 3:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-y' suffix candidate")
        return None

    if debug:
        print(f"[DEBUG🔍] Token received: '{token}'")
        print(f"[DEBUG🔍] Overrides keys: {list(RECOVER_BASE_OVERRIDES.keys())}")
        print(f"[DEBUG🔍] Does token in overrides? → {token in RECOVER_BASE_OVERRIDES}")

    # 1) Override direct
    if token in RECOVER_BASE_OVERRIDES:
        base = RECOVER_BASE_OVERRIDES[token]
        if debug:
            print(f"📌 [Y OVERRIDE] '{token}' → '{base}' via RECOVER_BASE_OVERRIDES")
        return base

    base = token[:-1]
    candidates: list[str] = [base]

    # Heuristique courant pour couleurs: rosy→rose, shiny→shine (base+e est déjà couvert, on le met tôt)
    candidates.append(base + "e")

    # Duplicate consonant collapse (fuzzy→fuz / gloss y edge-cases)
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed = base[:-1]
        candidates.append(collapsed)
        candidates.append(collapsed + "e")

    # second y-strip: “glossyy” → “gloss”
    if base.endswith("y") and len(base) > 3:
        candidates.append(base[:-1])

    for candidate in candidates:
        if candidate in known_modifiers or candidate in known_tones:
            if debug:
                print(f"📌 [MATCHED] '{token}' → '{candidate}'")
            return candidate

    if debug:
        print(f"[RETURN NONE] recover_y() got no base for '{token}'")
    return None


def recover_ed(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ed forms (muted→mut(e), paled→pale, tapped→tap).
    - Tries e-restore, CVC collapse, raw base.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ed") or len(token) <= 3:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ed' suffix candidate")
        return None

    base = token[:-2]
    candidates = [base + "e"]  # faded→fade

    # CVC collapse: tapped→tap
    if len(base) >= 3 and base[-1] == base[-2]:
        candidates.append(base[:-1])

    # raw base last: muted→mut
    candidates.append(base)

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug:
                print(f"📌 [MATCHED] '{token}' → '{cand}'")
            return cand

    if debug:
        print(f"[RETURN NONE] recover_ed() got no base for '{token}'")
    return None


def recover_ing(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ing forms (glowing→glow/e).
    - Tries base+e then base.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ing") or len(token) <= 5:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ing' suffix candidate")
        return None

    base = token[:-3]
    candidates = [base + "e", base]

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug:
                print(f"📌 [MATCHED] '{token}' → '{cand}'")
            return cand

    if debug:
        print(f"[RETURN NONE] recover_ing() got no base for '{token}'")
    return None


def recover_ied(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover -ied → -y (tried→try).
    - Checks the y-form in known sets.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ied") or len(token) <= 3:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ied' suffix candidate")
        return None

    candidate = token[:-3] + "y"

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            print(f"📌 [IED → Y] '{token}' → '{candidate}'")
        return candidate

    if debug:
        print(f"[RETURN NONE] recover_ied() got no base for '{token}'")
    return None


def recover_er(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover comparative -er (darker→dark).
    - Strips er and checks direct base.
    Returns: Base token if found, else None.
    """
    if not token.endswith("er") or len(token) <= 2:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-er' suffix candidate")
        return None

    candidate = token[:-2]

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            print(f"📌 [COMPARATIVE -ER] '{token}' → '{candidate}'")
        return candidate

    if debug:
        print(f"[RETURN NONE] recover_er() got no base for '{token}'")
    return None


def recover_ier(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover comparative -ier (trendier→trendy / fancy→fancier).
    - Tries stem, stem+y, and overrides on y-form.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ier") or len(token) <= 4:
        if debug:
            print(f"[SKIP] '{token}' is not a valid '-ier' form")
        return None

    stem = token[:-3]
    if debug:
        print(f"[TRY IER] '{token}' → stem: '{stem}'")

    if stem in known_modifiers or stem in known_tones:
        if debug:
            print(f"[IER STRIP] '{token}' → '{stem}' (direct match)")
        return stem

    if stem and stem[-1] not in "aeiou":
        y_form = stem + "y"
        if y_form in known_modifiers or y_form in known_tones:
            if debug:
                print(f"[IER → Y MATCH] '{token}' → '{y_form}'")
            return y_form

        if y_form in RECOVER_BASE_OVERRIDES:
            override = RECOVER_BASE_OVERRIDES[y_form]
            if override in known_modifiers or override in known_tones:
                if debug:
                    print(f"[IER → Y → OVERRIDE] '{token}' → '{override}'")
                return override

    if debug:
        print(f"[RETURN NONE] recover_ier() got no base for '{token}'")
    return None


def recover_ish(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ish forms (greenish→green, ivorish→ivory/ivor+e).
    - Tries direct base, collapse double consonant, +y/+e, then recover_base(base).
    Returns: Base token if found, else None.
    """
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
        if debug:
            print(f"[✅ ISH DIRECT] '{token}' → '{base}'")
        return base

    collapsed = _collapse_double_consonant(base, known_modifiers, known_tones, debug=False)
    if collapsed:
        if debug:
            print(f"[✅ ISH COLLAPSED] '{token}' → '{collapsed}'")
        return collapsed

    extended_y = base + "y"
    if extended_y in known_modifiers or extended_y in known_tones:
        if debug:
            print(f"[✅ ISH +Y] '{token}' → '{extended_y}'")
        return extended_y

    extended_e = base + "e"
    if extended_e in known_modifiers or extended_e in known_tones:
        if debug:
            print(f"[✅ ISH +E] '{token}' → '{extended_e}'")
        return extended_e

    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base as _recover_base
    recovered = _recover_base(
        base,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        fuzzy_fallback=True,
        fuzzy_threshold=75,
        use_cache=False,
        debug=debug,
        depth=1,
    )
    if debug:
        print(f"[🧪 MATCH TEST] recover_base('{base}') → '{recovered}'")

    if recovered:
        if debug:
            print(f"[🔁 ISH CHAINED] '{token}' → '{base}' → '{recovered}'")
        return recovered

    return None


def recover_ness(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ness (softness→soft, creaminess→creamy→cream).
    - Handles trailing i→y collapse; checks base directly.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ness") or len(token) <= 5:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ness' suffix candidate")
        return None

    base = token[:-4]

    if debug:
        print(f"[DEBUG🌀] recover_ness('{token}')")
        print(f"[STRIP NESS] → '{base}'")

    if base.endswith("i"):
        collapsed = base[:-1]
        if collapsed in known_modifiers or collapsed in known_tones:
            if debug:
                print(f"📌 [NESS → Y COLLAPSE] '{token}' → '{collapsed}'")
            return collapsed

    if base in known_modifiers or base in known_tones:
        if debug:
            print(f"📌 [NESS STRIP] '{token}' → '{base}'")
        return base

    if debug:
        print(f"[RETURN NONE] recover_ness() got no base for '{token}'")
    return None


def recover_ly(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ly adverbs (softly→soft, warmly→warm).
    - Special-case “ally”→“ic”, else strip -ly.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ly") or len(token) <= 3:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ly' suffix candidate")
        return None

    if token.endswith("ally"):
        candidate = token[:-4] + "ic"
    else:
        candidate = token[:-2]

    if debug:
        print(f"[DEBUG🌀] recover_ly('{token}')")
        print(f"[CANDIDATE] '{token}' → '{candidate}'")

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            print(f"📌 [ADVERB -LY] '{token}' → '{candidate}'")
        return candidate

    if debug:
        print(f"[RETURN NONE] recover_ly() got no base for '{token}'")
    return None


def recover_en(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -en adjectives (golden→gold).
    - Strips -en and checks direct base.
    Returns: Base token if found, else None.
    """
    if not token.endswith("en") or len(token) <= 3:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-en' suffix candidate")
        return None

    candidate = token[:-2]

    if debug:
        print(f"[DEBUG🌀] recover_en('{token}')")
        print(f"[CANDIDATE] '{token}' → '{candidate}'")

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            print(f"📌 [EN SUFFIX] '{token}' → '{candidate}'")
        return candidate

    if debug:
        print(f"[RETURN NONE] recover_en() got no base for '{token}'")
    return None


def recover_ey(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover base from -ey forms (bronzey→bronze, beigey→beige).
    - Prefers base+e; allowlist fallback for specific roots.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ey") or len(token) <= 4:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ey' suffix candidate")
        return None

    base = token[:-2]
    candidate = base + "e"

    if debug:
        print(f"[DEBUG🌀] recover_ey('{token}')")
        print(f"[CHECK] '{token}' → base: '{base}', candidate: '{candidate}'")

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            print(f"📌 [EY +E RESTORE] '{token}' → '{candidate}'")
        return candidate

    if base in Y_SUFFIX_ALLOWLIST:
        if debug:
            print(f"📌 [EY ALLOWLIST] '{token}' → '{base}'")
        return base

    if debug:
        print(f"⛔ [BLOCKED] '-ey' not recoverable: '{base}'")
    return None


def recover_ee_to_y(
    token: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Recover -ee→-y (ivoree→ivory).
    - Builds candidate base with trailing y.
    Returns: Base token if found, else None.
    """
    if not token.endswith("ee") or len(token) < 4:
        if debug:
            print(f"[SKIP] '{token}' → not a valid '-ee' suffix candidate")
        return None

    base = token[:-2] + "y"

    if debug:
        print(f"[DEBUG🌀] recover_ee_to_y('{token}')")
        print(f"[CANDIDATE] '{token}' → '{base}'")

    if base in known_modifiers or base in known_tones:
        if debug:
            print(f"📌 [EE→Y] '{token}' → '{base}'")
        return base

    if debug:
        print(f"[RETURN NONE] recover_ee_to_y() got no base for '{token}'")
    return None


# ─────────────────────────────────────────────
# 4) Helpers
# ─────────────────────────────────────────────

def _collapse_double_consonant(
    base: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Collapse trailing doubled consonant if the single form is known.
    - E.g., “redd”→“red”.
    Returns: Collapsed base if found, else None.
    """
    if len(base) < 2 or base[-1] != base[-2]:
        return None

    collapsed = base[:-1]
    if collapsed in known_modifiers or collapsed in known_tones:
        if debug:
            print(f"[🔁 COLLAPSE DOUBLE] '{base}' → '{collapsed}'")
        return collapsed

    return None
