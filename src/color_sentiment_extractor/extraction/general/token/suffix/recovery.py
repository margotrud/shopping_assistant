"""
recovery.py.

Does: Provides all base recovery functions for -ed, -y, -ish, etc.
Used by: suffix.registry and token_utils for normalization.
"""

from __future__ import annotations

from functools import lru_cache

from color_sentiment_extractor.extraction.color.suffix import (
    build_ey_variant,
    build_y_variant,
    is_cvc_ending,
)
from color_sentiment_extractor.extraction.general.token.suffix.constants import (
    Y_SUFFIX_ALLOWLIST,  # used as an allowlist for certain -ey bases as well
)
import logging

log: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "recover_ed",
    "recover_ee_to_y",
    "recover_en",
    "recover_er",
    "recover_ey",
    "recover_ied",
    "recover_ier",
    "recover_ing",
    "recover_ish",
    "recover_ly",
    "recover_ness",
    "recover_y",
]


# ─────────────────────────────────────────────
# Internal helper to avoid circular imports
# ─────────────────────────────────────────────
def _get_overrides_and_guards():
    """
    Lazy-import override/config sets from modifier_resolution to avoid
    circular import between recovery <-> modifier_resolution.
    """
    from color_sentiment_extractor.extraction.color.recovery.modifier_resolution import (
        ED_SUFFIX_ALLOWLIST,
        NON_SUFFIXABLE_MODIFIERS,
        RECOVER_BASE_OVERRIDES,
    )
    return RECOVER_BASE_OVERRIDES, NON_SUFFIXABLE_MODIFIERS, ED_SUFFIX_ALLOWLIST


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
    Does: Build a suffix-augmented vocabulary from known tokens/modifiers(/tones) with
          deterministic iteration (stable outputs).
          - Recovers base via rules/overrides; generates valid -y / -ey / -ed
            (CVC doubling, y→ied handling, e-drop) without over-producing.
          - Avoids default -ed; restricts -ey to build_ey_variant + known allowlists.
    Returns: Set of valid base and suffixed tokens.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import (
        recover_base,
    )

    (
        RECOVER_BASE_OVERRIDES,
        NON_SUFFIXABLE_MODIFIERS,
        ED_SUFFIX_ALLOWLIST,
    ) = _get_overrides_and_guards()

    known_tones = known_tones or set()

    webcolor_names = webcolor_names or set()

    # seed includes tones to allow e.g., rose→rosy
    raw_inputs = known_tokens | known_modifiers | known_tones
    # reference vocabulary to help fallback recoveries
    recovery_vocab = known_modifiers | known_tokens | known_tones | webcolor_names

    augmented: set[str] = set()

    if debug:
        log.debug("Starting suffix vocab build from %d tokens", len(raw_inputs))

    for raw in sorted(raw_inputs):
        if debug:
            log.debug("[PROCESS] Raw token: %r", raw)

        # 1) Base via override or recover_base
        base = RECOVER_BASE_OVERRIDES.get(raw) or recover_base(raw, use_cache=True)

        # 1.b) Minimal manual fallback if base not resolved and raw not known
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
                        log.debug("[MANUAL OVERRIDE] %r → %r via fallback", raw, retry)
                    base = retry

        if not base:
            if debug:
                log.debug("[SKIP] No base recovered for: %r", raw)
            continue

        if len(base) < 3:
            if debug:
                log.debug("[SKIP] Base too short: %r", base)
            continue

        forms = {raw, base}

        # Respect explicit non-suffixable bases
        if base in NON_SUFFIXABLE_MODIFIERS:
            if debug:
                log.debug("[BLOCKED] Non-suffixable base: %r", base)
            augmented.update(forms)
            continue

        # 2) Generate -y
        y_form = build_y_variant(base, debug=False)
        if y_form:
            forms.add(y_form)

        if debug:
            log.debug("[DEBUG] base=%r, fallback_y=%r", base, base + "y")

        # Force-include base+y if it's already in known_modifiers (not generated)
        fallback_y = base + "y"
        if fallback_y in known_modifiers and fallback_y not in forms:
            forms.add(fallback_y)
            if debug:
                log.debug("[FORCE -y] %r → %r", base, fallback_y)

        # 3) Generate -ey STRICT: only if build_ey_variant AND present in known_modifiers
        ey_form = build_ey_variant(base, raw, debug=False)
        if ey_form and ey_form in known_modifiers:
            forms.add(ey_form)

        # 4) Generate -ed (only for modifiers; no default blanket)
        if base in known_modifiers:
            if base in ED_SUFFIX_ALLOWLIST:
                ed_form = base + "ed"
                forms.add(ed_form)
                if debug:
                    log.debug("[ALLOW -ed] %r → %r", base, ed_form)
            elif is_cvc_ending(base):
                ed_form = base + base[-1] + "ed"
                forms.add(ed_form)
                if debug:
                    log.debug("[CVC -ed] %r → %r", base, ed_form)
            elif base.endswith("y"):
                ed_form = base[:-1] + "ed"  # flashy → flashed
                forms.add(ed_form)
                if debug:
                    log.debug("[Y -ed] %r → %r", base, ed_form)
            elif base.endswith("e"):
                ed_form = base[:-1] + "ed"  # pale → paled
                forms.add(ed_form)
                if debug:
                    log.debug("[E -ed] %r → %r", base, ed_form)

        # 5) Allowlist “raw+ed” when raw is directly allow-listed
        if raw in ED_SUFFIX_ALLOWLIST:
            ed_form_raw = raw + "ed"
            forms.add(ed_form_raw)
            if debug:
                log.debug("[ALLOW -ed RAW] %r → %r", raw, ed_form_raw)

        # 6) CVC + y (e.g., “mat” → “matty”)
        if is_cvc_ending(base):
            cvc_y = base + base[-1] + "y"
            forms.add(cvc_y)
            if debug:
                log.debug("[CVC -y] %r → %r", base, cvc_y)

        if debug:
            log.debug("Final forms for %r: %s", raw, sorted(forms))

        augmented.update(forms)

    if debug:
        log.debug("Done. Final augmented vocab (%d items)", len(augmented))

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
          (Scope intentionally limited to {-y, -ey, -ed} for pipeline usage.)
    Returns: True if recovered base is known and not blocked.
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    (
        RECOVER_BASE_OVERRIDES,
        NON_SUFFIXABLE_MODIFIERS,
        _ED_SUFFIX_ALLOWLIST,
    ) = _get_overrides_and_guards()

    if not token.endswith(("y", "ey", "ed")):
        if debug:
            log.debug("[SKIP] %r → not a handled suffix form", token)
        return False

    # If token is already known (and not in overrides), it's not a *variant*.
    if (token in known_modifiers or token in known_tones) and token not in RECOVER_BASE_OVERRIDES:
        if debug:
            log.debug("[SKIP] %r is already known and not an override", token)
        return False

    base = recover_base(
        token,
        allow_fuzzy=allow_fuzzy,
        debug=debug,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
    )

    is_known_base = base in known_modifiers or base in known_tones
    is_blocked = base in NON_SUFFIXABLE_MODIFIERS if base else False
    valid = bool(base) and is_known_base and not is_blocked

    if debug:
        log.debug(
            "%s | token: %r → base: %r | known_base: %s | blocked: %s",
            "VALID" if valid else "INVALID",
            token,
            base,
            is_known_base,
            is_blocked,
        )
    return valid


# ─────────────────────────────────────────────
# 3) Suffix Recovery Functions
# ─────────────────────────────────────────────
def recover_y(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -y forms (creamy→cream/e, dusty→dust, shiny→shine).
          Tries base, base+e, double-consonant collapse, second y-strip.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("y") or len(token) < 3:
        if debug:
            log.debug("[SKIP] %r → not a valid '-y' suffix candidate", token)
        return None

    if debug:
        log.debug("[recover_y] token=%r", token)

    RECOVER_BASE_OVERRIDES, _, _ = _get_overrides_and_guards()

    # 1) Direct override
    if token in RECOVER_BASE_OVERRIDES:
        base = RECOVER_BASE_OVERRIDES[token]
        if debug:
            log.debug("[Y OVERRIDE] %r → %r", token, base)
        return base

    base = token[:-1]
    candidates: list[str] = [base, base + "e"]

    # Double consonant collapse
    if len(base) >= 3 and base[-1] == base[-2]:
        collapsed = base[:-1]
        candidates.append(collapsed)
        candidates.append(collapsed + "e")

    # second y-strip: “glossyy” → “gloss”
    if base.endswith("y") and len(base) > 3:
        candidates.append(base[:-1])

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug:
                log.debug("[MATCHED] %r → %r", token, cand)
            return cand

    if debug:
        log.debug("[RETURN NONE] recover_y(%r)", token)
    return None


def recover_ed(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ed forms (muted→mute, paled→pale, tapped→tap).
          Tries e-restore, CVC collapse, then raw base.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ed") or len(token) <= 3:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ed' suffix candidate", token)
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
                log.debug("[MATCHED] %r → %r", token, cand)
            return cand

    if debug:
        log.debug("[RETURN NONE] recover_ed(%r)", token)
    return None


def recover_ing(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ing forms (glowing→glow/e).
          Tries base+e then base.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ing") or len(token) <= 5:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ing' suffix candidate", token)
        return None

    base = token[:-3]
    candidates = [base + "e", base]

    for cand in candidates:
        if cand in known_modifiers or cand in known_tones:
            if debug:
                log.debug("[MATCHED] %r → %r", token, cand)
            return cand

    if debug:
        log.debug("[RETURN NONE] recover_ing(%r)", token)
    return None


def recover_ied(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover -ied → -y (tried→try).
          Checks the y-form in known sets.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ied") or len(token) <= 3:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ied' suffix candidate", token)
        return None

    candidate = token[:-3] + "y"

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            log.debug("[IED → Y] %r → %r", token, candidate)
        return candidate

    if debug:
        log.debug("[RETURN NONE] recover_ied(%r)", token)
    return None


def recover_er(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover comparative -er (darker→dark).
          Strips -er and checks base.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("er") or len(token) <= 2:
        if debug:
            log.debug("[SKIP] %r → not a valid '-er' suffix candidate", token)
        return None

    candidate = token[:-2]
    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            log.debug("[COMPARATIVE -ER] %r → %r", token, candidate)
        return candidate

    if debug:
        log.debug("[RETURN NONE] recover_er(%r)", token)
    return None


def recover_ier(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover comparative -ier (trendier→trendy / fancy→fancier).
          Tries stem, stem+y, and overrides on y-form.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ier") or len(token) <= 4:
        if debug:
            log.debug("[SKIP] %r is not a valid '-ier' form", token)
        return None

    stem = token[:-3]
    if debug:
        log.debug("[TRY IER] %r → stem=%r", token, stem)

    if stem in known_modifiers or stem in known_tones:
        if debug:
            log.debug("[IER STRIP] %r → %r (direct match)", token, stem)
        return stem

    RECOVER_BASE_OVERRIDES, _, _ = _get_overrides_and_guards()

    if stem and stem[-1] not in "aeiou":
        y_form = stem + "y"
        if y_form in known_modifiers or y_form in known_tones:
            if debug:
                log.debug("[IER → Y MATCH] %r → %r", token, y_form)
            return y_form

        if y_form in RECOVER_BASE_OVERRIDES:
            override = RECOVER_BASE_OVERRIDES[y_form]
            if override in known_modifiers or override in known_tones:
                if debug:
                    log.debug("[IER → Y → OVERRIDE] %r → %r", token, override)
                return override

    if debug:
        log.debug("[RETURN NONE] recover_ier(%r)", token)
    return None


def recover_ish(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ish forms (greenish→green, ivorish→ivory/ivor+e).
          Tries direct base, collapse doubled consonant, +y/+e, then recover_base(base).
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if debug:
        log.debug("[recover_ish] token=%r", token)

    if "ish" not in token or len(token) <= 4:
        return None

    idx = token.rfind("ish")
    if idx < 2:
        return None

    raw_base = token[:idx]
    base = raw_base.strip("-_. ")

    if base in known_modifiers or base in known_tones:
        if debug:
            log.debug("[ISH DIRECT] %r → %r", token, base)
        return base

    collapsed = _collapse_double_consonant(base, known_modifiers, known_tones, debug=False)
    if collapsed:
        if debug:
            log.debug("[ISH COLLAPSED] %r → %r", token, collapsed)
        return collapsed

    extended_y = base + "y"
    if extended_y in known_modifiers or extended_y in known_tones:
        if debug:
            log.debug("[ISH +Y] %r → %r", token, extended_y)
        return extended_y

    extended_e = base + "e"
    if extended_e in known_modifiers or extended_e in known_tones:
        if debug:
            log.debug("[ISH +E] %r → %r", token, extended_e)
        return extended_e

    from color_sentiment_extractor.extraction.general.token.base_recovery import (
        recover_base as _recover_base,
    )

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
        log.debug("[ISH recover_base] base=%r → %r", base, recovered)

    if recovered:
        if debug:
            log.debug("[ISH CHAINED] %r → %r → %r", token, base, recovered)
        return recovered

    return None


def recover_ness(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ness (softness→soft, creaminess→creamy→cream).
          Handles trailing i-drop toward base; checks base directly.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ness") or len(token) <= 5:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ness' suffix candidate", token)
        return None

    base = token[:-4]
    if debug:
        log.debug("[recover_ness] %r → base=%r", token, base)

    if base.endswith("i"):
        collapsed = base[:-1]
        if collapsed in known_modifiers or collapsed in known_tones:
            if debug:
                log.debug("[NESS i-DROP] %r → %r", token, collapsed)
            return collapsed

    if base in known_modifiers or base in known_tones:
        if debug:
            log.debug("[NESS STRIP] %r → %r", token, base)
        return base

    if debug:
        log.debug("[RETURN NONE] recover_ness(%r)", token)
    return None


def recover_ly(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ly adverbs (softly→soft, warmly→warm).
          Special-case “ally”→“ic”, else strip -ly. Filtered by known_*.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ly") or len(token) <= 3:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ly' suffix candidate", token)
        return None

    candidate = token[:-4] + "ic" if token.endswith("ally") else token[:-2]

    if debug:
        log.debug("[recover_ly] %r → candidate=%r", token, candidate)

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            log.debug("[ADVERB -LY] %r → %r", token, candidate)
        return candidate

    if debug:
        log.debug("[RETURN NONE] recover_ly(%r)", token)
    return None


def recover_en(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -en adjectives (golden→gold).
          Strips -en and checks direct base.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("en") or len(token) <= 3:
        if debug:
            log.debug("[SKIP] %r → not a valid '-en' suffix candidate", token)
        return None

    candidate = token[:-2]
    if debug:
        log.debug("[recover_en] %r → candidate=%r", token, candidate)

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            log.debug("[EN SUFFIX] %r → %r", token, candidate)
        return candidate

    if debug:
        log.debug("[RETURN NONE] recover_en(%r)", token)
    return None


def recover_ey(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover base from -ey forms (bronzey→bronze, beigey→beige).
          Prefers base+e; allowlist fallback for specific roots.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ey") or len(token) <= 4:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ey' suffix candidate", token)
        return None

    base = token[:-2]
    candidate = base + "e"

    if debug:
        log.debug("[recover_ey] %r → base=%r, candidate=%r", token, base, candidate)

    if candidate in known_modifiers or candidate in known_tones:
        if debug:
            log.debug("[EY +E RESTORE] %r → %r", token, candidate)
        return candidate

    # Reuse Y_SUFFIX_ALLOWLIST as a conservative allowlist of acceptable roots for -ey
    if base in Y_SUFFIX_ALLOWLIST:
        if debug:
            log.debug("[EY ALLOWLIST] %r → %r", token, base)
        return base

    if debug:
        log.debug("[BLOCKED] '-ey' not recoverable for base=%r", base)
    return None


def recover_ee_to_y(
    token: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Recover -ee→-y (ivoree→ivory).
          Builds candidate base with trailing y.
    Returns: Base token if found, else None.
    """
    token = (token or "").strip().lower()
    if not token.endswith("ee") or len(token) < 4:
        if debug:
            log.debug("[SKIP] %r → not a valid '-ee' suffix candidate", token)
        return None

    base = token[:-2] + "y"
    if debug:
        log.debug("[recover_ee_to_y] %r → %r", token, base)

    if base in known_modifiers or base in known_tones:
        if debug:
            log.debug("[EE→Y] %r → %r", token, base)
        return base

    if debug:
        log.debug("[RETURN NONE] recover_ee_to_y(%r)", token)
    return None


# ─────────────────────────────────────────────
# 4) Helpers
# ─────────────────────────────────────────────
def _collapse_double_consonant(
    base: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str | None:
    """
    Does: Collapse trailing doubled consonant if the single form is known (e.g., “redd”→“red”).
    Returns: Collapsed base if found, else None.
    """
    if len(base) < 2 or base[-1] != base[-2]:
        return None

    collapsed = base[:-1]
    if collapsed in known_modifiers or collapsed in known_tones:
        if debug:
            log.debug("[COLLAPSE DOUBLE] %r → %r", base, collapsed)
        return collapsed

    return None
