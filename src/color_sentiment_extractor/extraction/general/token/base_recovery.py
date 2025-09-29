# ──────────────────────────────────────────────────────────────
# Token Base Recovery
# Single entrypoint for base-form recovery across the project.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Set

import logging

from color_sentiment_extractor.extraction.color.constants import (
    RECOVER_BASE_OVERRIDES,
    SEMANTIC_CONFLICTS,
    BLOCKED_TOKENS,
)
from color_sentiment_extractor.extraction.general.utils.load_config import load_config
# ⚠️ Évite l’import lourd depuis color.vocab ; charge via config
# from color_sentiment_extractor.extraction.color.vocab import known_tones as KNOWN_TONES

# Registre suffixes (liste ordonnée) + dispatcher optionnel si dispo
from color_sentiment_extractor.extraction.general.token.suffix.registry import (
    SUFFIX_RECOVERY_FUNCS,
)
try:
    # Optionnel : dispo si tu as ajouté le dispatcher suffix-aware
    from color_sentiment_extractor.extraction.general.token.suffix.registry import (
        recover_with_registry as _recover_with_registry,
    )
except Exception:  # pragma: no cover
    _recover_with_registry = None  # type: ignore

# Fuzzy util (assure-toi que __init__.py de fuzzy ré-exporte bien la fonction ;
# sinon, utilise ...general.fuzzy.fuzzy_match: fuzzy_match_token_safe)
from color_sentiment_extractor.extraction.general.fuzzy.fuzzy_match import (
    fuzzy_match_token_safe,
)

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_FUZZY_THRESHOLD = 90

# Vowel sets (module-level to avoid recreating)
VOWELS_CONS: Set[str] = set("aeiou")   # y as consonant
VOWELS_VOW:  Set[str] = set("aeiouy")  # y as vowel

# Canonical sets (chargées depuis la config validée)
KNOWN_MODIFIERS: Set[str] = load_config("known_modifiers", mode="set")
KNOWN_TONES:     Set[str] = load_config("known_tones",     mode="set")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def recover_base(
    token: str,
    *,
    allow_fuzzy: bool = True,
    debug: bool = False,
    **legacy_kwargs,
) -> Optional[str]:
    """
    Does: Map noisy/suffixed token to canonical base via overrides → suffix rules → direct hits → optional fuzzy.
    Returns: Base token or None.
    """
    if not token:
        return None

    # Back-compat param
    if "fuzzy_fallback" in legacy_kwargs and isinstance(legacy_kwargs["fuzzy_fallback"], bool):
        allow_fuzzy = legacy_kwargs["fuzzy_fallback"]

    # Optional vocab overrides (essaitests2/back-compat)
    km = legacy_kwargs.get("known_modifiers", KNOWN_MODIFIERS)
    kt = legacy_kwargs.get("known_tones",     KNOWN_TONES)
    fuzzy_threshold = int(legacy_kwargs.get("fuzzy_threshold", _DEFAULT_FUZZY_THRESHOLD))

    # Normalize vocab to sets once
    if not isinstance(km, set): km = set(km)
    if not isinstance(kt, set): kt = set(kt)

    # Light normalization (no singularize here)
    norm = str(token).lower().strip()
    raw = norm.replace(" ", "").replace("-", "").replace("_", "")

    if not raw:
        return None

    # ✅ Use a set[str] for known tokens (safer for fuzzy_core expectations)
    combined_known: Set[str] = set(km) | set(kt)

    # Optional caching branch (use normalized raw in cache key)
    if legacy_kwargs.get("use_cache"):
        return _recover_base_cached_with_params(
            raw=raw,
            allow_fuzzy=allow_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            km=frozenset(km),
            kt=frozenset(kt),
        )

    # Direct execution
    return _recover_base_impl(
        raw=raw,
        known_modifiers=km,
        known_tones=kt,
        debug=debug,
        fuzzy_fallback=allow_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        depth=0,
        max_depth=3,
        combined_known=combined_known,
    )


@lru_cache(maxsize=10_000)
def _recover_base_cached_with_params(
    raw: str,
    allow_fuzzy: bool,
    fuzzy_threshold: int,
    km: frozenset,
    kt: frozenset,
) -> Optional[str]:
    """Does: Cached entry with stable params in the cache key. Returns: Base token or None."""
    return _recover_base_impl(
        raw=raw,
        known_modifiers=set(km),
        known_tones=set(kt),
        debug=False,
        fuzzy_fallback=allow_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        depth=0,
        max_depth=3,
        # ✅ keep a set here too
        combined_known=set(km) | set(kt),
    )


# ──────────────────────────────────────────────────────────────
# Internal implementation
# ──────────────────────────────────────────────────────────────

def _recover_base_impl(
    raw: str,
    *,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool,
    fuzzy_fallback: bool,
    fuzzy_threshold: int,
    depth: int,
    max_depth: int,
    combined_known: Set[str],
) -> Optional[str]:
    if debug:
        logger.debug(f"[recover_base] raw='{raw}' depth={depth} fuzzy={fuzzy_fallback}")

    if depth > max_depth:
        if debug: logger.debug(f"[⚠️ MAX DEPTH] Reached at '{raw}'")
        return None

    # 1) Manual override (dict or iterable-of-pairs/sets)
    ov = _match_override(raw, known_modifiers, known_tones)
    if ov:
        if debug: logger.debug(f"[override] '{raw}' → '{ov}'")
        return ov

    # 2) Suffix recovery chain (ordered) — with dispatcher si dispo
    if _recover_with_registry:
        try:
            result = _recover_with_registry(raw, known_modifiers, known_tones, debug=False)  # type: ignore[misc]
        except Exception:  # pragma: no cover
            logger.exception("[suffix-dispatch] recover_with_registry crashed")
            result = None
        if result and result != raw:
            chained = _recover_base_impl(
                raw=result,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
                fuzzy_threshold=fuzzy_threshold,
                depth=depth + 1,
                max_depth=max_depth,
                combined_known=combined_known,
            ) or result
            if _is_known_token(chained, known_modifiers, known_tones):
                if debug: logger.debug(f"[suffix ✓] '{raw}' → '{chained}'")
                return chained

    # Fallback: parcours de la liste ordonnée
    for func in SUFFIX_RECOVERY_FUNCS:
        if debug: logger.debug(f"[suffix] trying {getattr(func, '__name__', str(func))}('{raw}')")
        result = _call_suffix_func(func, raw, known_modifiers, known_tones, debug=False)

        if result and result != raw:
            # One level of chaining
            chained = _recover_base_impl(
                raw=result,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
                fuzzy_threshold=fuzzy_threshold,
                depth=depth + 1,
                max_depth=max_depth,
                combined_known=combined_known,
            ) or result

            if _is_known_token(chained, known_modifiers, known_tones):
                if debug: logger.debug(f"[suffix ✓] '{raw}' → '{chained}'")
                return chained

            # Suffix fuzzy salvage (top-level only)
            if depth == 0 and fuzzy_fallback:
                candidate = chained or result
                try:
                    fuzzy_mid = fuzzy_match_token_safe(candidate, combined_known, fuzzy_threshold, False)
                except Exception:  # pragma: no cover
                    logger.exception(
                        f"[fuzzy_mid] crash on candidate={candidate!r} | known={len(combined_known)} | thr={fuzzy_threshold}"
                    )
                    fuzzy_mid = None

                if fuzzy_mid:
                    if (candidate, fuzzy_mid) not in BLOCKED_TOKENS and (fuzzy_mid, candidate) not in BLOCKED_TOKENS and not _is_semantic_conflict(candidate, fuzzy_mid):
                        if debug: logger.debug(f"[suffix fuzzy ✓] '{raw}' → '{candidate}' → '{fuzzy_mid}'")
                        return fuzzy_mid

    # 3) Direct hit
    if _is_known_token(raw, known_modifiers, known_tones):
        if debug: logger.debug(f"[direct ✓] '{raw}'")
        return raw

    # 4) Fuzzy fallback (top-level only)
    if not fuzzy_fallback or depth != 0:
        return None

    # Allow short tokens (len==3) under stricter conditions
    if not (len(raw) >= 4 or (len(raw) == 3 and raw.isalpha())):
        return None

    local_threshold = fuzzy_threshold
    if len(raw) == 3:
        local_threshold = max(local_threshold, 85)

    if debug: logger.debug(f"[fuzzy] probing '{raw}' (threshold={local_threshold})")
    try:
        fuzzy_match = fuzzy_match_token_safe(raw, combined_known, local_threshold, False)
    except Exception:  # pragma: no cover
        logger.exception(
            f"[fuzzy_match] crash on raw={raw!r} | known={len(combined_known)} | thr={local_threshold}"
        )
        fuzzy_match = None

    if fuzzy_match:
        blocked = (raw, fuzzy_match) in BLOCKED_TOKENS or (fuzzy_match, raw) in BLOCKED_TOKENS
        first_letter_mismatch = (len(raw) <= 4 and fuzzy_match[0] != raw[0])
        conflict = _is_semantic_conflict(raw, fuzzy_match)

        if first_letter_mismatch:
            if debug: logger.debug(f"[fuzzy ×] first-letter guard: {raw} → {fuzzy_match} (try abbr fallback)")
            fuzzy_match = None
        elif blocked:
            if debug: logger.debug(f"[fuzzy ×] blocked by BLOCKED_TOKENS: {raw} ↔ {fuzzy_match} (try abbr fallback)")
            fuzzy_match = None
        elif conflict:
            if debug: logger.debug(f"[fuzzy ×] semantic conflict: {raw} ↔ {fuzzy_match} (try abbr fallback)")
            fuzzy_match = None
        else:
            if debug: logger.debug(f"[fuzzy ✓] '{raw}' → '{fuzzy_match}'")
            return fuzzy_match

    # 4b) Abbreviation (consonant-skeleton) fallback for short alpha tokens
    if len(raw) in (3, 4) and raw.isalpha():

        def _sk(s: str, vowels: Set[str]) -> str:
            return "".join(ch for ch in s if ch not in vowels)

        target = _sk(raw, VOWELS_CONS)

        best = None
        # Prefer modifiers over tones; among same type prefer shorter, then lexicographic
        best_score = (-1, 10**9, "")

        for cand in combined_known:
            if not cand:
                continue
            if cand[0] != raw[0]:
                continue
            if len(cand) < len(raw) or len(cand) > 8:
                continue
            if _sk(cand, VOWELS_CONS) == target or _sk(cand, VOWELS_VOW) == target:
                is_mod = 1 if cand in known_modifiers else 0
                score = (is_mod, len(cand), cand)
                if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]) or (score[0] == best_score[0] and score[1] == best_score[1] and score[2] < best_score[2]):
                    best = cand
                    best_score = score

        if best:
            if (raw, best) in BLOCKED_TOKENS or (best, raw) in BLOCKED_TOKENS:
                if debug: logger.debug(f"[abbr ×] blocked by BLOCKED_TOKENS: {raw} ↔ {best}")
                return None
            if _is_semantic_conflict(raw, best):
                if debug: logger.debug(f"[abbr ×] semantic conflict: {raw} ↔ {best}")
                return None
            if debug: logger.debug(f"[abbr ✓] '{raw}' → '{best}' (consonant skeleton)")
            return best

    if debug: logger.debug(f"[no-match] '{raw}'")
    return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _call_suffix_func(
    func,
    raw: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """Does: Call a suffix recovery func with flexible signature. Returns: Result or None."""
    try:
        return func(raw, known_modifiers, known_tones, debug=debug)
    except TypeError:
        try:
            return func(raw, known_modifiers, known_tones)
        except TypeError:
            try:
                return func(raw)
            except TypeError:
                return None


def _is_known_token(tok: str, known_modifiers: Set[str], known_tones: Set[str]) -> bool:
    return tok in known_modifiers or tok in known_tones


def _is_semantic_conflict(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    al, bl = a.lower(), b.lower()
    if isinstance(SEMANTIC_CONFLICTS, dict):
        for k, vs in SEMANTIC_CONFLICTS.items():
            group = {str(k).lower(), *[str(v).lower() for v in (vs if isinstance(vs, Iterable) else [vs])]}
            if al in group and bl in group and al != bl:
                return True
    else:
        for group in SEMANTIC_CONFLICTS:
            g = {str(x).lower() for x in (group if isinstance(group, Iterable) else [group])}
            if al in g and bl in g and al != bl:
                return True
    return False


def _match_override(tok: str, known_modifiers: Set[str], known_tones: Set[str]) -> Optional[str]:
    """Does: Apply RECOVER_BASE_OVERRIDES (dict or iterable of (src,dst)/sets). Returns: Base or None."""
    if isinstance(RECOVER_BASE_OVERRIDES, dict):
        base = RECOVER_BASE_OVERRIDES.get(tok)
        if base and (base in known_modifiers or base in known_tones):
            return base
    else:
        for item in RECOVER_BASE_OVERRIDES:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                src, dst = item
                if src == tok and (dst in known_modifiers or dst in known_tones):
                    return dst
            elif isinstance(item, (set, frozenset)) and len(item) >= 2 and tok in item:
                candidates = sorted(
                    [x for x in item if x in known_modifiers or x in known_tones],
                    key=lambda s: (len(s), s),
                )
                if candidates:
                    return candidates[0]
    return None


def is_known_modifier(tok: str) -> bool:
    return tok in KNOWN_MODIFIERS


def is_known_tone(tok: str) -> bool:
    return tok in KNOWN_TONES
