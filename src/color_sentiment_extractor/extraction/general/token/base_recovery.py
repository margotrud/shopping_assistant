# src/color_sentiment_extractor/extraction/general/token/base_recovery.py
# ──────────────────────────────────────────────────────────────
# Token Base Recovery
# Single entrypoint for base-form recovery across the project.
# ──────────────────────────────────────────────────────────────

"""
base_recovery.

Does: Map noisy/suffixed tokens to canonical bases via
      overrides → suffix recovery → direct hits → fuzzy/abbr fallbacks,
      with depth safeguards and optional suffix dispatcher.
Returns: recover_base() plus helpers is_known_modifier()/is_known_tone().
Used by: Token normalization flows and color/modifier extraction stages.

Notes:
- Normalization ici volontairement légère: lower + strip + remove spaces/hyphens/underscores.
  (On n'appelle PAS normalize_token pour garder la sémantique v1.)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from collections.abc import Set as AbcSet
from functools import lru_cache

from color_sentiment_extractor.extraction.color.constants import (
    BLOCKED_TOKENS,
    RECOVER_BASE_OVERRIDES,
    SEMANTIC_CONFLICTS,
)
from color_sentiment_extractor.extraction.color.vocab import (
    get_known_tones as _vocab_get_known_tones,
)
from color_sentiment_extractor.extraction.general.fuzzy import fuzzy_match_token_safe

# Suffix registry (ordered list) + optional suffix-aware dispatcher
from color_sentiment_extractor.extraction.general.token.suffix.registry import (
    SUFFIX_RECOVERY_FUNCS,
    RecoverFn,  # <- on importe le type pour annoter proprement
)
from color_sentiment_extractor.extraction.general.utils.load_config import (
    ConfigFileNotFound,
    load_config,
)

# Annotation **avant** le try/except pour que mypy accepte l'affectation à None
_recover_with_registry_impl: RecoverFn | None
try:
    # Optional: available if dispatcher is included in the build
    from color_sentiment_extractor.extraction.general.token.suffix.registry import (
        recover_with_registry as _recover_with_registry_impl,
    )
except ImportError:  # pragma: no cover
    _recover_with_registry_impl = None

# Local optional callable for mypy
_recover_with_registry: RecoverFn | None = _recover_with_registry_impl

__all__ = [
    "recover_base",
    "is_known_modifier",
    "is_known_tone",
]

logger = logging.getLogger(__name__)

# Defaults / constants
_DEFAULT_FUZZY_THRESHOLD = 90

# Vowel sets (module-level to avoid recreation)
VOWELS_CONS: set[str] = set("aeiou")  # y as consonant
VOWELS_VOW: set[str] = set("aeiouy")  # y as vowel

# Canonical sets (loaded from validated config)
# Note: load_config(mode="set") returns a FrozenSet[str] → on le typpe en FrozenSet.
KNOWN_MODIFIERS: frozenset[str] = load_config("known_modifiers", mode="set")
KNOWN_TONES: frozenset[str] = frozenset(_vocab_get_known_tones())

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_default_known_tones() -> frozenset[str]:
    # Vient du vocab (CSS + XKCD + fallbacks), pas du JSON disque
    return frozenset(_vocab_get_known_tones())


@lru_cache(maxsize=1)
def _get_default_known_modifiers() -> frozenset[str]:
    # Prend known_modifiers depuis data/ si présent, sinon set vide (safe)
    try:
        return load_config("known_modifiers", mode="set")
    except ConfigFileNotFound:
        return frozenset()


def _call_suffix_func(
    func: Callable[..., str | None],
    raw: str,
    known_modifiers: AbcSet[str],
    known_tones: AbcSet[str],
    debug: bool = False,
) -> str | None:
    """Does: Call a suffix recovery function with flexible signature. Returns: Result or None."""
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


def _is_known_token(tok: str, known_modifiers: AbcSet[str], known_tones: AbcSet[str]) -> bool:
    return tok in known_modifiers or tok in known_tones


def _is_semantic_conflict(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    al, bl = a.lower(), b.lower()
    try:
        if isinstance(SEMANTIC_CONFLICTS, dict):
            for k, vs in SEMANTIC_CONFLICTS.items():
                group = {
                    str(k).lower(),
                    *[str(v).lower() for v in (vs if isinstance(vs, Iterable) else [vs])],
                }
                if al in group and bl in group and al != bl:
                    return True
        else:
            for group in SEMANTIC_CONFLICTS:
                g = {str(x).lower() for x in (group if isinstance(group, Iterable) else [group])}
                if al in g and bl in g and al != bl:
                    return True
    except Exception:  # defensive
        return False
    return False


def _match_override(tok: str, known_modifiers: AbcSet[str], known_tones: AbcSet[str]) -> str | None:
    """
    Does: Apply RECOVER_BASE_OVERRIDES (dict or iterable of (src, dst)/sets).
    Returns: Base or None.
    """
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


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def recover_base(
    token: str,
    *,
    allow_fuzzy: bool = True,
    known_modifiers: Iterable[str] | None = None,
    known_tones: Iterable[str] | None = None,
    debug: bool = False,
    **legacy_kwargs,
) -> str | None:
    """
    Does: Map a noisy/suffixed token to a canonical base via:
          overrides → suffix rules (registry/dispatcher) → direct hit → fuzzy → abbr fallback.
    Returns: Base token or None.

    Back-compat:
      - legacy_kwargs["fuzzy_fallback"] (bool) mirrors allow_fuzzy (v1 compat).
      - legacy_kwargs may pass known_modifiers/known_tones/fuzzy_threshold/use_cache.
    """
    if not token:
        return None

    # Back-compat param
    if "fuzzy_fallback" in legacy_kwargs and isinstance(legacy_kwargs["fuzzy_fallback"], bool):
        allow_fuzzy = legacy_kwargs["fuzzy_fallback"]

    # Optional vocab overrides
    km0: frozenset[str] = (
        frozenset(known_modifiers)
        if known_modifiers is not None
        else _get_default_known_modifiers()
    )
    kt0: frozenset[str] = (
        frozenset(known_tones) if known_tones is not None else _get_default_known_tones()
    )

    fuzzy_threshold = int(legacy_kwargs.get("fuzzy_threshold", _DEFAULT_FUZZY_THRESHOLD))

    # Normalize vocab to sets once (variables locales mutables)
    km: set[str] = set(km0)
    kt: set[str] = set(kt0)

    # Light normalization (no singularize here)
    norm = str(token).lower().strip()
    raw = norm.replace(" ", "").replace("-", "").replace("_", "")

    if not raw:
        return None

    # Use a set[str] for known tokens (safer for fuzzy_core expectations)
    combined_known: set[str] = km | kt

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
    km: frozenset[str],
    kt: frozenset[str],
) -> str | None:
    """Does: Cached entry with stable params in the cache key. Returns: Base token or None."""
    km_set = set(km)
    kt_set = set(kt)
    return _recover_base_impl(
        raw=raw,
        known_modifiers=km_set,
        known_tones=kt_set,
        debug=False,
        fuzzy_fallback=allow_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        depth=0,
        max_depth=3,
        combined_known=km_set | kt_set,
    )


# ──────────────────────────────────────────────────────────────
# Internal implementation
# ──────────────────────────────────────────────────────────────


def _recover_base_impl(
    raw: str,
    *,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool,
    fuzzy_fallback: bool,
    fuzzy_threshold: int,
    depth: int,
    max_depth: int,
    combined_known: set[str],
) -> str | None:
    if debug:
        logger.debug("[recover_base] raw=%r depth=%d fuzzy=%s", raw, depth, fuzzy_fallback)

    if depth > max_depth:
        if debug:
            logger.debug("[MAX DEPTH] reached at %r", raw)
        return None

    # 1) Manual override (dict or iterable-of-pairs/sets)
    ov = _match_override(raw, known_modifiers, known_tones)
    if ov:
        if debug:
            logger.debug("[override] %r → %r", raw, ov)
        return ov

    # 2) Suffix recovery chain (ordered) — prefer dispatcher if available
    if _recover_with_registry is not None:
        try:
            # IMPORTANT: appel **positionnel** (pas debug=...) pour éviter l'erreur mypy
            result = _recover_with_registry(raw, known_modifiers, known_tones, False)
        except Exception:  # pragma: no cover
            logger.exception("[suffix-dispatch] recover_with_registry crashed")
            result = None
        if result and result != raw:
            chained = (
                _recover_base_impl(
                    raw=result,
                    known_modifiers=known_modifiers,
                    known_tones=known_tones,
                    debug=False,
                    fuzzy_fallback=False,
                    fuzzy_threshold=fuzzy_threshold,
                    depth=depth + 1,
                    max_depth=max_depth,
                    combined_known=combined_known,
                )
                or result
            )
            if _is_known_token(chained, known_modifiers, known_tones):
                if debug:
                    logger.debug("[suffix ✓] %r → %r", raw, chained)
                return chained

    # Fallback: walk ordered list
    for func in SUFFIX_RECOVERY_FUNCS:
        if debug:
            logger.debug("[suffix] trying %s(%r)", getattr(func, "__name__", str(func)), raw)
        result = _call_suffix_func(func, raw, known_modifiers, known_tones, debug=False)

        if result and result != raw:
            # One level of chaining
            chained = (
                _recover_base_impl(
                    raw=result,
                    known_modifiers=known_modifiers,
                    known_tones=known_tones,
                    debug=False,
                    fuzzy_fallback=False,
                    fuzzy_threshold=fuzzy_threshold,
                    depth=depth + 1,
                    max_depth=max_depth,
                    combined_known=combined_known,
                )
                or result
            )

            if _is_known_token(chained, known_modifiers, known_tones):
                if debug:
                    logger.debug("[suffix ✓] %r → %r", raw, chained)
                return chained

            # Suffix fuzzy salvage (top-level only)
            if depth == 0 and fuzzy_fallback:
                candidate = chained or result
                try:
                    fuzzy_mid = fuzzy_match_token_safe(
                        candidate, combined_known, fuzzy_threshold, False
                    )
                except Exception:  # pragma: no cover
                    logger.exception(
                        "[fuzzy_mid] crash on candidate=%r | known=%d | thr=%d",
                        candidate,
                        len(combined_known),
                        fuzzy_threshold,
                    )
                    fuzzy_mid = None

                if fuzzy_mid:
                    if (
                        (candidate, fuzzy_mid) not in BLOCKED_TOKENS
                        and (fuzzy_mid, candidate) not in BLOCKED_TOKENS
                        and not _is_semantic_conflict(candidate, fuzzy_mid)
                    ):
                        if debug:
                            logger.debug("[suffix fuzzy ✓] %r → %r → %r", raw, candidate, fuzzy_mid)
                        return fuzzy_mid

    # 3) Direct hit
    if _is_known_token(raw, known_modifiers, known_tones):
        if debug:
            logger.debug("[direct ✓] %r", raw)
        return raw

    # 4) Fuzzy fallback (top-level only)
    if not fuzzy_fallback or depth != 0:
        return None

    # Allow short tokens (len==3) under stricter conditions
    if not (len(raw) >= 4 or (len(raw) == 3 and raw.isalpha())):
        return None

    local_threshold = max(_DEFAULT_FUZZY_THRESHOLD, 85) if len(raw) == 3 else fuzzy_threshold

    if debug:
        logger.debug("[fuzzy] probing %r (threshold=%d)", raw, local_threshold)
    try:
        fuzzy_match = fuzzy_match_token_safe(raw, combined_known, local_threshold, False)
    except Exception:  # pragma: no cover
        logger.exception(
            "[fuzzy_match] crash on raw=%r | known=%d | thr=%d",
            raw,
            len(combined_known),
            local_threshold,
        )
        fuzzy_match = None

    if fuzzy_match:
        blocked = (raw, fuzzy_match) in BLOCKED_TOKENS or (fuzzy_match, raw) in BLOCKED_TOKENS
        first_letter_mismatch = len(raw) <= 4 and fuzzy_match[0] != raw[0]
        conflict = _is_semantic_conflict(raw, fuzzy_match)

        if first_letter_mismatch:
            if debug:
                logger.debug(
                    "[fuzzy ×] first-letter guard: %s → %s (try abbr fallback)", raw, fuzzy_match
                )
            fuzzy_match = None
        elif blocked:
            if debug:
                logger.debug(
                    "[fuzzy ×] blocked by BLOCKED_TOKENS: %s ↔ %s (try abbr fallback)",
                    raw,
                    fuzzy_match,
                )
            fuzzy_match = None
        elif conflict:
            if debug:
                logger.debug(
                    "[fuzzy ×] semantic conflict: %s ↔ %s (try abbr fallback)", raw, fuzzy_match
                )
            fuzzy_match = None
        else:
            if debug:
                logger.debug("[fuzzy ✓] %r → %r", raw, fuzzy_match)
            return fuzzy_match

    # 4b) Abbreviation (consonant-skeleton) fallback for short alpha tokens
    if len(raw) in (3, 4) and raw.isalpha():

        @lru_cache(maxsize=8_192)
        def _sk_cons(s: str) -> str:
            return "".join(ch for ch in s if ch not in VOWELS_CONS)

        @lru_cache(maxsize=8_192)
        def _sk_vow(s: str) -> str:
            return "".join(ch for ch in s if ch not in VOWELS_VOW)

        target = _sk_cons(raw)

        best: str | None = None
        # Prefer modifiers over tones; among same type prefer shorter, then lexicographic
        best_score: tuple[int, int, str] = (-1, 10**9, "")

        for cand in combined_known:
            if not cand:
                continue
            if cand[0] != raw[0]:
                continue
            if len(cand) < len(raw) or len(cand) > 8:
                continue
            if _sk_cons(cand) == target or _sk_vow(cand) == target:
                is_mod = 1 if cand in known_modifiers else 0
                score = (is_mod, len(cand), cand)
                if (
                    score[0] > best_score[0]
                    or (score[0] == best_score[0] and score[1] < best_score[1])
                    or (
                        score[0] == best_score[0]
                        and score[1] == best_score[1]
                        and score[2] < best_score[2]
                    )
                ):
                    best = cand
                    best_score = score

        if best:
            if (raw, best) in BLOCKED_TOKENS or (best, raw) in BLOCKED_TOKENS:
                if debug:
                    logger.debug("[abbr ×] blocked by BLOCKED_TOKENS: %s ↔ %s", raw, best)
                return None
            if _is_semantic_conflict(raw, best):
                if debug:
                    logger.debug("[abbr ×] semantic conflict: %s ↔ %s", raw, best)
                return None
            if debug:
                logger.debug("[abbr ✓] %r → %r (consonant skeleton)", raw, best)
            return best

    if debug:
        logger.debug("[no-match] %r", raw)
    return None
