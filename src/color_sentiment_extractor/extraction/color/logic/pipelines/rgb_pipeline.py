"""
rgb_pipeline.py.
===============

Does: Resolve RGB from descriptive color phrases via rules, pre-normalization,
LLM, and DB/fuzzy fallbacks.
Returns: (simplified phrase, RGB or None) and helpers for LLM-first/DB-first
resolution.
Used By: color resolution pipelines, modifier–tone mapping, user input
parsing/grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, cast

from color_sentiment_extractor.extraction.color import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.color.llm import (
    query_llm_for_rgb,
)
from color_sentiment_extractor.extraction.color.recovery import (
    simplify_color_description_with_llm,
    simplify_phrase_if_needed,
)
from color_sentiment_extractor.extraction.color.suffix import build_y_variant
from color_sentiment_extractor.extraction.color.utils import (
    _try_simplified_match,
    fuzzy_match_rgb_from_known_colors,
)
from color_sentiment_extractor.extraction.general.token import (
    normalize_token,
    recover_base,
)

logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]


class LLMClient(Protocol):
    """Minimal surface for an LLM client usable here."""

    def query(self, text: str) -> str | None: ...

    # If your real client exposes extra methods (query_rgb, etc.) that's fine,
    # Protocol is structural so extra attrs won't break callers.


# ── Internal typing helpers ──────────────────────────────────────────────
def _coerce_rgb(val: object) -> RGB | None:
    """
    Try to interpret arbitrary `val` as an RGB triple.
    Only accepts (r,g,b) where each is int.
    Returns None if not valid.
    """
    if isinstance(val, tuple) and len(val) == 3:
        r, g, b = val
        if all(isinstance(x, int) for x in (r, g, b)):
            return (r, g, b)
    if isinstance(val, list) and len(val) == 3:
        r, g, b = val
        if all(isinstance(x, int) for x in (r, g, b)):
            # cast list -> tuple for consistency
            return (r, g, b)
    return None


def _best_rgb_from_name_like(name: str) -> RGB | None:
    """
    Wrapper around fuzzy_match_rgb_from_known_colors() which may return
    a non-RGB (e.g. str) depending on implementation.
    We standardize to Optional[RGB] for mypy.
    """
    raw = fuzzy_match_rgb_from_known_colors(name)
    return _coerce_rgb(raw)


def _as_any(client: LLMClient | None) -> Any:
    """
    simplify_color_description_with_llm() is typed to expect a different
    LLMClient class (from another module). At runtime our client is fine,
    but mypy sees incompatible nominal types.

    We downcast to Any at the call site to silence arg-type complaints.
    """
    return client


# ── Small text utilities ─────────────────────────────────────────────────
def _sanitize_simplified(s: str) -> str:
    """
    Lowercase, keep letters/hyphen/space, collapse spaces, drop 1-char tokens,
    and cap to 2 words. Returns cleaned phrase or "".
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [w for w in s.split() if len(w) >= 2][:2]
    return " ".join(parts) if parts else ""


def _is_known_color_token(tok: str, known_modifiers: set[str], known_tones: set[str]) -> bool:
    """Returns True iff token is exactly a known modifier or tone."""
    if not tok:
        return False
    t = tok.strip().lower()
    return (t in known_tones) or (t in known_modifiers)


def _normalize_modifier_tone(
    phrase: str,
    known_modifiers: set[str],
    known_tones: set[str],
    debug: bool = False,
) -> str:
    """
    Normalize “modifier tone” while preserving surface forms like -y/-ish
    when still consistent with a known base modifier.
    """
    if not phrase:
        return ""
    p = normalize_token(phrase)
    parts = p.split()
    if len(parts) != 2:
        return p

    left, right = parts[0], parts[1]

    # right must be a known tone to proceed
    if right not in known_tones:
        return p

    # (1) left already a known modifier → keep
    if left in known_modifiers:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM KEEP] %r", f"{left} {right}")
        return f"{left} {right}"

    # (2) if left ends with y/ish and its base is a known modifier,
    #     allow keeping the surface form
    if left.endswith(("y", "ish")):
        base = recover_base(
            left,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            debug=False,
            fuzzy_fallback=False,
        )
        if base and base in known_modifiers:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[PRE-NORM PRESERVE] %r (base=%r)",
                    f"{left} {right}",
                    base,
                )
            return f"{left} {right}"

    # (3) otherwise recover canonical base and maybe build y-variant
    base = recover_base(
        left,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        debug=False,
        fuzzy_fallback=False,
    )
    if base and base in known_modifiers:
        y_form = build_y_variant(base)
        if y_form and y_form in known_modifiers and y_form != base:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Y-VARIANT] %r -> %r", base, y_form)
            return f"{y_form} {right}"
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM CANON] %r -> %r", left, base)
        return f"{base} {right}"

    # (4) fallback
    return f"{left} {right}"


# ── LLM-first RGB resolution (with fallbacks) ────────────────────────────
def get_rgb_from_descriptive_color_llm_first(
    input_color: str,
    llm_client: LLMClient | None,
    cache: dict[str, Any] | None = None,
    debug: bool = False,
) -> RGB | None:
    """
    Resolve RGB from descriptive color using LLM; fallback to DB and fuzzy matches.
    Returns RGB or None.
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[START] input_color=%r", input_color)

    # Pas de client LLM → DB/fuzzy direct
    if not llm_client:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[NO LLM CLIENT] Using fallbacks only")

        rgb_direct = _try_simplified_match(input_color, debug=debug)
        coerced = _coerce_rgb(rgb_direct)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[DB MATCH raw] → %s (coerced=%s)", rgb_direct, coerced)
        if coerced:
            return coerced

        fuzzy_rgb = _best_rgb_from_name_like(input_color)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FUZZY raw] → %s", fuzzy_rgb)
        return fuzzy_rgb

    # 1) Demande directe LLM → RGB
    rgb_llm = query_llm_for_rgb(
        input_color,
        llm_client,
        cache=cast(Any, cache),
        debug=debug,
    )
    coerced_llm = _coerce_rgb(rgb_llm)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[LLM RGB] → %s (coerced=%s)", rgb_llm, coerced_llm)
    if coerced_llm:
        return coerced_llm

    # 2) LLM pour simplifier la phrase
    simplified = (
        simplify_color_description_with_llm(
            input_color,
            _as_any(llm_client),
            cache=cache,
            debug=debug,
        )
        or ""
    )
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SIMPLIFIED] → %r", simplified)

    # 3) DB sur la version simplifiée
    rgb_simple = _try_simplified_match(simplified, debug=debug)
    coerced_simpl = _coerce_rgb(rgb_simple)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[DB MATCH simplified] → %s (coerced=%s)",
            rgb_simple,
            coerced_simpl,
        )
    if coerced_simpl:
        return coerced_simpl

    # 4) fuzzy sur la version simplifiée
    fuzzy_simple = _best_rgb_from_name_like(simplified)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FUZZY simplified] → %s", fuzzy_simple)

    return fuzzy_simple


def resolve_rgb_with_llm(
    phrase: str,
    llm_client: LLMClient | None,
    cache: dict[str, Any] | None = None,
    debug: bool = False,
    prefer_db_first: bool = False,
) -> RGB | None:
    """
    Entry point for RGB resolution with strategy flag.
    - DB-first (fast path / no LLM)
    - or LLM-first (richer recovery).
    """
    if prefer_db_first or not llm_client:
        rgb_fast = _try_simplified_match(phrase, debug=debug)
        coerced_fast = _coerce_rgb(rgb_fast)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[DB FIRST] → %s (coerced=%s)", rgb_fast, coerced_fast)
        if coerced_fast:
            return coerced_fast

        fuzzy_fast = _best_rgb_from_name_like(phrase)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FUZZY FIRST] → %s", fuzzy_fast)
        if fuzzy_fast or not llm_client:
            return fuzzy_fast

    return get_rgb_from_descriptive_color_llm_first(
        input_color=phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug,
    )


# ── Public API ──────────────────────────────────────────────────────────
def process_color_phrase(
    phrase: str,
    known_modifiers: set[str],
    known_tones: set[str],
    llm_client: LLMClient | None = None,
    cache: dict[str, Any] | None = None,
    debug: bool = False,
) -> tuple[str, RGB | None]:
    """
    Simplify a color phrase to stable “modifier tone” and resolve RGB via
    DB/fuzzy/LLM.
    Returns: (simplified_phrase or "", RGB or None).
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("============================================================")
        logger.debug("[INPUT PHRASE] %r", phrase)
        logger.debug("[MODIFIERS] %d", len(known_modifiers))
        logger.debug("[TONES]     %d", len(known_tones))

    raw_norm = normalize_token(phrase)

    # Fast path: phrase est un seul token connu → pas d'LLM
    if " " not in raw_norm and _is_known_color_token(raw_norm, known_modifiers, known_tones):
        simplified_lock = raw_norm
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[KNOWN TOKEN LOCK] %r → %r — skip LLM",
                phrase,
                simplified_lock,
            )

        rgb_locked = resolve_rgb_with_llm(
            simplified_lock,
            llm_client=llm_client,
            cache=cache,
            debug=debug,
            prefer_db_first=True,
        )
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FINAL RGB] %r → %s", simplified_lock, rgb_locked)
            logger.debug("============================================================")
        return simplified_lock, rgb_locked if isinstance(rgb_locked, tuple) else None

    # 1) Règles pures (pas d'LLM)
    simplified = (
        simplify_phrase_if_needed(
            phrase,
            known_modifiers,
            known_tones,
            llm_client=None,
            cache=cache,
            debug=debug,
        )
        or ""
    )
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[AFTER RULES] %r", simplified)

    # 1.5) Pré-normalisation (ex: “dust rose” → “dusty rose”)
    pre_norm = _normalize_modifier_tone(
        _sanitize_simplified(phrase),
        known_modifiers,
        known_tones,
        debug=debug,
    )
    pre_parts = pre_norm.split()
    pre_norm_locked = len(pre_parts) == 2 and pre_parts[1] in known_tones

    if pre_norm_locked:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM LOCK] %r → %r — skip LLM", phrase, pre_norm)
        simplified = pre_norm
    else:
        if pre_norm and not simplified:
            simplified = pre_norm

    # 2) Fallback LLM si pas locké et encore ambigu
    need_llm = (
        not pre_norm_locked
        and llm_client is not None
        and (
            not simplified
            or simplified == phrase
            or not (len(simplified.split()) == 2 and simplified.split()[1] in known_tones)
        )
    )
    if need_llm:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[LLM FALLBACK] Trying LLM for %r…", phrase)
        llm_simpl = (
            simplify_color_description_with_llm(
                phrase,
                _as_any(llm_client),
                cache=cache,
                debug=debug,
            )
            or ""
        )
        if llm_simpl:
            simplified = llm_simpl
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[LLM RETURNED] %r", simplified)

    # 2.5) Sanitize + normalisation finale
    simplified = _sanitize_simplified(simplified)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SANITIZED] %r", simplified)
    simplified = _normalize_modifier_tone(simplified, known_modifiers, known_tones, debug=debug)

    # 3) Semantic conflict fix (si ex: {"warm","cool"} etc.)
    if simplified:
        tokens = simplified.split()
        for i, t in enumerate(tokens):
            for conflict in SEMANTIC_CONFLICTS:
                if t in conflict:
                    # choix déterministe → plus petit lexicographiquement
                    replacement = sorted(conflict - {t})[0]
                    if debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[CONFLICT] %r in %s → %r",
                            t,
                            set(conflict),
                            replacement,
                        )
                    tokens[i] = replacement
                    break
        simplified = " ".join(tokens)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[AFTER SEMANTIC FIX] %r", simplified)

    # 4) Résolution RGB finale
    final_rgb = resolve_rgb_with_llm(
        simplified or phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug,
        prefer_db_first=pre_norm_locked,
    )

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FINAL RGB] %r → %s", simplified, final_rgb)
        logger.debug("============================================================")

    return simplified or "", final_rgb if isinstance(final_rgb, tuple) else None
