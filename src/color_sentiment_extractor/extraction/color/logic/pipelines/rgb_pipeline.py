"""
rgb_pipeline.py
===============

Does: Resolve RGB from descriptive color phrases via rules, pre-normalization, LLM, and DB/fuzzy fallbacks.
Returns: (simplified phrase, RGB or None) and helpers for LLM-first/DB-first resolution.
Used By: color resolution pipelines, modifier–tone mapping, user input parsing/grounding.
"""

from __future__ import annotations

# ── Imports ───────────────────────────────────────────────────────────────────
import logging
import re
from typing import Optional, Tuple, Set, Protocol

from color_sentiment_extractor.extraction.color import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.color.llm import query_llm_for_rgb
from color_sentiment_extractor.extraction.color.recovery import (
    simplify_color_description_with_llm,
    simplify_phrase_if_needed,
)
from color_sentiment_extractor.extraction.color.suffix import build_y_variant
from color_sentiment_extractor.extraction.color.utils import (
    fuzzy_match_rgb_from_known_colors,
    _try_simplified_match,
)
from color_sentiment_extractor.extraction.general.token import recover_base, normalize_token

# ── Globals & Types ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]


class LLMClient(Protocol):
    """Does: Minimal interface for LLM clients used here. Returns: str/RGB-like payloads."""
    def query(self, text: str) -> Optional[str]: ...
    # If your client exposes a dedicated RGB method, you can extend this Protocol:
    # def query_rgb(self, phrase: str) -> Optional[RGB]: ...


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sanitize_simplified(s: str) -> str:
    """
    Does: Lowercase; keep letters/hyphen/space; collapse spaces; drop 1-char tokens; cap to 2 words.
    Returns: Cleaned phrase ready for normalization or "".
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [w for w in s.split() if len(w) >= 2][:2]
    return " ".join(parts) if parts else ""


def _is_known_color_token(tok: str, known_modifiers: Set[str], known_tones: Set[str]) -> bool:
    """
    Does: Check if token is exactly a known modifier or tone.
    Returns: True if token ∈ known_modifiers ∪ known_tones, else False.
    """
    if not tok:
        return False
    t = tok.strip().lower()
    return (t in known_tones) or (t in known_modifiers)


def _normalize_modifier_tone(
    phrase: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False,
) -> str:
    """
    Does: Normalize “modifier tone” while preserving useful surface forms (-y/-ish) when valid.
    Returns: Normalized “mod tone” or original phrase if no safe normalization applies.
    """
    if not phrase:
        return ""
    p = normalize_token(phrase)  # letters, spaces, hyphens
    parts = p.split()
    if len(parts) != 2:
        return p

    left, right = parts[0], parts[1]

    # Right must be a known tone to proceed
    if right not in known_tones:
        return p

    # (1) left already a known modifier → keep
    if left in known_modifiers:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM KEEP] %r", f"{left} {right}")
        return f"{left} {right}"

    # (2) preserve surface if left ends with -y/-ish and its base is a known modifier
    if left.endswith(("y", "ish")):
        base = recover_base(left, known_modifiers=known_modifiers, known_tones=known_tones,
                            debug=False, fuzzy_fallback=False)
        if base and base in known_modifiers:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[PRE-NORM PRESERVE] %r (base=%r)", f"{left} {right}", base)
            return f"{left} {right}"

    # (3) otherwise try to recover a canonical modifier base; prefer a valid -y variant if present
    base = recover_base(left, known_modifiers=known_modifiers, known_tones=known_tones,
                        debug=False, fuzzy_fallback=False)
    if base and base in known_modifiers:
        y_form = build_y_variant(base)
        if y_form and y_form in known_modifiers and y_form != base:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Y-VARIANT] %r -> %r", base, y_form)
            return f"{y_form} {right}"
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM CANON] %r -> %r", left, base)
        return f"{base} {right}"

    # (4) fallback: keep as-is
    return f"{left} {right}"


# ── LLM-first RGB resolution (with fallbacks) ─────────────────────────────────
def get_rgb_from_descriptive_color_llm_first(
    input_color: str,
    llm_client: Optional[LLMClient],
    cache=None,
    debug: bool = False,
) -> Optional[RGB]:
    """
    Does: Resolve RGB from descriptive color using LLM; fallback to DB and fuzzy matches.
    Returns: RGB tuple or None if unresolved.
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[START] input_color=%r", input_color)

    if not llm_client:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[NO LLM CLIENT] Using fallbacks only")
        # 1) direct DB match
        rgb = _try_simplified_match(input_color, debug=debug)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[DB MATCH raw] → %s", rgb)
        if rgb:
            return rgb
        # 2) fuzzy on raw
        rgb = fuzzy_match_rgb_from_known_colors(input_color)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FUZZY raw] → %s", rgb)
        return rgb

    rgb = query_llm_for_rgb(input_color, llm_client, cache=cache, debug=debug)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[LLM RGB] → %s", rgb)
    if rgb:
        return rgb

    simplified = simplify_color_description_with_llm(input_color, llm_client, cache=cache, debug=debug) or ""
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SIMPLIFIED] → %r", simplified)

    rgb = _try_simplified_match(simplified, debug=debug)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[DB MATCH simplified] → %s", rgb)
    if rgb:
        return rgb

    rgb = fuzzy_match_rgb_from_known_colors(simplified)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FUZZY simplified] → %s", rgb)
    return rgb


def resolve_rgb_with_llm(
    phrase: str,
    llm_client: Optional[LLMClient],
    cache=None,
    debug: bool = False,
    prefer_db_first: bool = False,
) -> Optional[RGB]:
    """
    Does: Entry point for RGB resolution with switchable strategy (DB-first or LLM-first).
    Returns: RGB tuple or None.
    """
    if prefer_db_first or not llm_client:
        rgb = _try_simplified_match(phrase, debug=debug)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[DB FIRST] → %s", rgb)
        if rgb:
            return rgb
        rgb = fuzzy_match_rgb_from_known_colors(phrase)
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FUZZY FIRST] → %s", rgb)
        if rgb or not llm_client:
            return rgb

    return get_rgb_from_descriptive_color_llm_first(
        input_color=phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug,
    )


# ── Public API ────────────────────────────────────────────────────────────────
def process_color_phrase(
    phrase: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client: Optional[LLMClient] = None,
    cache=None,
    debug: bool = False,
) -> Tuple[str, Optional[RGB]]:
    """
    Does: Simplify a color phrase to stable “modifier tone” and resolve RGB via DB/fuzzy/LLM.
    Returns: (simplified_phrase or "", RGB or None).
    """
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("============================================================")
        logger.debug("[INPUT PHRASE] %r", phrase)
        logger.debug("[MODIFIERS] %d", len(known_modifiers))
        logger.debug("[TONES]     %d", len(known_tones))

    # Fast path: single known token → lock and avoid LLM
    raw_norm = normalize_token(phrase)
    if " " not in raw_norm and _is_known_color_token(raw_norm, known_modifiers, known_tones):
        simplified = raw_norm
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[KNOWN TOKEN LOCK] %r → %r — skip LLM", phrase, simplified)
        rgb = resolve_rgb_with_llm(
            simplified,
            llm_client=llm_client,
            cache=cache,
            debug=debug,
            prefer_db_first=True,
        )
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[FINAL RGB] %r → %s", simplified, rgb)
            logger.debug("============================================================")
        return simplified, (None if rgb is False else rgb)

    # 1) Pure rules (no LLM)
    simplified = simplify_phrase_if_needed(
        phrase,
        known_modifiers,
        known_tones,
        llm_client=None,
        cache=cache,
        debug=debug,
    ) or ""
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[AFTER RULES] %r", simplified)

    # 1.5) Pre-normalization from raw (e.g., “dust rose” → “dusty rose”)
    pre_norm = _normalize_modifier_tone(
        _sanitize_simplified(phrase),
        known_modifiers,
        known_tones,
        debug=debug,
    )
    pre_parts = pre_norm.split()
    pre_norm_locked = (len(pre_parts) == 2 and pre_parts[1] in known_tones)

    if pre_norm_locked:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PRE-NORM LOCK] %r → %r — skip LLM", phrase, pre_norm)
        simplified = pre_norm
    else:
        if pre_norm and not simplified:
            simplified = pre_norm

    # 2) LLM fallback only if not locked and still needed
    go_llm = (
        not pre_norm_locked
        and llm_client is not None
        and (
            not simplified
            or simplified == phrase
            or not (len(simplified.split()) == 2 and simplified.split()[1] in known_tones)
        )
    )
    if go_llm:
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[LLM FALLBACK] Trying LLM for %r…", phrase)
        llm_simpl = simplify_color_description_with_llm(
            phrase, llm_client=llm_client, cache=cache, debug=debug
        ) or ""
        if llm_simpl:
            simplified = llm_simpl
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[LLM RETURNED] %r", simplified)

    # 2.5) Sanitize + final normalization (idempotent)
    simplified = _sanitize_simplified(simplified)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SANITIZED] %r", simplified)
    simplified = _normalize_modifier_tone(simplified, known_modifiers, known_tones, debug=debug)

    # 3) Semantic conflict fix (deterministic: pick lexicographically smallest alternative)
    if simplified:
        tokens = simplified.split()
        for i, t in enumerate(tokens):
            for conflict in SEMANTIC_CONFLICTS:
                if t in conflict:
                    replacement = sorted(conflict - {t})[0]
                    if debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("[CONFLICT] %r in %s → %r", t, set(conflict), replacement)
                    tokens[i] = replacement
                    break
        simplified = " ".join(tokens)
    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[AFTER SEMANTIC FIX] %r", simplified)

    # 4) RGB resolution (DB/fuzzy first when pre-norm is locked)
    rgb = resolve_rgb_with_llm(
        simplified or phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug,
        prefer_db_first=pre_norm_locked,
    )

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[FINAL RGB] %r → %s", simplified, rgb)
        logger.debug("============================================================")

    return simplified or "", (None if rgb is False else rgb)
