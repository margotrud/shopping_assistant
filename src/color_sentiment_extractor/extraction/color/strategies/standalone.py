# src/color_sentiment_extractor/extraction/color/strategies/standalone.py
from __future__ import annotations

"""
standalone.py

Does: Extract standalone color terms (tones/modifiers) using strict token matching plus
      expression-based modifier injection, with rule/LLM filtering and final merge.
Returns: set[str] of normalized standalone terms; helpers for tone-only extraction and
         safe injection/capping.
Used by: Color phrase extraction pipelines; pre-compound enrichment and downstream RGB routing.
"""

from typing import Iterable, List, Set, Dict, Optional, cast
import logging

from spacy.tokens import Token

# Public surface
__all__ = [
    "extract_lone_tones",
    "extract_standalone_phrases",
]

__docformat__ = "google"

log = logging.getLogger(__name__)

# ── Domain imports ───────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color.constants import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.recovery import _extract_filtered_tokens
from color_sentiment_extractor.extraction.general.expression.expression_helpers import (
    _inject_expression_modifiers,
)
from color_sentiment_extractor.extraction.general.token import normalize_token
from color_sentiment_extractor.extraction.general.types import TokenLike


# ──────────────────────────────────────────────────────────────
# 1) Tone-only extraction (strict, no LLM)
# ──────────────────────────────────────────────────────────────
def extract_lone_tones(
    tokens: Iterable[Token],
    known_tones: Set[str],
    debug: bool = False,
) -> Set[str]:
    """Extract strict standalone tones present in `known_tones`, skipping cosmetic nouns."""
    matches: Set[str] = set()
    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)
        if raw in COSMETIC_NOUNS:
            if debug:
                log.debug("[⛔ COSMETIC BLOCK] '%s' blocked", raw)
            continue
        if raw in known_tones:
            matches.add(raw)
            if debug:
                log.debug("[🎯 LONE TONE] Found '%s'", raw)
    return matches


# ──────────────────────────────────────────────────────────────
# 2) Injection gating/capping helper
# ──────────────────────────────────────────────────────────────
def _gate_and_cap_injection(
    tokens: Iterable[Token],
    expression_map: Dict[str, List[str]],
    known_modifiers: Set[str],
    injected: Optional[List[str]],
    max_injected: int = 5,
    debug: bool = False,
) -> List[str]:
    """
    Keep only modifiers that:
      • are triggered by an alias actually present in tokens,
      • belong to known_modifiers,
      • are not already present in the input.
    Then cap to `max_injected` in order of first alias appearance.
    """
    present: List[str] = [normalize_token(t.text, keep_hyphens=True) for t in tokens]
    present_set: Set[str] = set(present)

    allowed: List[str] = []
    seen: Set[str] = set()

    # Iterate aliases in order of appearance; deterministic and stable
    for alias in present:
        for m in expression_map.get(alias, []):
            if m in known_modifiers and m not in present_set and m not in seen:
                seen.add(m)
                allowed.append(m)
                if len(allowed) >= max_injected:
                    if debug:
                        log.debug("[🔒 INJECTION CAPPED] %d terms kept", len(allowed))
                    # Respect the cap immediately
                    return allowed

    # If we had a precomputed injected list, intersect while preserving its order
    if injected:
        allowed_set = set(allowed)
        ordered_intersection = [m for m in injected if m in allowed_set]
        if debug:
            log.debug("[🧰 INJECTION GATED] kept=%s", ordered_intersection)
        return ordered_intersection[:max_injected]

    return allowed[:max_injected]


# ──────────────────────────────────────────────────────────────
# 3) Final combination helper
# ──────────────────────────────────────────────────────────────
def _finalize_standalone_phrases(
    injected: Optional[Iterable[str]],
    filtered: Optional[Iterable[str]],
    debug: bool = False,
) -> Set[str]:
    """Union of injected modifiers and filtered tokens, as a set."""
    injected_set = set(injected or [])
    filtered_set = set(filtered or [])
    combined = injected_set | filtered_set
    if debug:
        log.debug("[✅ FINAL STANDALONE SET] %s", combined)
    return combined


# ──────────────────────────────────────────────────────────────
# 4) Main entrypoint for standalone phrase extraction
# ──────────────────────────────────────────────────────────────
def extract_standalone_phrases(
    tokens: Iterable[Token],
    known_modifiers: Set[str],
    known_tones: Set[str],
    expression_map: Dict[str, List[str]],
    llm_client,
    debug: bool = False,
    *,
    max_injected: int = 5,
) -> Set[str]:
    """
    Extract standalone tones/modifiers by:
      1) Expression-based modifier injection (gated + capped),
      2) Rule-based + LLM fallback token filtering,
      3) Final union.
    """
    # always materialize iterable → avoids consuming generators
    tokens_list: List[Token] = list(tokens)

    if debug:
        log.debug("=" * 70)
        log.debug("🎯 ENTER extract_standalone_phrases()")
        log.debug("=" * 70)
        # Avoid dumping full token details in production logs
        log.debug(
            "[📚 STATS] #TOKENS=%d | #MODIFIERS=%d | #TONES=%d",
            len(tokens_list),
            len(known_modifiers),
            len(known_tones),
        )

    # 1) Expression-based modifier injection (gated + capped)
    # _inject_expression_modifiers expects Dict[str, Dict[str, List[str]]] | None
    typed_expression_map: Dict[str, Dict[str, List[str]]] | None = (
        {alias: {"_": mods} for alias, mods in expression_map.items()}
        if expression_map is not None
        else None
    )

    injected_raw: List[str] = _inject_expression_modifiers(
        tokens_list,
        known_modifiers,
        known_tones,
        typed_expression_map,
        debug=debug,
    )

    gated_injected: List[str] = _gate_and_cap_injection(
        tokens=tokens_list,
        expression_map=expression_map,
        known_modifiers=known_modifiers,
        injected=injected_raw,
        max_injected=max_injected,
        debug=debug,
    )
    if debug:
        log.debug(
            "[🧬 EXPRESSION INJECTION] kept=%s (total=%d)",
            sorted(gated_injected),
            len(gated_injected),
        )

    # 2) Rule + LLM fallback filtering
    filtered_terms: Set[str] = _extract_filtered_tokens(
        cast(Iterable[TokenLike], tokens_list),
        known_modifiers,
        known_tones,
        llm_client,
        debug=debug,
    )
    if debug:
        log.debug(
            "[🧠 RULE + LLM FILTERED] kept=%s (total=%d)",
            sorted(filtered_terms),
            len(filtered_terms),
        )

    # 3) Final combination + cleanup
    final: Set[str] = _finalize_standalone_phrases(
        gated_injected,
        filtered_terms,
        debug=debug,
    )
    if debug:
        log.debug("[🏁 FINAL STANDALONE PHRASES] total=%d", len(final))

    return final
