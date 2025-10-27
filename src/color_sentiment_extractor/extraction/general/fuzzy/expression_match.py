# src/color_sentiment_extractor/extraction/color/fuzzy/expression_match.py

"""
expression_match.py.

Does: Match input against expression aliases + modifiers with fuzzy fallbacks,
      de-duplication, and embedded-alias conflict resolution (longest-wins).

Returns: cached_match_expression_aliases(), match_expression_aliases() →
set of canonical expressions.

Used by: Expression→tone mapping stages in color extraction pipelines.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from rapidfuzz import fuzz  # performant, no numpy dependency

# Public alias validation helpers (monorepo coherence)
from color_sentiment_extractor.extraction.general.fuzzy.alias_validation import (
    _handle_multiword_alias,
    is_valid_singleword_alias,
)

# Embedded-conflict rule (single-token morphological embedding)
from color_sentiment_extractor.extraction.general.fuzzy.conflict_rules import (
    is_embedded_alias_conflict,
)
from color_sentiment_extractor.extraction.general.token.normalize import (
    get_tokens_and_counts,
    normalize_token,
)
from color_sentiment_extractor.extraction.general.utils import load_config

__all__ = [
    "cached_match_expression_aliases",
    "match_expression_aliases",
]

__docformat__ = "google"

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
ALIAS_PARTIAL_MIN = 90  # fuzzy cut for modifiers/pass-2
CACHE_SIZE = 1000  # LRU for cached matcher


# ─────────────────────────────────────────────────────────────────────────────
# Cached config
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_expression_def() -> dict[str, dict]:
    """Does: Load validated expression definitions (aliases, modifiers) from config."""
    return load_config("expression_definition", mode="validated_dict")


# ─────────────────────────────────────────────────────────────────────────────
# Core alias acceptance
# ─────────────────────────────────────────────────────────────────────────────


def _contains_as_words(haystack: str, needle: str) -> bool:
    """Does: Word-boundary search (space/hyphen aware) for `needle` inside `haystack`."""
    H = normalize_token(haystack, keep_hyphens=True)
    N = normalize_token(needle, keep_hyphens=True)
    if not H or not N:
        return False
    return re.search(rf"\b{re.escape(N)}\b", H) is not None


def should_accept_alias_match(
    alias: str,
    input_text: str,
    tokens: list[str],
    matched_aliases: set[str] | None = None,
    debug: bool = False,
) -> bool:
    """
    Does: Decide whether alias (single/multi-word) should be accepted against input
    (word-boundary aware).

    Returns:
        True if accepted via exact/word-boundary or delegated matchers.
    """
    alias = (alias or "").strip().lower()
    matched_aliases = matched_aliases or set()

    # Evite re-matcher une sous-partie d’un alias déjà accepté (multiword > single)
    for m in matched_aliases:
        if " " in m and alias in m.split():
            if debug:
                log.debug("[⛔ DUP-SUBPART] %r ∈ %r", alias, m)
            return False

    # Word-boundary containment rapide
    if _contains_as_words(input_text, alias):
        if debug:
            log.debug("[✅ CONTAINS@WB] %r in input", alias)
        return True

    # Délégation stricte
    return (
        _handle_multiword_alias(alias, input_text, debug=debug)
        if " " in alias
        else is_valid_singleword_alias(alias, input_text, tokens, matched_aliases, debug=debug)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main matcher
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=CACHE_SIZE)
def cached_match_expression_aliases(input_text: str) -> set[str]:
    """Does: Cached matcher for default expression map from config."""
    expression_def = _get_expression_def()
    return match_expression_aliases(input_text, expression_def)


def match_expression_aliases(
    input_text: str,
    expression_map: dict[str, dict],
    debug: bool = False,
) -> set[str]:
    """
    Does: Match aliases first (longest aliases first), then modifiers (ranked),
    then resolve conflicts.

    Returns:
        Set of canonical expressions.
    """
    input_tokens: list[str] = list(get_tokens_and_counts(input_text).keys())
    input_tokens_lc = [t.lower() for t in input_tokens]
    input_lower = normalize_token(input_text, keep_hyphens=True)

    matched_expressions: set[str] = set()
    matched_aliases: set[str] = set()
    expr_to_matched_alias: dict[str, str] = {}

    # --- Pass 1: Alias matches (longest aliases first) ---
    for expr, props in expression_map.items():
        aliases = sorted(props.get("aliases", []), key=lambda a: (-len(a.split()), -len(a)))
        for alias in aliases:
            if should_accept_alias_match(
                alias, input_text, input_tokens, matched_aliases, debug=debug
            ):
                canonical = expr
                matched_expressions.add(canonical)
                a_norm = normalize_token(alias, keep_hyphens=True)
                matched_aliases.add(a_norm)
                expr_to_matched_alias[canonical] = a_norm
                if debug:
                    log.debug("[✅ ALIAS] %r → %s", alias, canonical)
                break  # un alias suffit par expression

    # Tokens des alias acceptés (pour exclure ces mots côté modifiers)
    alias_token_blocklist: set[str] = {tok for a in matched_aliases for tok in a.split()}

    # --- Pass 2: Modifiers (scoring) ---
    mod_scores: dict[str, int] = {}
    for expr, props in expression_map.items():
        # si déjà matché via alias, pas besoin de passer par les modifiers
        if expr in matched_expressions:
            continue

        # exclusivité stricte : si un alias a matché ailleurs,
        # ne pas créer d'autres matches via modifiers
        if expr not in matched_expressions and matched_expressions:
            continue

        score = 0
        for mod in props.get("modifiers", []):
            m = normalize_token(mod, keep_hyphens=True).lower()
            if not m:
                continue

            # Skip si un des tokens du mod est déjà “pris” par un alias
            if any(part in alias_token_blocklist for part in m.split()):
                continue

            # direct token match
            if m in input_tokens_lc:
                score += 1
                continue

            # fuzzy sur phrase complète ou token-wise
            if fuzz.ratio(input_lower, m) >= ALIAS_PARTIAL_MIN or any(
                fuzz.ratio(tok, m) >= ALIAS_PARTIAL_MIN for tok in input_tokens_lc
            ):
                score += 1

        if score > 0:
            mod_scores[expr] = score

    if mod_scores:
        top = max(mod_scores.values())
        for expr, sc in mod_scores.items():
            if sc == top:
                matched_expressions.add(expr)
                if debug:
                    log.debug("[✅ MOD-SCORE] %r ← %s", expr, sc)

    # --- Pass 3: Embedded conflicts (use only actually matched aliases) ---
    resolved = _resolve_embedded_conflicts(matched_expressions, expr_to_matched_alias, debug=debug)

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Embedded alias cleanup (longest-wins)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_embedded_conflicts(
    matched_expressions: set[str],
    expr_to_alias: dict[str, str],
    debug: bool = False,
) -> set[str]:
    """
    Does: Remove expressions whose matched alias is embedded inside another matched alias
    (single-token embedding).

    Returns:
        Clean set with preference for longer aliases.
    """
    if len(matched_expressions) <= 1:
        return matched_expressions

    keep: set[str] = set(matched_expressions)  # start optimistic
    items: list[tuple[str, str]] = [
        (expr, expr_to_alias.get(expr, "")) for expr in matched_expressions
    ]

    def _is_single_token(s: str) -> bool:
        return not bool(re.search(r"[\s_\-]", s))

    for i in range(len(items)):
        expr_i, alias_i = items[i]
        for j in range(i + 1, len(items)):
            expr_j, alias_j = items[j]
            if not alias_i or not alias_j:
                continue

            ai = normalize_token(alias_i, keep_hyphens=True)
            aj = normalize_token(alias_j, keep_hyphens=True)

            # only consider single-token morphological embeddings (rose ⊂ rosewood)
            if _is_single_token(ai) and _is_single_token(aj):
                if ai != aj and is_embedded_alias_conflict(aj, ai):
                    # keep longer (aj), drop shorter (ai)
                    if len(aj) > len(ai) and expr_i in keep:
                        keep.discard(expr_i)
                        if debug:
                            log.debug("[⛔ EMBEDDED] %r ⊂ %r → drop %r", ai, aj, expr_i)
                elif aj != ai and is_embedded_alias_conflict(ai, aj):
                    if len(ai) > len(aj) and expr_j in keep:
                        keep.discard(expr_j)
                        if debug:
                            log.debug("[⛔ EMBEDDED] %r ⊂ %r → drop %r", aj, ai, expr_j)

    return keep
