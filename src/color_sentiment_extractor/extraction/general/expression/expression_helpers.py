# src/color_sentiment_extractor/extraction/general/expression/expression_helpers.py

"""
expression_helpers.py.
=====================

Does: Utilities to build expression trigger vocabularies and perform expression-driven
      matching & mapping:
      - Build trigger tokens (aliases + modifiers) from expression definitions
      - Match expressions (exact + fuzzy) and map to valid tones
      - Inject modifiers from expression hits and token/base recovery
      - Apply contextual promotion and suppression rules

Returns: Public helpers for expression alias matching, tone mapping, and glued-token
vocabulary.
Used by: Expression matching (contextual tone detection), compound token splitting.
"""

from __future__ import annotations

from color_sentiment_extractor.extraction.general.fuzzy.expression_match import (
    match_expression_aliases,
)

# ── Public surface ───────────────────────────────────────────────────────────
__all__ = [
    # Trigger vocab
    "get_all_trigger_tokens",
    "get_all_alias_tokens",
    # Matching & mapping
    "extract_exact_alias_tokens",
    "get_matching_expression_tags_cached",
    "map_expressions_to_tones",
    # Rules
    "apply_expression_context_rules",
    "apply_expression_suppression_rules",
    # Glued-token vocab
    "get_glued_token_vocabulary",
]
__docformat__ = "google"

# ── Imports & Typing ─────────────────────────────────────────────────────────
import logging
import re
import time
from collections.abc import Iterable
from functools import lru_cache

from color_sentiment_extractor.extraction.color.constants import (
    EXPRESSION_SUPPRESSION_RULES,
    SEMANTIC_CONFLICTS,
)
from color_sentiment_extractor.extraction.color.vocab import get_all_webcolor_names
from color_sentiment_extractor.extraction.general.token import (
    normalize_token,
    recover_base,
)
from color_sentiment_extractor.extraction.general.utils import load_config

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Config (validated/raw) ───────────────────────────────────────────────────
_CONTEXT_MAP = load_config("expression_context_rules", mode="validated_dict")
_EXPRESSION_MAP_NORM = load_config("expression_definition", mode="validated_dict")
_EXPRESSION_MAP_RAW = load_config("expression_definition", mode="raw")

# ── Local helpers ────────────────────────────────────────────────────────────
def _norm(s):
    """Normalize token while keeping hyphens."""
    return normalize_token(s, keep_hyphens=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Cached vocab accessors (no “hard imports”)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_known_modifiers() -> frozenset[str]:
    return frozenset(load_config("known_modifiers", mode="set"))


@lru_cache(maxsize=1)
def _get_known_tones() -> frozenset[str]:
    return frozenset(load_config("known_tones", mode="set"))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Trigger Token Utilities
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_all_trigger_tokens() -> dict[str, list[str]]:
    """
    Does: Build map of expression → flattened tokens (aliases + modifiers) from *raw* defs.
    Returns: Dict[expression → list of tokens].
    """
    expression_map = _EXPRESSION_MAP_RAW
    trigger_map: dict[str, list[str]] = {}

    for expr, rules in expression_map.items():
        mods = rules.get("modifiers", []) or []
        aliases = rules.get("aliases", []) or []
        tokens = list({*mods, *aliases})
        if tokens:
            trigger_map[expr] = tokens
    return trigger_map


def get_all_alias_tokens(expression_map: dict[str, dict[str, list[str]]]) -> set[str]:
    """
    Does: Collect all normalized aliases + modifiers from an expression_map.
    Returns: Set of normalized tokens.
    """
    out: set[str] = set()
    for entry in expression_map.values():
        for a in entry.get("aliases", []) or []:
            out.add(_norm(a))
        for m in entry.get("modifiers", []) or []:
            out.add(_norm(m))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3) Normalized alias / modifier indexes (perf helpers)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_normalized_alias_map() -> dict[str, list[str]]:
    """
    Does: Build normalized alias → list(expressions) from definitions.
    Returns: Dict[alias -> [expr, ...]].
    """
    alias_map: dict[str, list[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        for alias in data.get("aliases", []) or []:
            norm = _norm(alias)
            alias_map.setdefault(norm, []).append(expr)
    return alias_map


@lru_cache(maxsize=1)
def _modifier_to_exprs() -> dict[str, set[str]]:
    """
    Does: Build normalized modifier → set(expressions) index.
    Returns: Dict[modifier -> {expr, ...}].
    """
    idx: dict[str, set[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        for m in data.get("modifiers", []) or []:
            nm = _norm(m)
            idx.setdefault(nm, set()).add(expr)
    return idx


@lru_cache(maxsize=1)
def _expr_norm_to_raw() -> dict[str, str]:
    """Does: Map normalized expression key → raw key for user-facing results."""
    return {_norm(e): e for e in _EXPRESSION_MAP_NORM.keys()}


@lru_cache(maxsize=1)
def _expr_to_valid_tones() -> dict[str, list[str]]:
    """Does: Precompute expression → list of valid (normalized) tones (present in known_tones)."""
    kt = _get_known_tones()
    out: dict[str, list[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        # Dans tes configs, ces champs portent des tons (même si nommés "modifiers").
        mods_or_tones = [_norm(m) for m in (data.get("modifiers") or [])]
        valid = [m for m in mods_or_tones if m in kt]
        if valid:
            out[expr] = valid
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4) Expression Matching Logic
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _flattened_items_for_exact_match() -> list[str]:
    """
    Does: Cache a flattened list of aliases+modifiers from normalized map
          for exact-span scanning (sorted outside if needed).
    """
    items: set[str] = set()
    for v in _EXPRESSION_MAP_NORM.values():
        items.update(v.get("aliases", []) or [])
        items.update(v.get("modifiers", []) or [])
    return list(items)


def extract_exact_alias_tokens(
    text: str,
    expression_map: dict[str, dict[str, list[str]]],
) -> list[str]:
    """
    Does: Find exact alias/modifier spans, preferring longer multi-word forms.
    Returns: Sorted list of normalized matches (no subsumed shorter forms).
    """
    text_lower = _norm(text)
    items = _flattened_items_for_exact_match()
    pairs = [(s, _norm(s)) for s in items]

    # Trier par longueur décroissante du *norm* pour éviter les sous-inclusions
    taken: set[str] = set()
    out: list[str] = []
    for _, normed in sorted(pairs, key=lambda x: -len(x[1])):
        # Frontières robustes : évite les matches au sein de mots ou séparateurs type "/"
        pattern = rf"(?<![\w/]){re.escape(normed)}(?![\w/])"
        if re.search(pattern, text_lower) and not any(normed in t and normed != t for t in taken):
            out.append(normed)
            taken.add(normed)
    return sorted(set(out))


@lru_cache(maxsize=512)
def get_matching_expression_tags_cached(text: str, debug: bool = False) -> set[str]:
    """
    Does: Cached wrapper for expression tag matching.
    Returns: Set of raw expression names.
    """
    return _get_matching_expression_tags(text, _EXPRESSION_MAP_NORM, debug)


def _get_matching_expression_tags(
    text: str,
    expression_map: dict[str, dict[str, list[str]]],
    debug: bool = False,
) -> set[str]:
    """
    Does: Delegate alias/expr matching to shared matcher, then map aliases→expressions.
    Returns: Set of raw expression names.
    """
    hits = match_expression_aliases(text, expression_map)  # set[str]
    alias_map = _get_normalized_alias_map()
    expr_norm_to_raw = _expr_norm_to_raw()

    out: set[str] = set()
    for item in hits:
        ni = _norm(item)
        if ni in expr_norm_to_raw:
            out.add(expr_norm_to_raw[ni])
        if ni in alias_map:
            out.update(alias_map[ni])

    if debug:
        log.debug("[match] input='%s' → exprs=%s", text, sorted(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5) Expression → Tone Mapping
# ─────────────────────────────────────────────────────────────────────────────


def map_expressions_to_tones(text: str, debug: bool = False) -> dict[str, list[str]]:
    """
    Does: Map matched expressions → tone lists using context + suppression rules.
    Returns: Dict[expression → list of valid tone modifiers].
    """
    results: dict[str, list[str]] = {}

    start_total = time.perf_counter()
    text_lower = _norm(text)
    tokens = text_lower.split()

    t1 = time.perf_counter()
    raw_matched = get_matching_expression_tags_cached(text, debug=debug)
    t2 = time.perf_counter()

    promoted = apply_expression_context_rules(tokens, raw_matched, _CONTEXT_MAP)
    t3 = time.perf_counter()

    if promoted:
        raw_matched |= promoted
    longest_matched_aliases = apply_expression_suppression_rules(raw_matched)
    t4 = time.perf_counter()

    expr_to_valid = _expr_to_valid_tones()
    for expr in longest_matched_aliases:
        if expr in expr_to_valid:
            results[expr] = expr_to_valid[expr]

    end_total = time.perf_counter()

    if debug:
        log.debug(
            "⏱️ map_expressions_to_tones():\n"
            "  Tokenize + normalize:            %.4fs\n"
            "  Match expressions (cached):      %.4fs\n"
            "  Apply context rules:             %.4fs\n"
            "  Suppression rules + filtering:   %.4fs\n"
            "  Final loop + tone mapping:       %.4fs\n"
            "  TOTAL:                           %.4fs",
            (t1 - start_total),
            (t2 - t1),
            (t3 - t2),
            (t4 - t3),
            (end_total - t4),
            (end_total - start_total),
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6) Modifiers injection (alias/expr hits + token recovery)
# ─────────────────────────────────────────────────────────────────────────────


def _iter_token_strings(tokens: Iterable) -> list[str]:
    """
    Does: Normalize input tokens: accepts a list of strings OR spaCy tokens/spans/docs.
    Returns: List of raw token strings (lowercasing is applied later where needed).
    """
    out: list[str] = []
    for t in tokens:
        if hasattr(t, "text"):
            out.append(str(t.text))
        else:
            out.append(str(t))
    return out


def _apply_suffix_ties(candidates: set[str]) -> set[str]:
    """
    Does: Resolve root vs '-y' ties by keeping the '-y' form when both exist.
    Returns: Pruned set of modifiers.
    """
    mods = set(candidates)
    for m in [m for m in mods if m.endswith("y")]:
        root = m[:-1]
        if root in mods:
            mods.discard(root)
    return mods


def _apply_semantic_conflicts_local(candidates: set[str]) -> set[str]:
    """
    Does: Resolve semantic conflicts using SEMANTIC_CONFLICTS from config.
          Supports two shapes:
            1) dict: {canonical: [variants...]}
            2) iterable of groups: {('airy','fairy'), ('matte','shiny'), ...}
    Tie-break rule (deterministic): prefer canonical if present, else shortest, then lexicographic.
    Returns: Pruned set of modifiers.
    """
    conf = SEMANTIC_CONFLICTS

    if not conf or not candidates:
        return candidates

    def n(s: str) -> str:
        return _norm(s)

    if isinstance(conf, dict):
        canon_of: dict[str, str] = {}
        for canon, variants in conf.items():
            c = n(canon)
            canon_of[c] = c
            for v in variants or []:
                canon_of[n(v)] = c

        groups: dict[str, list[str]] = {}
        for m in candidates:
            key = canon_of.get(n(m), n(m))
            groups.setdefault(key, []).append(m)

        keep: set[str] = set()
        for canon, forms in groups.items():
            preferred = next(
                (f for f in forms if n(f) == canon),
                sorted(forms, key=lambda s: (len(s), n(s)))[0],
            )
            keep.add(preferred)
        return keep

    # Iterable de groupes
    try:
        ngroups = []
        for g in conf:
            if isinstance(g, (tuple, list, set, frozenset)):
                grp = {n(x) for x in g if isinstance(x, str)}
                if len(grp) >= 2:
                    ngroups.append(grp)

        if not ngroups:
            return candidates

        survivors: set[str] = set(candidates)
        cand_norm_map = {m: n(m) for m in candidates}

        for grp in ngroups:
            present = [m for m in candidates if cand_norm_map[m] in grp]
            if len(present) >= 2:
                winner = sorted(
                    present,
                    key=lambda s: (len(cand_norm_map[s]), cand_norm_map[s]),
                )[0]
                for m in present:
                    if m != winner:
                        survivors.discard(m)
        return survivors
    except Exception:
        return candidates


def _inject_expression_modifiers(
    tokens: Iterable,
    known_modifiers: set[str] | None = None,
    known_tones: set[str] | None = None,
    expression_map: dict[str, dict[str, list[str]]] | None = None,
    debug: bool = False,
) -> list[str]:
    """
    Does: Inject modifiers from alias/expr hits + token→base recovery (with fuzzy fallback).
    Accepts: list of strings OR spaCy Doc/Span (tokens can have .text / .lower_).
    Returns: Sorted list of normalized modifiers.
    """
    if known_modifiers is None:
        known_modifiers = set(_get_known_modifiers())
    if known_tones is None:
        known_tones = set(_get_known_tones())
    if expression_map is None:
        expression_map = _EXPRESSION_MAP_NORM

    tok_strs = _iter_token_strings(tokens)
    tok_lc = [s.lower() for s in tok_strs]
    raw_text = " ".join(tok_strs)

    if debug:
        log.debug("[START] _inject_expression_modifiers")
        log.debug("Tokens: %s", tok_strs)
        log.debug("Raw: '%s'", raw_text)

    # STEP 1 — alias/expr hits (exact + fuzzy via shared matcher)
    matched_exprs: set[str] = set(
        _get_matching_expression_tags(raw_text, expression_map, debug=False)
    )

    # STEP 2 — token-scoped: alias→expr, else base→expr (unified fuzzy flag)
    alias_map = _get_normalized_alias_map()
    norm_known_mods = {_norm(m) for m in known_modifiers}
    mod_to_expr = _modifier_to_exprs()

    for t in tok_lc:
        tn = _norm(t)

        # Direct alias → expr(s)
        if tn in alias_map:
            matched_exprs.update(alias_map[tn])
            if debug:
                log.debug("  Token '%s' is alias → exprs %s", t, sorted(alias_map[tn]))
            continue

        # Base recovery (strict + fuzzy fallback controlled by one arg name)
        base = recover_base(
            t,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            debug=False,
            fuzzy_fallback=True,
        )

        if base:
            bn = _norm(base)
            for expr in mod_to_expr.get(bn, ()):
                matched_exprs.add(expr)
                if debug:
                    log.debug("  Token '%s' → base '%s' → expr '%s'", t, base, expr)

    # STEP 3 — collect modifiers from matched expressions + raw tokens that are modifiers
    collected: set[str] = set()
    for expr in matched_exprs:
        for m in expression_map.get(expr, {}).get("modifiers", []) or []:
            nm = _norm(m)
            if nm in norm_known_mods:
                collected.add(nm)

    for t in tok_lc:
        nt = _norm(t)
        if nt in norm_known_mods:
            collected.add(nt)

    # Conflits puis ties
    before = set(collected)
    collected = _apply_semantic_conflicts_local(collected)
    collected = _apply_suffix_ties(collected)

    if debug:
        removed = sorted(before - collected)
        if removed:
            log.debug("  Removed by ties/conflicts: %s", removed)
        log.debug("[END] Final injected modifiers: %s", sorted(collected))

    return sorted(collected)


# ─────────────────────────────────────────────────────────────────────────────
# 7) Contextual Promotion and Suppression
# ─────────────────────────────────────────────────────────────────────────────


def apply_expression_context_rules(
    tokens: list[str],
    matched_expressions: set[str],
    context_map: dict[str, list[dict[str, list[str]]]],
) -> set[str]:
    """
    Does: Promote expressions not directly matched when required tokens + clues co-occur.
    Returns: Set of expression names to promote.
    """
    token_set = set(tokens)
    promotions: set[str] = set()
    unmatched = set(context_map.keys()) - set(matched_expressions)

    for expression in unmatched:
        for rule in context_map[expression]:
            required = rule.get("require_tokens", []) or []
            clues = rule.get("context_clues", []) or []
            if any(tok in token_set for tok in required) and any(
                clue in token_set for clue in clues
            ):
                promotions.add(expression)
                break

    return promotions


def apply_expression_suppression_rules(matched: set[str]) -> set[str]:
    """
    Does: Apply suppression rules to remove lower-priority expressions if a dominant exists.
    Returns: Filtered set of expression tags.
    """
    return matched - {
        expr for dominant in matched for expr in EXPRESSION_SUPPRESSION_RULES.get(dominant, set())
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8) Glued Token Vocabulary & Expression Input Normalization
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_glued_token_vocabulary() -> frozenset[str]:
    known_modifiers = _get_known_modifiers()
    known_tones = _get_known_tones()
    webcolors = get_all_webcolor_names()
    return frozenset(
        {
            *(_norm(x) for x in known_tones),
            *(_norm(x) for x in known_modifiers),
            *(_norm(x) for x in webcolors),
        }
    )
