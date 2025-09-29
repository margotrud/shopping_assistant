"""
expression_helpers.py
=====================

Helpers to extract trigger vocabularies and expression-related token sets.

Used By:
--------
- Expression matching (contextual tone detection)
- Compound token splitting (glued token fallback)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────
# 1) Imports & Config
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Set, Iterable
import re
from functools import lru_cache
import time

from color_sentiment_extractor.extraction.color.constants import (
    EXPRESSION_SUPPRESSION_RULES,
    SEMANTIC_CONFLICTS,
)
from color_sentiment_extractor.extraction.general.utils import load_config
from color_sentiment_extractor.extraction.general.token import (normalize_token,
recover_base)
from color_sentiment_extractor.extraction.general.fuzzy import (
    match_expression_aliases,
)


# Chargements config (versions “validation” et “raw”)
_CONTEXT_MAP = load_config("expression_context_rules", mode="validated_dict")
_EXPRESSION_MAP_NORM = load_config("expression_definition", mode="validated_dict")
_EXPRESSION_MAP_RAW = load_config("expression_definition", mode="raw")


# ──────────────────────────────────────────────────────────────
# 2) Cached vocab accessors (no “hard imports”)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_known_modifiers() -> Set[str]:
    return frozenset(load_config("known_modifiers", mode="set"))

@lru_cache(maxsize=1)
def _get_known_tones() -> Set[str]:
    return frozenset(load_config("known_tones", mode="set"))

@lru_cache(maxsize=1)
def _get_all_webcolor_names() -> Set[str]:
    # Nom de clé à adapter à ta config si besoin (“webcolors_all” / “all_webcolor_names”, etc.)
    # On tente plusieurs clés courantes pour robustesse.
    for key in ("webcolors_all", "all_webcolor_names", "webcolor_names"):
        try:
            vals = load_config(key, mode="set")
            if vals:
                return frozenset(vals)
        except Exception:
            pass
    return frozenset()


# ──────────────────────────────────────────────────────────────
# 3) Trigger Token Utilities
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_all_trigger_tokens() -> Dict[str, List[str]]:
    """
    Does: Build map of expression → flattened tokens (aliases + modifiers).
    Returns: Dict[expression → list of tokens].
    """
    expression_map = _EXPRESSION_MAP_RAW
    trigger_map: Dict[str, List[str]] = {}

    for expr, rules in expression_map.items():
        mods = rules.get("modifiers", []) or []
        aliases = rules.get("aliases", []) or []
        tokens = list({*mods, *aliases})
        if tokens:
            trigger_map[expr] = tokens
    return trigger_map


def get_all_alias_tokens(expression_map: Dict) -> Set[str]:
    """
    Does: Collect all normalized aliases + modifiers from an expression_map.
    Returns: Set of normalized tokens.
    """
    out: Set[str] = set()
    for entry in expression_map.values():
        for a in entry.get("aliases", []) or []:
            out.add(normalize_token(a, keep_hyphens=True))
        for m in entry.get("modifiers", []) or []:
            out.add(normalize_token(m, keep_hyphens=True))
    return out


# ──────────────────────────────────────────────────────────────
# 4) Normalized alias / modifier indexes (perf helpers)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_normalized_alias_map() -> Dict[str, List[str]]:
    """
    Does: Build normalized alias → list(expressions) from definitions.
    Returns: Dict[alias -> [expr, ...]].
    """
    alias_map: Dict[str, List[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        for alias in data.get("aliases", []) or []:
            norm = normalize_token(alias, keep_hyphens=True)
            alias_map.setdefault(norm, []).append(expr)
    return alias_map


@lru_cache(maxsize=1)
def _modifier_to_exprs() -> Dict[str, Set[str]]:
    """
    Does: Build normalized modifier → set(expressions) index.
    Returns: Dict[modifier -> {expr, ...}].
    """
    idx: Dict[str, Set[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        for m in data.get("modifiers", []) or []:
            nm = normalize_token(m, keep_hyphens=True)
            idx.setdefault(nm, set()).add(expr)
    return idx


@lru_cache(maxsize=1)
def _expr_norm_to_raw() -> Dict[str, str]:
    """
    Does: Map normalized expression key → raw key for user-facing results.
    """
    return {normalize_token(e, keep_hyphens=True): e for e in _EXPRESSION_MAP_NORM.keys()}


@lru_cache(maxsize=1)
def _expr_to_valid_tones() -> Dict[str, List[str]]:
    """
    Does: Precompute expression → list of valid (normalized) tones (present in known_tones).
    """
    kt = _get_known_tones()
    out: Dict[str, List[str]] = {}
    for expr, data in _EXPRESSION_MAP_NORM.items():
        mods = [normalize_token(m, keep_hyphens=True)
                for m in (data.get("modifiers") or [])]
        valid = [m for m in mods if m in kt]
        if valid:
            out[expr] = valid
    return out


# ──────────────────────────────────────────────────────────────
# 5) Expression Matching Logic
# ──────────────────────────────────────────────────────────────

def extract_exact_alias_tokens(text: str, expression_map: Dict) -> List[str]:
    """
    Does: Find exact alias/modifier spans, preferring longer multi-word forms.
    Returns: Sorted list of normalized matches (no subsumed shorter forms).
    """
    text_lower = normalize_token(text, keep_hyphens=True)
    items = set()
    for v in expression_map.values():
        items.update(v.get("aliases", []) or [])
        items.update(v.get("modifiers", []) or [])
    pairs = [(s, normalize_token(s, keep_hyphens=True)) for s in items]

    taken, out = set(), []
    for _, norm in sorted(pairs, key=lambda x: -len(x[1])):
        # Frontières robustes (pas seulement \b, gère hyphens/accents)
        pattern = rf"(?<!\w){re.escape(norm)}(?!\w)"
        if re.search(pattern, text_lower) and not any(
            norm in t and norm != t for t in taken
        ):
            out.append(norm)
            taken.add(norm)
    return sorted(set(out))


@lru_cache(maxsize=512)
def get_matching_expression_tags_cached(text: str, debug: bool = False) -> Set[str]:
    """
    Does: Cached wrapper for expression tag matching.
    Returns: Set of raw expression names.
    """
    return _get_matching_expression_tags(text, _EXPRESSION_MAP_NORM, debug)


def _get_matching_expression_tags(text: str, expression_map: dict, debug: bool = False) -> Set[str]:
    """
    Does: Delegate alias/expr matching to shared matcher, then map aliases→expressions.
    Returns: Set of raw expression names.
    """
    hits = match_expression_aliases(text, expression_map)  # set[str]
    alias_map = _get_normalized_alias_map()
    expr_norm_to_raw = _expr_norm_to_raw()

    out: Set[str] = set()
    for item in hits:
        ni = normalize_token(item, keep_hyphens=True)
        if ni in expr_norm_to_raw:
            out.add(expr_norm_to_raw[ni])
        if ni in alias_map:
            out.update(alias_map[ni])

    if debug:
        print(f"[match] input='{text}' → exprs={sorted(out)}")
    return out


# ──────────────────────────────────────────────────────────────
# 6) Expression → Tone Mapping
# ──────────────────────────────────────────────────────────────

def map_expressions_to_tones(text: str, debug: bool = False) -> Dict[str, List[str]]:
    """
    Does: Map matched expressions → tone lists using context + suppression rules.
    Returns: Dict[expression → list of valid tone modifiers].
    """
    results: Dict[str, List[str]] = {}

    start_total = time.perf_counter()
    text_lower = normalize_token(text, keep_hyphens=True)
    tokens = text_lower.split()

    t1 = time.perf_counter()
    raw_matched = get_matching_expression_tags_cached(text, debug=debug)
    t2 = time.perf_counter()

    promoted = apply_expression_context_rules(tokens, raw_matched, _CONTEXT_MAP)
    t3 = time.perf_counter()

    raw_matched |= promoted
    longest_matched_aliases = apply_expression_suppression_rules(raw_matched)
    t4 = time.perf_counter()

    expr_to_valid = _expr_to_valid_tones()
    for expr in longest_matched_aliases:
        if expr in expr_to_valid:
            results[expr] = expr_to_valid[expr]

    end_total = time.perf_counter()

    if debug:
        print(f"\n⏱️ map_expressions_to_tones():")
        print(f"  Tokenize + normalize:            {(t1 - start_total):.4f}s")
        print(f"  Match expressions (cached):      {(t2 - t1):.4f}s")
        print(f"  Apply context rules:             {(t3 - t2):.4f}s")
        print(f"  Suppression rules + filtering:   {(t4 - t3):.4f}s")
        print(f"  Final loop + tone mapping:       {(end_total - t4):.4f}s")
        print(f"  TOTAL:                           {(end_total - start_total):.4f}s\n")

    return results


# ──────────────────────────────────────────────────────────────
# 7) Modifiers injection (alias/expr hits + token recovery)
# ──────────────────────────────────────────────────────────────

def _iter_token_strings(tokens: Iterable) -> List[str]:
    """
    Normalize input tokens: accepts a list of strings OR spaCy tokens/spans/docs.
    Returns a list of raw token strings (lowercased version available as needed).
    """
    out: List[str] = []
    for t in tokens:
        if hasattr(t, "text"):
            out.append(str(t.text))
        else:
            out.append(str(t))
    return out


def _apply_suffix_ties(candidates: Set[str]) -> Set[str]:
    """
    Does: Resolve root vs '-y' ties by keeping the '-y' form when both exist.
    Returns: Pruned set of modifiers.
    """
    mods = set(candidates)
    for m in list(mods):
        if m.endswith("y"):
            root = m[:-1]
            if root in mods:
                mods.discard(root)
    return mods


def _apply_semantic_conflicts_local(candidates: Set[str]) -> Set[str]:
    """
    Does: Resolve semantic conflicts using SEMANTIC_CONFLICTS from config.
          Supports two shapes:
            1) dict: {canonical: [variants...]}
            2) set/list of groups: {('airy','fairy'), ('matte','shiny'), ...}
    Returns: Pruned set of modifiers (deterministic winner per conflict group).
    """
    conf = SEMANTIC_CONFLICTS

    if not conf or not candidates:
        return candidates

    def norm(s: str) -> str:
        return normalize_token(s, keep_hyphens=True)

    if isinstance(conf, dict):
        canon_of = {}
        for canon, variants in conf.items():
            c = norm(canon)
            canon_of[c] = c
            for v in (variants or []):
                canon_of[norm(v)] = c

        groups: Dict[str, List[str]] = {}
        for m in candidates:
            key = canon_of.get(norm(m), norm(m))
            groups.setdefault(key, []).append(m)

        keep = set()
        for canon, forms in groups.items():
            preferred = next(
                (f for f in forms if norm(f) == canon),
                sorted(forms, key=lambda s: (len(s), s))[0],
            )
            keep.add(preferred)
        return keep

    # Iterable of groups (pairs/sets/tuples)
    try:
        ngroups = []
        for g in conf:
            if isinstance(g, (tuple, list, set, frozenset)):
                grp = {norm(x) for x in g if isinstance(x, str)}
                if len(grp) >= 2:
                    ngroups.append(grp)

        if not ngroups:
            return candidates

        survivors = set(candidates)
        cand_norm_map = {m: norm(m) for m in candidates}

        for grp in ngroups:
            present = [m for m in candidates if cand_norm_map[m] in grp]
            if len(present) >= 2:
                winner = sorted(present, key=lambda s: (len(cand_norm_map[s]), cand_norm_map[s]))[0]
                for m in present:
                    if m != winner:
                        survivors.discard(m)
        return survivors
    except Exception:
        return candidates


def _inject_expression_modifiers(
    tokens: Iterable,
    known_modifiers: Set[str] | None = None,
    known_tones: Set[str] | None = None,
    expression_map: dict | None = None,
    debug: bool = False,
) -> List[str]:
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
        print("\n[START] _inject_expression_modifiers")
        print(f"Tokens: {tok_strs}")
        print(f"Raw: '{raw_text}'")

    # STEP 1 — alias/expr hits (exact + fuzzy via shared matcher)
    matched_exprs = set(_get_matching_expression_tags(raw_text, expression_map, debug=False))

    # STEP 2 — token-scoped: alias→expr, else base→expr
    alias_map = _get_normalized_alias_map()
    norm_known_mods = {normalize_token(m, keep_hyphens=True) for m in known_modifiers}
    mod_to_expr = _modifier_to_exprs()

    for t in tok_lc:
        tn = normalize_token(t, keep_hyphens=True)

        # Direct alias → expr(s)
        if tn in alias_map:
            matched_exprs.update(alias_map[tn])
            if debug:
                print(f"  Token '{t}' is alias → exprs {sorted(alias_map[tn])}")
            continue

        # Base recovery (strict + optional fuzzy fallback)
        base = recover_base(
            t,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            debug=False,
            fuzzy_fallback=True,
        )

        if not base:
            cand = recover_base(t, allow_fuzzy=True)
            if cand and normalize_token(cand, keep_hyphens=True) in norm_known_mods:
                base = cand

        if base:
            bn = normalize_token(base, keep_hyphens=True)
            for expr in mod_to_expr.get(bn, ()):
                matched_exprs.add(expr)
                if debug:
                    print(f"  Token '{t}' → base '{base}' → expr '{expr}'")

    # STEP 3 — collect modifiers from matched expressions + raw tokens that are modifiers
    collected = set()
    for expr in matched_exprs:
        for m in expression_map.get(expr, {}).get("modifiers", []) or []:
            nm = normalize_token(m, keep_hyphens=True)
            if nm in norm_known_mods:
                collected.add(nm)

    for t in tok_lc:
        nt = normalize_token(t, keep_hyphens=True)
        if nt in norm_known_mods:
            collected.add(nt)

    # Conflicts puis ties (ou inverse selon ta règle métier)
    before = set(collected)
    collected = _apply_semantic_conflicts_local(collected)
    collected = _apply_suffix_ties(collected)

    if debug:
        removed = sorted(before - collected)
        if removed:
            print(f"  Removed by ties/conflicts: {removed}")
        print("[END] Final injected modifiers:", sorted(collected))

    return sorted(collected)


# ──────────────────────────────────────────────────────────────
# 8) Contextual Promotion and Suppression
# ──────────────────────────────────────────────────────────────

def apply_expression_context_rules(
    tokens: List[str],
    matched_expressions: Set[str],
    context_map: Dict[str, List[Dict[str, List[str]]]],
) -> Set[str]:
    """
    Does: Promote expressions not directly matched when required tokens + clues co-occur.
    Returns: Set of expression names to promote.
    """
    token_set = set(tokens)
    promotions = set()
    unmatched = set(context_map.keys()) - set(matched_expressions)

    for expression in unmatched:
        for rule in context_map[expression]:
            required = rule.get("require_tokens", []) or []
            clues = rule.get("context_clues", []) or []
            if any(tok in token_set for tok in required) and any(clue in token_set for clue in clues):
                promotions.add(expression)
                break

    return promotions


def apply_expression_suppression_rules(matched: Set[str]) -> Set[str]:
    """
    Does: Apply suppression rules to remove lower-priority expressions if a dominant exists.
    Returns: Filtered set of expression tags.
    """
    return matched - {
        expr
        for dominant in matched
        for expr in EXPRESSION_SUPPRESSION_RULES.get(dominant, set())
    }


# ──────────────────────────────────────────────────────────────
# 9) Glued Token Vocabulary & Expression Input Normalization
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_glued_token_vocabulary() -> Set[str]:
    """
    Does: Return full vocab for glued-token splitting (tones + modifiers + webcolor names).
    Returns: Set of tokens (normalized, hyphens preserved).
    """
    known_modifiers = _get_known_modifiers()
    known_tones = _get_known_tones()
    webcolors = _get_all_webcolor_names()
    # Normaliser pour cohérence
    norm = lambda s: normalize_token(s, keep_hyphens=True)
    return frozenset({*(norm(x) for x in known_tones),
                      *(norm(x) for x in known_modifiers),
                      *(norm(x) for x in webcolors)})
