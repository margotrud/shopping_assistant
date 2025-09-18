"""
expression_helpers.py
=====================

Helpers to extract trigger vocabularies and expression-related token sets.

Used By:
--------
- Expression matching (contextual tone detection)
- Compound token splitting (glued token fallback)
"""

# ──────────────────────────────────────────────────────────────
# 1. Imports and Config
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Set
import re
from functools import lru_cache
import time
from extraction.color.constants import EXPRESSION_SUPPRESSION_RULES, SEMANTIC_CONFLICTS
from extraction.general.utils.load_config import load_config
from extraction.color.vocab import known_tones, all_webcolor_names
from extraction.general.token.normalize import normalize_token
from extraction.general.fuzzy.expression_match import match_expression_aliases
from extraction.general.token.base_recovery import recover_base
_CONTEXT_MAP = load_config("expression_context_rules", mode="validated_dict")
_EXPRESSION_MAP_CACHED = load_config("expression_definition", mode="validated_dict")
_EXPRESSION_MAP_RAW = load_config("expression_definition", mode="raw")
_KNOWN_TONES = known_tones


# ──────────────────────────────────────────────────────────────
# 2. Trigger Token Utilities
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
        tokens = list(set(mods + aliases))
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
# 3. Expression Matching Logic
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
    for raw, norm in sorted(pairs, key=lambda x: -len(x[1])):
        if re.search(rf"\b{re.escape(norm)}\b", text_lower) and not any(
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
    return _get_matching_expression_tags(text, _EXPRESSION_MAP_CACHED, debug)


@lru_cache(maxsize=1)
def _get_normalized_alias_map() -> Dict[str, List[str]]:
    """
    Does: Build normalized alias → list(expressions) from cached definitions.
    Returns: Dict[alias -> [expr, ...]].
    """
    alias_map: Dict[str, List[str]] = {}
    for expr, data in _EXPRESSION_MAP_CACHED.items():
        for alias in data.get("aliases", []) or []:
            norm = normalize_token(alias, keep_hyphens=True)
            alias_map.setdefault(norm, []).append(expr)
    return alias_map


def _get_matching_expression_tags(text: str, expression_map: dict, debug: bool = False) -> Set[str]:
    """
    Does: Delegate alias/expr matching to shared matcher, then map aliases→expressions.
    Returns: Set of raw expression names.
    """
    hits = match_expression_aliases(text, expression_map)  # set[str]
    alias_map = _get_normalized_alias_map()

    # Build normalized expr-key → raw-key map so we return raw keys
    expr_norm_to_raw = {normalize_token(e, keep_hyphens=True): e for e in expression_map.keys()}

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
# 4. Expression → Tone Mapping
# ──────────────────────────────────────────────────────────────

def map_expressions_to_tones(text: str, debug: bool = True) -> Dict[str, List[str]]:
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

    for expr, data in _EXPRESSION_MAP_CACHED.items():
        aliases = data.get("aliases", []) or []
        if expr not in longest_matched_aliases and not any(
            alias in longest_matched_aliases for alias in aliases
        ):
            continue
        modifiers = data.get("modifiers", []) or []
        valid_tones = [m for m in modifiers if m in _KNOWN_TONES]
        if valid_tones:
            results[expr] = valid_tones

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


def _apply_suffix_ties(candidates: Set[str], known_modifiers: Set[str]) -> Set[str]:
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

    # Fast path: nothing to do
    if not conf or not candidates:
        return candidates

    # Normalize helper
    def norm(s: str) -> str:
        return normalize_token(s, keep_hyphens=True)

    # Shape 1: dict(canonical -> variants)
    if isinstance(conf, dict):
        canon_of = {}
        for canon, variants in conf.items():
            c = norm(canon)
            # map canonical to itself so it wins if present
            canon_of[c] = c
            for v in (variants or []):
                canon_of[norm(v)] = c

        groups: Dict[str, List[str]] = {}
        for m in candidates:
            key = canon_of.get(norm(m), norm(m))
            groups.setdefault(key, []).append(m)

        keep = set()
        for canon, forms in groups.items():
            # Prefer the form that equals canonical after normalization; else shortest; else alpha.
            preferred = next((f for f in forms if norm(f) == canon),
                             sorted(forms, key=lambda s: (len(s), s))[0])
            keep.add(preferred)
        return keep

    # Shape 2: iterable of conflict groups (pairs/sets/tuples)
    try:
        # Build list of normalized groups (only groups with size >= 2 matter)
        groups = []
        for g in conf:
            if isinstance(g, (tuple, list, set, frozenset)):
                grp = {norm(x) for x in g if isinstance(x, str)}
                if len(grp) >= 2:
                    groups.append(grp)

        if not groups:
            return candidates

        survivors = set(candidates)
        cand_norm_map = {m: norm(m) for m in candidates}

        for grp in groups:
            present = [m for m in candidates if cand_norm_map[m] in grp]
            if len(present) >= 2:
                # Deterministic winner: shortest after norm; then alphabetical
                winner = sorted(present, key=lambda s: (len(cand_norm_map[s]), cand_norm_map[s]))[0]
                for m in present:
                    if m != winner:
                        survivors.discard(m)

        return survivors
    except Exception:
        # If anything goes weird, fail open (no pruning)
        return candidates


def _inject_expression_modifiers(
    tokens,
    known_modifiers: Set[str],
    known_tones: Set[str],
    expression_map: dict,
    debug: bool = False,
) -> List[str]:
    """
    Does: Inject modifiers from alias/expr hits + token→base recovery (with fuzzy fallback).
    Returns: Sorted list of normalized modifiers.
    """
    tok_strs = [t.text for t in tokens]
    tok_lc = [t.lower_ for t in tokens]
    raw_text = " ".join(tok_strs)  # ✅ FIX: works for lists or spaCy Doc/Span

    if debug:
        print("\n[START] _inject_expression_modifiers")
        print(f"Tokens: {tok_strs}")
        print(f"Raw: '{raw_text}'")

    # STEP 1 — alias/expr hits (exact + fuzzy via shared matcher)
    matched_exprs = set(_get_matching_expression_tags(raw_text, expression_map, debug=False))

    # STEP 2 — token-scoped: alias→expr, else base→expr
    alias_map = _get_normalized_alias_map()
    norm_known_mods = {normalize_token(m, keep_hyphens=True) for m in known_modifiers}

    for t in tok_lc:
        tn = normalize_token(t, keep_hyphens=True)

        # Direct alias → expr(s)
        if tn in alias_map:
            matched_exprs.update(alias_map[tn])
            if debug:
                print(f"  Token '{t}' is alias → exprs {sorted(alias_map[tn])}")
            continue

        # Base recovery (strict)
        base = recover_base(
            t,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            debug=False,
            fuzzy_fallback=True,
        )

        # Fallback (fuzzy) only if strict failed; accept only modifiers
        if not base:
            cand = recover_base(t, allow_fuzzy=True)
            if cand and normalize_token(cand, keep_hyphens=True) in norm_known_mods:
                base = cand

        if base:
            bn = normalize_token(base, keep_hyphens=True)
            for expr, info in expression_map.items():
                mods = {normalize_token(m, keep_hyphens=True) for m in info.get("modifiers", []) or []}
                if bn in mods:
                    matched_exprs.add(expr)
                    if debug:
                        print(f"  Token '{t}' → base '{base}' → expr '{expr}'")

    # STEP 3 — collect modifiers from matched expressions + raw tokens that are modifiers
    collected = set()
    for expr in matched_exprs:
        for m in expression_map.get(expr, {}).get("modifiers", []) or []:
            if normalize_token(m, keep_hyphens=True) in norm_known_mods:
                collected.add(normalize_token(m, keep_hyphens=True))

    for t in tok_lc:
        if normalize_token(t, keep_hyphens=True) in norm_known_mods:
            collected.add(normalize_token(t, keep_hyphens=True))

    # Tie + conflict cleanup
    before = set(collected)
    collected = _apply_suffix_ties(collected, known_modifiers)
    collected = _apply_semantic_conflicts_local(collected)

    if debug:
        removed = sorted(before - collected)
        if removed:
            print(f"  Removed by ties/conflicts: {removed}")
        print("[END] Final injected modifiers:", sorted(collected))

    return sorted(collected)



# ──────────────────────────────────────────────────────────────
# 5. Contextual Promotion and Suppression
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
# 6. Glued Token Vocabulary & Expression Input Normalization
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_glued_token_vocabulary() -> Set[str]:
    """
    Does: Return full vocab for glued-token splitting (tones + modifiers + webcolor names).
    Returns: Set of tokens.
    """
    known_modifiers = load_config("known_modifiers", mode="set")
    return known_tones.union(known_modifiers).union(all_webcolor_names)


