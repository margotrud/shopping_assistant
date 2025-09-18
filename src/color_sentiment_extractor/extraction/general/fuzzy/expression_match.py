"""
expression_match.py
====================

Matches user input against predefined expression aliases.
Includes fuzzy fallback, multiword/singleword detection,
duplicate suppression, and embedded conflict resolution.
"""


from functools import lru_cache
from fuzzywuzzy import fuzz

from color_sentiment_extractor.extraction.general.fuzzy.alias_validation import _handle_multiword_alias, is_valid_singleword_alias
from color_sentiment_extractor.extraction.general.fuzzy.scoring import fuzzy_token_overlap_count
from color_sentiment_extractor.extraction.general.utils.load_config import load_config
from color_sentiment_extractor.extraction.general.token.normalize import get_tokens_and_counts

# ─────────────────────────────────────────────────────────────
# 1. Core alias acceptance logic
# ─────────────────────────────────────────────────────────────

def should_accept_alias_match(alias, input_text, tokens, matched_aliases=None, debug=False):
    """
    Does: Determines whether an alias (single or multi-word) should be accepted based on:
    - Direct containment
    - Avoiding duplicates
    - Delegating to single/multiword handlers
    Returns: Boolean flag indicating acceptance
    """
    alias = alias.strip().lower()
    input_text_lc = input_text.lower().strip()
    matched_aliases = matched_aliases or set()
    is_multiword = " " in alias

    if alias in input_text_lc:
        if debug:
            print(f"[✅ DIRECT CONTAINS MATCH] alias '{alias}' found inside input → accepting")
        return True

    for matched in matched_aliases:
        if alias in matched and matched != alias:
            if debug:
                print(f"[⛔ SKIP] '{alias}' is part of already matched multiword: '{matched}'")
            return False

    return (
        _handle_multiword_alias(alias, input_text, debug=False)
        if is_multiword
        else is_valid_singleword_alias(alias, input_text, tokens, matched_aliases, debug=False)
    )


# ─────────────────────────────────────────────────────────────
# 2. Main expression alias matcher
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1000)
def cached_match_expression_aliases(input_text: str) -> set[str]:
    expression_def = load_config("expression_definition", mode="validated_dict")
    return match_expression_aliases(input_text, expression_def)

def match_expression_aliases(input_text, expression_map, debug=False):
    """
    Does: Matches expression aliases (multi and single-word) and modifiers from expression_map.
    - Alias matches take priority and can run in exclusive mode.
    - Modifiers matched via tokens or fuzzy matching.
    - Known tone tokens can directly trigger matches if included in modifiers.
    Returns: Set of canonical expressions matched from user input.
    """
    input_tokens = list(get_tokens_and_counts(input_text).keys())
    input_lower = input_text.lower()
    input_token_list = [tok.lower() for tok in input_tokens]

    matched_expressions = set()
    matched_aliases = set()
    alias_matched_expressions = set()  # track expressions matched via alias

    # --- Pass 1: Alias matches (longest first) ---
    for expr, props in expression_map.items():
        aliases = sorted(props.get("aliases", []), key=lambda a: (-len(a.split()), -len(a)))
        for alias in aliases:
            if should_accept_alias_match(alias, input_text, input_tokens, matched_aliases, debug=True):
                matched_expressions.add(expr)
                alias_matched_expressions.add(expr)
                matched_aliases.add(alias.strip().lower())
                if debug:
                    print(f"[✅ MATCH] Alias '{alias}' → {expr}")
                break  # one alias match per expression

    # --- Build alias token set for modifier skip ---
    alias_tokens = {tok for alias in matched_aliases for tok in alias.split()}

    # --- Pass 2: Modifier matches (ranked) ---
    mod_match_scores = {}
    for expr, props in expression_map.items():
        # Skip expressions already matched via alias
        if expr in alias_matched_expressions:
            if debug:
                print(f"[⏭️ SKIP MODIFIER PASS] '{expr}' already matched via alias")
            continue

        # If alias match exists, skip unrelated expressions
        if alias_matched_expressions:
            allowed_with_aliases = set()  # keep empty for strict exclusivity
            if expr not in allowed_with_aliases:
                if debug:
                    print(f"[⏭️ SKIP UNRELATED] Alias match exists → skip '{expr}'")
                continue

        score = 0
        for mod in props.get("modifiers", []):
            mod_lower = mod.lower()

            # Skip if modifier token is already part of an alias match
            if mod_lower in alias_tokens:
                continue

            # Direct token match (e.g., tones like 'rose', 'beige')
            if mod_lower in input_token_list:
                score += 1
                continue

            # Fuzzy match token or whole phrase
            if (
                fuzz.ratio(input_lower, mod_lower) >= 90
                or any(fuzz.ratio(tok, mod_lower) >= 90 for tok in input_token_list)
            ):
                score += 1

        if score > 0:
            mod_match_scores[expr] = score

    # Keep only top scoring expressions (avoid many matches from same tokens)
    if mod_match_scores:
        max_score = max(mod_match_scores.values())
        for expr, score in mod_match_scores.items():
            if score == max_score:
                matched_expressions.add(expr)

    # --- Conflict removal ---
    result = _remove_embedded_conflicts(
        matched_expressions, matched_aliases, input_text, expression_map, debug=False
    )

    # --- Restore alias matches removed by conflicts ---
    result |= alias_matched_expressions

    return result

# ─────────────────────────────────────────────────────────────
# 3. Embedded alias cleanup
# ─────────────────────────────────────────────────────────────

def _remove_embedded_conflicts(matched_expressions, matched_aliases, input_text, expression_map, debug=False):
    """
    Does: Suppresses expressions if their matched alias is embedded in a longer alias.
    Returns: Cleaned set of matched expressions.
    """
    expressions_to_remove = set()

    for expr in matched_expressions:
        aliases = expression_map[expr].get("aliases", [])
        for alias in aliases:
            for other_expr in matched_expressions:
                if other_expr == expr:
                    continue
                other_aliases = expression_map[other_expr].get("aliases", [])
                for other_alias in other_aliases:
                    if alias in other_alias and alias != other_alias:
                        if debug:
                            print(f"[⛔ REMOVE EMBEDDED] '{alias}' in '{other_alias}' → Removing '{expr}'")
                        expressions_to_remove.add(expr)

    return matched_expressions - expressions_to_remove
