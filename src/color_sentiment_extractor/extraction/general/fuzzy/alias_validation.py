"""
alias_validation.py
====================

Handles validation of alias tokens against tokenized input.
Includes fuzzy matching, suffix/root equivalence, multiword reordering,
semantic conflict detection, and rhyme suppression.
"""

import re
from typing import List

from fuzzywuzzy import fuzz

from extraction.color.vocab import known_tones
from extraction.color.constants import SEMANTIC_CONFLICTS
from extraction.general.utils.load_config import load_config
from extraction.general.fuzzy.scoring import rhyming_conflict, fuzzy_token_overlap_count
from extraction.general.token.base_recovery import recover_base
from extraction.general.token.normalize import normalize_token
from extraction.color.recovery.fuzzy_recovery import is_suffix_root_match
from extraction.general.fuzzy.fuzzy_core import is_exact_match


known_modifiers = load_config("known_modifiers", mode="set")

# ─────────────────────────────────────────────────────────────
# 1. Single-token alias checks
# ─────────────────────────────────────────────────────────────

def is_token_fuzzy_match(
    alias: str,
    tokens: list[str],
    input_text: str | None = None,
    matched_aliases: set[str] | None = None,
    debug: bool = True,
    min_score: int = 85
) -> bool:
    """
    Does: Checks if alias fuzzy-matches any token, with suffix and rhyme filtering.
    Returns: True if alias matches token via fuzzy or root derivation, avoiding semantic/rhyme traps.
    """
    alias = alias.strip().lower()
    matched_aliases = matched_aliases or set()

    if debug:
        print(f"\n[🔎 ALIAS CHECK] '{alias}' vs token list: {tokens}")
        if input_text:
            print(f"[🧾 INPUT TEXT] '{input_text}'")

    # Reject if alias is not in the input at all
    if input_text:
        input_tokens = input_text.strip().lower().split()
        if not any(
                t in input_text or recover_base(t, known_modifiers, known_tones, debug=False) in input_text
                for t in [alias]
        ):
            if debug:
                print(f"[⛔ BLOCKED: ALIAS NOT IN INPUT (OR BASE)] '{alias}' not in '{input_text}' → rejecting")
            return False

        # Block alias match if it's part of a phrase but not among allowed tokens
        if len(input_tokens) == 2 and alias in input_tokens and alias not in tokens:
            if debug:
                print(
                    f"[⛔ BLOCKED: EMBEDDED TOKEN] Alias '{alias}' in phrase '{input_text}', but not a listed token → blocked")
            return False

    for token in tokens:
        token = token.strip().lower()
        if debug:
            print(f"\n🔁 [TOKEN CHECK] Comparing alias '{alias}' to token '{token}'")

        if frozenset({alias, token}) in SEMANTIC_CONFLICTS:
            if debug:
                print(f"[🚫 BLOCKED BY SEMANTIC_CONFLICTS] '{alias}' vs '{token}'")
            continue

        score = fuzz.ratio(token, alias)
        if debug:
            print(f"[📏 FUZZY SCORE] fuzz.ratio('{token}', '{alias}') = {score}")

        if score >= min_score:
            if rhyming_conflict(alias, token):
                if debug:
                    print(f"[🚫 RHYME BLOCKED @HIGH] '{alias}' ↔ '{token}' blocked despite score {score}")
                continue
            if debug:
                print(f"[✅ FUZZY MATCH] Score {score} ≥ {min_score} → Accepting '{alias}' ↔ '{token}'")
            return True

        if debug:
            print(f"[🧪 LOW SCORE FALLBACK] Score {score} < {min_score} → Trying suffix/root match")
        if is_suffix_root_match(alias, token, debug=False):
            if debug:
                print(f"[🌱 ROOT MATCH SUCCESS] '{alias}' ↔ '{token}' matched by suffix/root")
            return True
        else:
            if debug:
                print(f"[❌ ROOT MATCH FAIL] '{alias}' ↔ '{token}' → No suffix/root match")

        if 75 <= score < min_score:
            if debug:
                print(f"[🧪 SOFT MATCH PATH] Score {score} in soft range (75–{min_score})")

            if rhyming_conflict(alias, token):
                if debug: print(f"[🚫 RHYME BLOCKED: SOFT MATCH] '{alias}' ↔ '{token}'")
                continue
            if frozenset({alias, token}) in SEMANTIC_CONFLICTS:
                if debug: print(f"[🚫 SEMANTIC BLOCKED: SOFT MATCH] '{alias}' ↔ '{token}'")
                continue

            base_alias = recover_base(alias, use_cache=True, debug=False)
            base_token = recover_base(token, use_cache=True, debug=False)

            if base_alias and base_token and base_alias != base_token:
                if debug:
                    print(f"[🚫 BASE MISMATCH BLOCK] '{alias}'→'{base_alias}' vs '{token}'→'{base_token}'")
                continue

            if alias[0] != token[0]:  # optional: same starting letter
                if debug:
                    print(f"[🚫 PREFIX MISMATCH] '{alias[0]}' vs '{token[0]}'")
                continue

            if debug:
                print(f"[✅ ACCEPTED: SOFT FUZZY MATCH] '{alias}' ↔ '{token}' with score {score}")
            return True

        if score >= 82:
            if not rhyming_conflict(alias, token):
                if debug:
                    print(f"[✅ FALLBACK FUZZY PASS] '{alias}' ↔ '{token}' with score {score}")
                return True
            else:
                if debug:
                    print(f"[🚫 RHYME BLOCKED] '{alias}' ↔ '{token}' rejected despite score {score}")

        if debug:
            print(f"[❌ NO MATCH] '{alias}' ↔ '{token}' → Rejected")

    if debug:
        print(f"[❌ FINAL] No match found for alias '{alias}' across all tokens.")
    return False


def is_valid_singleword_alias(
    alias: str,
    input_text: str,
    tokens: list[str],
    matched_aliases: set[str],
    debug: bool = False
) -> bool:
    """
    Does: Determines if a single-word alias is valid in context.
    - Rejects if alias is embedded in a multi-word alias already matched
    - Accepts if exact match to input
    - Accepts if it fuzzy-matches one of the tokens safely
    Returns: Boolean flag
    """
    for matched in matched_aliases:
        if alias in matched and len(matched.split()) > 1:
            if debug:
                print(f"[⛔ BLOCKED: token inside multiword] '{alias}' in '{matched}'")
            return False

    if is_exact_match(alias, input_text):
        if debug:
            print(f"[✅ EXACT MATCH] '{alias}' == '{input_text}'")
        return True

    return is_token_fuzzy_match(alias, tokens, matched_aliases=matched_aliases, debug=debug)

# ─────────────────────────────────────────────────────────────
# 2. Multiword alias matching
# ─────────────────────────────────────────────────────────────

def is_multiword_alias_match(
    alias: str,
    input_text: str,
    threshold: int = 85,
    debug: bool = False
) -> bool:
    """
    Does: Determines if a multiword alias matches the input phrase using fuzzy and token-set logic.
    Returns: True if alias matches input via partial ratio, reordered parts, or fuzzy overlap.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1. Simple partial ratio check
    partial_score = fuzz.partial_ratio(norm_alias, norm_input)
    if partial_score >= threshold:
        input_words = norm_input.split()
        alias_words = norm_alias.split()

        # ⛓️ Glued-token direct match (e.g. 'rosegold' == 'rose gold')
        if norm_input.replace(" ", "") == norm_alias.replace(" ", ""):
            if debug:
                print(f"[✅ DIRECT GLUE MATCH] {norm_input} == {norm_alias} (unglued)")
            return True

        shared_count = fuzzy_token_overlap_count(input_words, alias_words)

        if len(input_words) < len(alias_words) or shared_count < 2:
            if debug:
                print(f"[⛔ SKIP PARTIAL] Overlap too weak: shared_count={shared_count}")
            return False

        if debug:
            print(f"[🔍 FUZZ.partial_ratio] {norm_alias} ~ {norm_input} → {partial_score}")
        return True

    # 2. Reordered two-part alias
    alias_parts = norm_alias.split()
    input_parts = norm_input.split()
    if len(alias_parts) == 2 and sorted(alias_parts) == sorted(input_parts):
        if debug:
            print(f"[🔀 REORDERED MATCH] '{alias}' parts found in input")
        return True

    # 3. Token set + overlap fallback
    token_score = fuzz.token_set_ratio(norm_alias, norm_input)
    alias_tokens = set(norm_alias.split())
    input_tokens = set(norm_input.split())
    shared = alias_tokens & input_tokens

    if token_score >= threshold and len(shared) >= 2:
        if debug:
            print(f"[🌀 TOKEN SET MATCH] {norm_alias} ~ {norm_input} → {token_score} (shared: {shared})")
        return True
        if debug:
            print(f"[🌀 TOKEN SET MATCH] {norm_alias} ~ {norm_input} → {token_score}")
        return True

    return False

def should_accept_multiword_alias(
    alias: str,
    input_text: str,
    threshold: int = 80,
    debug: bool = False,
    strict: bool = True
) -> bool:
    """
    Does: Determines if a multiword alias is a good match for the input text.
    Uses: normalization, partial ratio, part-wise matching, and fallback similarity.
    Returns: True if alias should be accepted as a match.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1. Exact normalized match
    if norm_alias == norm_input:
        if debug: print("[✅ MATCH] Exact normalized match")
        return True

    # 2. Partial ratio fuzzy match
    score = fuzz.partial_ratio(norm_alias, norm_input)
    if debug: print(f"[🔍 FUZZ.partial_ratio] → {score}")
    if score >= threshold:
        return True

    alias_parts = norm_alias.split()
    input_parts = norm_input.split()

    # 3. Order-independent 2-word match (e.g. 'pink soft' vs 'soft pink')
    if len(alias_parts) == 2 and len(input_parts) == 2 and sorted(alias_parts) == sorted(input_parts):
        return True

    # 4. Per-token fuzzy matching with optional strict containment
    matched = 0
    for token in alias_parts:
        best_score = max(fuzz.partial_ratio(token, other) for other in input_parts)
        if best_score >= 85 and (
            not strict or not any(
                token.startswith(other) or other.startswith(token)
                or token.endswith(other) or other.endswith(token)
                for other in input_parts
            )
        ):
            matched += 1

    if matched == len(alias_parts):
        if debug: print("[✅ MATCH] All alias parts passed strict fuzzy containment")
        return True

    # 5. Loose fallback: token set ratio for long phrases
    loose_score = fuzz.token_set_ratio(alias, input_text)
    if debug: print(f"[🧪 FUZZ.token_set_ratio] → {loose_score}")
    return loose_score >= 92 and (len(alias.split()) > 2 or len(input_text.split()) > 2)

def _handle_multiword_alias(alias, input_text, debug: bool = False):
    """
    Does: Handles validation of multi-word alias matches against input.
    - Accepts exact match or delegates to fuzzy multiword matching logic.
    Returns: True if alias is accepted as matching the input phrase.
    """
    if is_exact_match(alias, input_text):
        if debug: print(f"[✅ EXACT MATCH] '{alias}' == '{input_text}'")
        return True
    return is_multiword_alias_match(alias, input_text, debug=debug)


# ─────────────────────────────────────────────────────────────
# 3. Match postprocessing
# ─────────────────────────────────────────────────────────────
def remove_subsumed_matches(matches: List[str]) -> List[str]:
    """
    Does: Removes shorter matches that are fully contained in longer ones.
    Returns: List of non-subsumed match strings.
    """
    filtered = []
    matches = sorted(matches, key=len, reverse=True)

    for candidate in matches:
        is_subsumed = any(
            candidate != existing and (
                (" " not in existing and existing.startswith(candidate)) or
                re.search(rf'\b{re.escape(candidate)}\b', existing)
            )
            for existing in filtered
        )
        if not is_subsumed:
            filtered.append(candidate)

    return filtered
