"""
alias_validation.py
====================

Does: Validate alias tokens (single & multiword) against input text using fuzzy scores,
      suffix/base equivalence, semantic-conflict & rhyme suppression, and overlap heuristics.
Exports: is_valid_singleword_alias, is_multiword_alias_match, should_accept_multiword_alias, remove_subsumed_matches
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Set

from rapidfuzz import fuzz as fuzz  # drop-in pour remplacer fuzzywuzzy

from color_sentiment_extractor.extraction.color.constants import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.general.utils import load_config
from color_sentiment_extractor.extraction.general.fuzzy import (
    is_exact_match,
    rhyming_conflict,
    fuzzy_token_overlap_count,
)
from color_sentiment_extractor.extraction.general.token import (recover_base,
                                                                normalize_token)
from color_sentiment_extractor.extraction.color.recovery import is_suffix_root_match


# ─────────────────────────────────────────────────────────────
# Cached vocab (avoid strong imports; single source of truth = config)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _known_modifiers() -> Set[str]:
    return frozenset(load_config("known_modifiers", mode="set"))

@lru_cache(maxsize=1)
def _known_tones() -> Set[str]:
    return frozenset(load_config("known_tones", mode="set"))


# ─────────────────────────────────────────────────────────────
# 1) Single-token alias checks
# ─────────────────────────────────────────────────────────────

def _tokenize_lower(s: str) -> List[str]:
    if not s:
        return []
    # garde les tirets si besoin plus tard via normalize_token(...)
    return normalize_token(s, keep_hyphens=True).split()

def is_token_fuzzy_match(
    alias: str,
    tokens: List[str],
    input_text: str | None = None,
    debug: bool = True,
    min_score: int = 85,
) -> bool:
    """
    Does: Checks if alias fuzzy-matches any token, with suffix/root and rhyme/semantic filters.
    Returns: True if alias matches token via fuzzy or root derivation.
    """
    alias = (alias or "").strip().lower()
    if not alias:
        if debug: print("[❌] Empty alias")
        return False

    # Precompute normalized input context
    input_words: Set[str] = set(_tokenize_lower(input_text or ""))
    # Also expose bases for containment tests
    input_bases: Set[str] = set(
        recover_base(w, use_cache=True, debug=False) or w
        for w in input_words
    )

    if debug:
        print(f"\n[🔎 ALIAS CHECK] '{alias}' vs tokens: {tokens}")
        if input_text:
            print(f"[🧾 INPUT TEXT] '{input_text}'")

    # Quick containment guard: alias or its base must appear among input words/bases
    if input_text:
        alias_base = recover_base(alias, use_cache=True, debug=False) or alias
        if (alias not in input_words) and (alias_base not in input_words) and (alias_base not in input_bases):
            if debug:
                print(f"[⛔] Alias '{alias}' (base '{alias_base}') not present in input tokens/bases → reject")
            return False

    for tok in tokens or []:
        token = (tok or "").strip().lower()
        if not token:
            continue

        if debug:
            print(f"\n🔁 [TOKEN CHECK] '{alias}' vs '{token}'")

        # conflicts first
        if frozenset({alias, token}) in SEMANTIC_CONFLICTS:
            if debug: print(f"[🚫 SEMANTIC_CONFLICTS] '{alias}' ↔ '{token}'")
            continue

        score = fuzz.ratio(token, alias)
        if debug: print(f"[📏 ratio] '{token}' vs '{alias}' = {score}")

        # High-confidence fuzzy path
        if score >= min_score:
            if rhyming_conflict(alias, token):
                if debug: print(f"[🚫 RHYME@HIGH] blocked despite score {score}")
                continue
            if debug: print(f"[✅ FUZZY OK] score {score} ≥ {min_score}")
            return True

        # Root/suffix equivalence path
        if debug: print(f"[🧪 ROOT PATH] trying suffix/root match")
        if is_suffix_root_match(alias, token, debug=False):
            if debug: print(f"[🌱 ROOT OK] '{alias}' ~ '{token}'")
            return True
        else:
            if debug: print(f"[❌ ROOT FAIL] '{alias}' !~ '{token}'")

        # Soft band
        if 75 <= score < min_score:
            if debug: print(f"[🧪 SOFT BAND] 75 ≤ {score} < {min_score}")
            if rhyming_conflict(alias, token):
                if debug: print(f"[🚫 RHYME@SOFT] blocked")
                continue
            if frozenset({alias, token}) in SEMANTIC_CONFLICTS:
                if debug: print(f"[🚫 SEMANTIC@SOFT] blocked")
                continue

            base_alias = recover_base(alias, use_cache=True, debug=False) or alias
            base_token = recover_base(token, use_cache=True, debug=False) or token

            if base_alias and base_token and base_alias != base_token:
                if debug: print(f"[🚫 BASE MISMATCH] '{alias}'→'{base_alias}' vs '{token}'→'{base_token}'")
                continue

            # optional prefix consistency (guards rhyme-y lookalikes)
            if alias and token and alias[0] != token[0]:
                if debug: print(f"[🚫 PREFIX] '{alias[0]}' != '{token[0]}'")
                continue

            if debug: print(f"[✅ SOFT ACCEPT] '{alias}' ↔ '{token}' (score {score})")
            return True

        # Medium fallback (82) if not a rhyme
        if score >= 82 and not rhyming_conflict(alias, token):
            if debug: print(f"[✅ MED FALLBACK] '{alias}' ↔ '{token}' (score {score})")
            return True

        if debug: print(f"[❌ NO MATCH] '{alias}' ↔ '{token}'")

    if debug: print(f"[❌ FINAL] no token matched for '{alias}'")
    return False


def is_valid_singleword_alias(
    alias: str,
    input_text: str,
    tokens: List[str],
    matched_aliases: set[str],
    debug: bool = False,
) -> bool:
    """
    Does: Validates a single-word alias within context, avoiding being a subpart of a matched multiword.
    Returns: True if exact match or safe fuzzy/root match against provided tokens.
    """
    # Avoid picking a single token already covered by a longer, accepted alias
    for m in matched_aliases:
        if m and (" " in m) and alias in m.split():
            if debug: print(f"[⛔ inside multiword] '{alias}' ∈ '{m}'")
            return False

    if is_exact_match(alias, input_text):
        if debug: print(f"[✅ EXACT] '{alias}' == '{input_text}'")
        return True

    return is_token_fuzzy_match(alias, tokens, input_text=input_text, debug=debug)


# ─────────────────────────────────────────────────────────────
# 2) Multiword alias matching
# ─────────────────────────────────────────────────────────────

def is_multiword_alias_match(
    alias: str,
    input_text: str,
    threshold: int = 85,
    debug: bool = False,
) -> bool:
    """
    Does: Matches a multiword alias to input via partial ratio, glue-equality, two-part reorder, or token-set overlap.
    Returns: True if strong multiword correspondence is detected.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1) Simple partial ratio
    partial_score = fuzz.partial_ratio(norm_alias, norm_input)
    if partial_score >= threshold:
        input_words = norm_input.split()
        alias_words = norm_alias.split()

        # rosegold == rose gold
        if norm_input.replace(" ", "") == norm_alias.replace(" ", ""):
            if debug: print(f"[✅ GLUE MATCH] '{norm_input}' == '{norm_alias}' (unglued)")
            return True

        shared_count = fuzzy_token_overlap_count(input_words, alias_words)
        if len(input_words) < len(alias_words) or shared_count < 2:
            if debug: print(f"[⛔ PARTIAL WEAK] shared={shared_count}")
            return False

        if debug: print(f"[✅ PARTIAL OK] score={partial_score}")
        return True

    # 2) Reordered two-part alias
    alias_parts = norm_alias.split()
    input_parts = norm_input.split()
    if len(alias_parts) == 2 and sorted(alias_parts) == sorted(input_parts):
        if debug: print(f"[🔀 REORDER OK] parts match order-independently")
        return True

    # 3) Token-set fallback
    token_score = fuzz.token_set_ratio(norm_alias, norm_input)
    alias_tokens = set(alias_parts)
    input_tokens = set(input_parts)
    shared = alias_tokens & input_tokens

    if token_score >= threshold and len(shared) >= 2:
        if debug: print(f"[🌀 TOKEN-SET OK] score={token_score} shared={shared}")
        return True

    return False


def should_accept_multiword_alias(
    alias: str,
    input_text: str,
    threshold: int = 80,
    debug: bool = False,
    strict: bool = True,
) -> bool:
    """
    Does: Determines if a multiword alias is acceptable via exact/partial/reordered/part-wise + loose fallback.
    Returns: True for clear acceptance signals.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1) Exact normalized
    if norm_alias == norm_input:
        if debug: print("[✅ EXACT NORM]")
        return True

    # 2) Partial ratio
    score = fuzz.partial_ratio(norm_alias, norm_input)
    if debug: print(f"[🔍 PARTIAL] {score}")
    if score >= threshold:
        return True

    alias_parts = norm_alias.split()
    input_parts = norm_input.split()

    # 3) Two-word reorder
    if len(alias_parts) == 2 and len(input_parts) == 2 and sorted(alias_parts) == sorted(input_parts):
        return True

    # 4) Part-wise strict containment
    matched = 0
    for token in alias_parts:
        best = max((fuzz.partial_ratio(token, other) for other in input_parts), default=0)
        if best >= 85 and (
            not strict or not any(
                token.startswith(other) or other.startswith(token)
                or token.endswith(other) or other.endswith(token)
                for other in input_parts
            )
        ):
            matched += 1
    if matched == len(alias_parts):
        if debug: print("[✅ PART-WISE OK]")
        return True

    # 5) Loose fallback for longer phrases
    loose = fuzz.token_set_ratio(alias, input_text)
    if debug: print(f"[🧪 TOKEN-SET LOOSE] {loose}")
    return loose >= 92 and (len(alias_parts) > 2 or len(input_parts) > 2)


def _handle_multiword_alias(alias: str, input_text: str, debug: bool = False) -> bool:
    """
    Does: Accepts exact match else delegates to multiword matcher.
    Returns: True if alias is accepted as matching the input phrase.
    """
    if is_exact_match(alias, input_text):
        if debug: print(f"[✅ EXACT] '{alias}' == '{input_text}'")
        return True
    return is_multiword_alias_match(alias, input_text, debug=debug)


# ─────────────────────────────────────────────────────────────
# 3) Match postprocessing
# ─────────────────────────────────────────────────────────────

def remove_subsumed_matches(matches: List[str]) -> List[str]:
    """
    Does: Removes shorter matches contained within longer ones (word-boundary aware).
    Returns: List of non-subsumed matches.
    """
    if not matches:
        return []

    filtered: List[str] = []
    for cand in sorted(matches, key=len, reverse=True):
        c = cand.strip()
        if not c:
            continue
        is_subsumed = any(
            c != ex and re.search(rf"\b{re.escape(c)}\b", ex)
            for ex in filtered
        )
        if not is_subsumed:
            filtered.append(c)
    return filtered
