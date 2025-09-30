# src/color_sentiment_extractor/extraction/color/fuzzy/alias_validation.py
from __future__ import annotations

"""
alias_validation.py

Does: Validate single/multiword color aliases with fuzzy scores, suffix/base equivalence,
      rhyme & semantic-conflict suppression, and overlap/subsumption heuristics.
Returns: Boolean match decisions + utilities to filter overlapping hits.
Used by: Color alias matching inside extraction pipelines.
"""

import logging
import re
from typing import List

from rapidfuzz import fuzz  # performant, no numpy dependency

from color_sentiment_extractor.extraction.color.constants import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.general.fuzzy import (
    is_exact_match,
    rhyming_conflict,
    fuzzy_token_overlap_count,
)
from color_sentiment_extractor.extraction.general.token import (
    recover_base,
    normalize_token,
)
from color_sentiment_extractor.extraction.color.recovery import is_suffix_root_match

__all__ = [
    "is_valid_singleword_alias",
    "is_multiword_alias_match",
    "should_accept_multiword_alias",
    "remove_subsumed_matches",
]

__docformat__ = "google"

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Tunables (centralisés) ───────────────────────────────────────────────────
MIN_STRONG = 85       # seuil "fort" pour fuzz.ratio / partial_ratio
MIN_MEDIUM = 82       # fallback medium si pas de conflit de rime
SOFT_BAND_LOW = 75    # bande "soft" (fenêtre d'acceptation conditionnelle)
TOKEN_SET_LOOSE = 92  # seuil loose pour token_set sur phrases longues


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_lower(s: str) -> List[str]:
    """Does: Normalize + split lowercase while conservant les tirets si utile plus tard."""
    if not s:
        return []
    return normalize_token(s, keep_hyphens=True).split()


# ─────────────────────────────────────────────────────────────────────────────
# 1) Single-token alias checks
# ─────────────────────────────────────────────────────────────────────────────

def is_token_fuzzy_match(
    alias: str,
    tokens: List[str],
    input_text: str | None = None,
    debug: bool = True,
    min_score: int = MIN_STRONG,
) -> bool:
    """
    Does: Check if alias matches any token via fuzzy or suffix/root equivalence with rhyme/semantic guards.
    Returns: True iff a safe match is found.
    """
    a = (alias or "").strip().lower()
    if not a:
        if debug:
            log.debug("[❌] Empty alias")
        return False

    # Precompute normalized context words + bases
    input_words = set(_tokenize_lower(input_text or ""))
    input_bases = {recover_base(w, use_cache=True, debug=False) or w for w in input_words}

    if debug and input_text:
        log.debug("🔎 ALIAS CHECK %r vs tokens=%s | input=%r", a, tokens, input_text)

    # Guard: alias or its base should appear in the input if provided
    if input_text:
        alias_base = recover_base(a, use_cache=True, debug=False) or a
        if (a not in input_words) and (alias_base not in input_words) and (alias_base not in input_bases):
            if debug:
                log.debug("[⛔] '%s' (base '%s') not in input tokens/bases → reject", a, alias_base)
            return False

    for tok in tokens or []:
        t = (tok or "").strip().lower()
        if not t:
            continue

        if debug:
            log.debug("🔁 TOKEN CHECK %r vs %r", a, t)

        # Hard semantic conflict guard
        if frozenset({a, t}) in SEMANTIC_CONFLICTS:
            if debug:
                log.debug("[🚫 SEMANTIC_CONFLICTS] %r ↔ %r", a, t)
            continue

        score = fuzz.ratio(t, a)
        if debug:
            log.debug("[📏 ratio] %r vs %r = %s", t, a, score)

        # Strong fuzzy acceptance
        if score >= min_score:
            if rhyming_conflict(a, t):
                if debug:
                    log.debug("[🚫 RHYME@HIGH] blocked despite score %s", score)
                continue
            if debug:
                log.debug("[✅ FUZZY OK] score %s ≥ %s", score, min_score)
            return True

        # Root/suffix equivalence path
        if debug:
            log.debug("[🧪 ROOT PATH] trying suffix/root match")
        if is_suffix_root_match(a, t, debug=False):
            if debug:
                log.debug("[🌱 ROOT OK] %r ~ %r", a, t)
            return True
        else:
            if debug:
                log.debug("[❌ ROOT FAIL] %r !~ %r", a, t)

        # Soft band with extra guards
        if SOFT_BAND_LOW <= score < min_score:
            if debug:
                log.debug("[🧪 SOFT BAND] %s ≤ %s < %s", SOFT_BAND_LOW, score, min_score)
            if rhyming_conflict(a, t):
                if debug:
                    log.debug("[🚫 RHYME@SOFT] blocked")
                continue
            if frozenset({a, t}) in SEMANTIC_CONFLICTS:
                if debug:
                    log.debug("[🚫 SEMANTIC@SOFT] blocked")
                continue

            base_a = recover_base(a, use_cache=True, debug=False) or a
            base_t = recover_base(t, use_cache=True, debug=False) or t
            if base_a and base_t and base_a != base_t:
                if debug:
                    log.debug("[🚫 BASE MISMATCH] %r→%r vs %r→%r", a, base_a, t, base_t)
                continue

            # Optional: simple prefix guard to reduce rhyme-y lookalikes
            if a and t and a[0] != t[0]:
                if debug:
                    log.debug("[🚫 PREFIX] %r != %r", a[0], t[0])
                continue

            if debug:
                log.debug("[✅ SOFT ACCEPT] %r ↔ %r (score %s)", a, t, score)
            return True

        # Medium fallback if not a rhyme
        if score >= MIN_MEDIUM and not rhyming_conflict(a, t):
            if debug:
                log.debug("[✅ MED FALLBACK] %r ↔ %r (score %s)", a, t, score)
            return True

        if debug:
            log.debug("[❌ NO MATCH] %r ↔ %r", a, t)

    if debug:
        log.debug("[❌ FINAL] no token matched for %r", a)
    return False


def is_valid_singleword_alias(
    alias: str,
    input_text: str,
    tokens: List[str],
    matched_aliases: set[str],
    debug: bool = False,
) -> bool:
    """
    Does: Validate a single-word alias in context, avoiding overlap with accepted multiwords.
    Returns: True iff exact or safe fuzzy/root match against tokens.
    """
    # Avoid picking a single token already covered by a longer, accepted alias
    for m in matched_aliases:
        if m and (" " in m) and alias in m.split():
            if debug:
                log.debug("[⛔ inside multiword] %r ∈ %r", alias, m)
            return False

    if is_exact_match(alias, input_text):
        if debug:
            log.debug("[✅ EXACT] %r == %r", alias, input_text)
        return True

    return is_token_fuzzy_match(alias, tokens, input_text=input_text, debug=debug)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Multiword alias matching
# ─────────────────────────────────────────────────────────────────────────────

def is_multiword_alias_match(
    alias: str,
    input_text: str,
    threshold: int = MIN_STRONG,
    debug: bool = False,
) -> bool:
    """
    Does: Match multiword alias via partial ratio, glue-equality, two-part reorder, or token-set overlap.
    Returns: True iff strong multiword correspondence is detected.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1) Partial ratio
    partial_score = fuzz.partial_ratio(norm_alias, norm_input)
    if partial_score >= threshold:
        input_words = norm_input.split()
        alias_words = norm_alias.split()

        # rosegold == rose gold
        if norm_input.replace(" ", "") == norm_alias.replace(" ", ""):
            if debug:
                log.debug("[✅ GLUE MATCH] %r == %r (unglued)", norm_input, norm_alias)
            return True

        shared_count = fuzzy_token_overlap_count(input_words, alias_words)
        if len(input_words) < len(alias_words) or shared_count < 2:
            if debug:
                log.debug("[⛔ PARTIAL WEAK] shared=%s", shared_count)
            return False

        if debug:
            log.debug("[✅ PARTIAL OK] score=%s", partial_score)
        return True

    # 2) Reordered two-part alias
    alias_parts = norm_alias.split()
    input_parts = norm_input.split()
    if len(alias_parts) == 2 and sorted(alias_parts) == sorted(input_parts):
        if debug:
            log.debug("[🔀 REORDER OK] parts match order-independently")
        return True

    # 3) Token-set fallback
    token_score = fuzz.token_set_ratio(norm_alias, norm_input)
    alias_tokens = set(alias_parts)
    input_tokens = set(input_parts)
    shared = alias_tokens & input_tokens

    if token_score >= threshold and len(shared) >= 2:
        if debug:
            log.debug("[🌀 TOKEN-SET OK] score=%s shared=%s", token_score, shared)
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
    Does: Decide acceptance for multiword alias via exact/partial/reordered/part-wise + loose token-set fallback.
    Returns: True for clear acceptance signals.
    """
    norm_alias = normalize_token(alias, keep_hyphens=True)
    norm_input = normalize_token(input_text, keep_hyphens=True)

    # 1) Exact normalized
    if norm_alias == norm_input:
        if debug:
            log.debug("[✅ EXACT NORM]")
        return True

    # 2) Partial ratio
    score = fuzz.partial_ratio(norm_alias, norm_input)
    if debug:
        log.debug("[🔍 PARTIAL] %s", score)
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
        if best >= MIN_STRONG and (
            not strict
            or not any(
                token.startswith(other)
                or other.startswith(token)
                or token.endswith(other)
                or other.endswith(token)
                for other in input_parts
            )
        ):
            matched += 1
    if matched == len(alias_parts):
        if debug:
            log.debug("[✅ PART-WISE OK]")
        return True

    # 5) Loose fallback for longer phrases
    loose = fuzz.token_set_ratio(alias, input_text)
    if debug:
        log.debug("[🧪 TOKEN-SET LOOSE] %s", loose)
    return loose >= TOKEN_SET_LOOSE and (len(alias_parts) > 2 or len(input_parts) > 2)


def _handle_multiword_alias(alias: str, input_text: str, debug: bool = False) -> bool:
    """
    Does: Accept exact match else delegate to multiword matcher.
    Returns: True iff alias matches input phrase.
    """
    if is_exact_match(alias, input_text):
        if debug:
            log.debug("[✅ EXACT] %r == %r", alias, input_text)
        return True
    return is_multiword_alias_match(alias, input_text, debug=debug)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Match postprocessing
# ─────────────────────────────────────────────────────────────────────────────

def remove_subsumed_matches(matches: List[str]) -> List[str]:
    """
    Does: Remove shorter matches contained within longer ones (word-boundary aware, hyphen-robust).
    Returns: List of non-subsumed matches.
    """
    if not matches:
        return []

    def _norm_for_sub(s: str) -> str:
        # Normalise pour détection robuste: remplace hyphens/underscores par espaces, squeeze whitespace
        s = re.sub(r"[-_]+", " ", s.strip())
        s = re.sub(r"\s+", " ", s)
        return s

    filtered: List[str] = []
    for cand in sorted(matches, key=len, reverse=True):
        c = cand.strip()
        if not c:
            continue
        nc = _norm_for_sub(c)
        is_subsumed = any(
            c != ex and re.search(rf"\b{re.escape(nc)}\b", _norm_for_sub(ex))
            for ex in filtered
        )
        if not is_subsumed:
            filtered.append(c)
    return filtered
