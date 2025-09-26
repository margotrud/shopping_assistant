"""
fuzzy_core.py
==============
Core fuzzy engine for token comparison.
Does: Similarity scoring, base recovery, conflict/negation-aware checks; safe best-match against a known set.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Set

from fuzzywuzzy import fuzz

from color_sentiment_extractor.extraction.general.fuzzy import (
    fuzzy_token_score,
    is_negation_conflict,
)
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def collapse_duplicates(s: str) -> str:
    """
    Does: Collapse repeated chars (e.g., 'cooool'→'col').
    Returns: String.
    """
    return re.sub(r"(.)\1+", r"\1", s)


def is_single_transposition(a: str, b: str) -> bool:
    """
    Does: Check one adjacent swap between equal-length strings.
    Returns: Boolean.
    """
    if len(a) != len(b):
        return False
    diffs = [(i, a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
    return len(diffs) == 2 and diffs[0][1] == diffs[1][2] and diffs[0][2] == diffs[1][1]


def is_single_substitution(a: str, b: str) -> bool:
    """
    Does: Check exactly one differing position between equal-length strings.
    Returns: Boolean.
    """
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b) if x != y) == 1


def _safe_norm(s: str) -> str:
    """
    Does: Lowercase, map hyphens/underscores to spaces, collapse spaces.
    Returns: Normalized string.
    """
    s = s.lower().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


# ─────────────────────────────────────────────────────────────
# 1) FUZZY SIMILARITY MATCHING
# ─────────────────────────────────────────────────────────────

def fuzzy_token_match(a: str, b: str) -> float:
    """
    Does: Return 100 for exact/derivational/edit-like matches; else fuzzy score (do not floor here).
    Returns: Float score in [0,100].
    """
    # lazy import to avoid circulars on module import
    try:
        from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base as _recover_base
    except Exception:
        _recover_base = None

    a = normalize_token(a, keep_hyphens=True)
    b = normalize_token(b, keep_hyphens=True)

    if a == b or is_single_transposition(a, b):
        return 100.0

    # Base recovery (no fuzzy) — centralizes suffix logic in recover_base
    base_a = _recover_base(a, allow_fuzzy=False) if _recover_base else None
    base_b = _recover_base(b, allow_fuzzy=False) if _recover_base else None
    if base_a == b or base_b == a or (base_a and base_b and base_a == base_b):
        return 100.0

    score = float(fuzzy_token_score(a, b))
    return score


def _in_conflict_groups(a: str, b: str, groups: Optional[Iterable]) -> bool:
    """
    Does: Generic membership check for semantic conflict groups.
    Returns: True if a and b are distinct members of any group.
    """
    if not groups:
        return False
    a = normalize_token(a)
    b = normalize_token(b)
    try:
        if isinstance(groups, dict):
            for k, v in groups.items():
                s = {normalize_token(k)} | {
                    normalize_token(x) for x in (v if isinstance(v, (list, tuple, set, frozenset)) else [v])
                }
                if a in s and b in s and a != b:
                    return True
        else:
            for g in groups:
                if isinstance(g, (list, tuple, set, frozenset)):
                    s = {normalize_token(x) for x in g}
                else:
                    s = {normalize_token(str(g))}
                if a in s and b in s and a != b:
                    return True
    except Exception:
        pass
    return False


def is_strong_fuzzy_match(
    a: str,
    b: str,
    threshold: int = 82,
    *,
    conflict_groups: Optional[Iterable] = None,
    negation_check: bool = True,
) -> bool:
    """
    Does: Strong match test with conflict/negation guards.
    Returns: True iff score ≥ threshold and no conflict.
    """
    a_norm = normalize_token(a, keep_hyphens=True)
    b_norm = normalize_token(b, keep_hyphens=True)

    if _in_conflict_groups(a_norm, b_norm, conflict_groups):
        return False
    if negation_check and is_negation_conflict(a_norm, b_norm):
        return False

    return fuzzy_token_match(a_norm, b_norm) >= float(threshold)


def is_exact_match(a: str, b: str) -> bool:
    """
    Does: Normalize, strip non-alnum, accept same or fuzz.ratio ≥ 90.
    Returns: Boolean.
    """
    def clean(text: str) -> str:
        norm = normalize_token(text, keep_hyphens=True)
        return re.sub(r"[^a-z0-9]", "", norm.lower())

    a_c = clean(a)
    b_c = clean(b)
    if a_c == b_c:
        return True
    return fuzz.ratio(a_c, b_c) >= 90


# ─────────────────────────────────────────────────────────────
# 2) SAFE BEST-MATCH AGAINST VOCAB
# ─────────────────────────────────────────────────────────────

def fuzzy_match_token_safe(
    raw_token: str,
    known_tokens: Set[str],
    threshold: int = 82,
    debug: bool = False,
) -> Optional[str]:
    """
    Does: Safe best-match: exact/edit-like/normalized/base-recovery → fuzzy.
    Returns: Candidate or None if below threshold.
    """
    # lazy import here too
    try:
        from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base as _recover_base
    except Exception:
        _recover_base = None

    # ✅ garde-fous types/vides
    if raw_token is None:
        return None
    raw = str(raw_token).lower().strip()
    if not raw:
        return None

    best_match, best_score = None, 0

    if debug:
        print(f"[INPUT] raw_token='{raw_token}' → norm='{raw}' (|known|={len(known_tokens)})")

    for candidate in known_tokens:
        # ✅ SKIP non-strings / vides
        if not isinstance(candidate, str):
            if debug:
                print(f"[SKIP non-str] {repr(candidate)}")
            continue
        cand = candidate.lower().strip()
        if not cand:
            if debug:
                print(f"[SKIP empty] {repr(candidate)}")
            continue

        # Pré-filtres rapides
        if abs(len(cand) - len(raw)) > 2:
            continue
        if raw and cand and raw[0] != cand[0] and min(len(raw), len(cand)) >= 3:
            continue

        # 1) exact
        if raw == cand:
            return candidate

        # 2) single transposition
        if is_single_transposition(raw, cand):
            return candidate

        # 3) dedupe equality
        raw_dedup = collapse_duplicates(raw)
        cand_dedup = collapse_duplicates(cand)
        if raw_dedup == cand_dedup:
            return candidate

        # 4) dedupe + single substitution (✅ compare aux deux dédupliqués)
        if is_single_substitution(raw_dedup, cand_dedup):
            return candidate

        # 5a) single deletion (cand plus long d’1)
        if len(cand) == len(raw) + 1:
            for i in range(len(cand)):
                if cand[:i] + cand[i + 1:] == raw:
                    if debug:
                        print(f"[DEL] '{raw}' ← '{cand}'")
                    return candidate

        # 5b) single insertion (raw plus long d’1)
        if len(raw) == len(cand) + 1:
            for i in range(len(raw)):
                if raw[:i] + raw[i + 1:] == cand:
                    if debug:
                        print(f"[INS] '{raw}' → '{cand}'")
                    return candidate

        # 6) hyphen/underscore/space normalization equivalence
        if _safe_norm(raw) == _safe_norm(cand):
            if debug:
                print(f"[NORM] '{raw}' ≡ '{candidate}'")
            return candidate

        # 7) track best fuzzy
        try:
            score = fuzz.token_sort_ratio(raw, cand)
        except Exception as e:
            if debug:
                print(f"[FUZZ ERROR] '{raw}' vs '{cand}': {e}")
            continue

        if score > best_score:
            best_score, best_match = score, candidate
        if debug and score >= threshold:
            print(f"[FUZZY≥{threshold}] '{raw}' vs '{cand}' = {score}")

    # 8) centralized base recovery (suffixes, overrides, optional fuzzy)
    base = _recover_base(raw, allow_fuzzy=False) if _recover_base else None
    if base and isinstance(base, str) and base in known_tokens:
        if debug:
            print(f"[BASE] '{raw}' → '{base}'")
        return base

    return best_match if best_score >= threshold else None
