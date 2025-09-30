# src/color_sentiment_extractor/extraction/general/fuzzy/scoring.py
from __future__ import annotations

"""
scoring.py

Does: Fuzzy scoring for tokens with context-aware tweaks: hybrid ratio/partial score,
      prefix bonus, rhyme & length penalties, and token-list overlap counting.
Returns: Token-level scorers (fuzzy_token_score, rhyming_conflict) and list-overlap counter.
Used by: General fuzzy matching across color/expression/token pipelines.
"""

import re
from typing import Iterable, List, Sequence
from rapidfuzz import fuzz as rf_fuzz

__all__ = [
    "fuzzy_token_score",
    "rhyming_conflict",
    "fuzzy_token_overlap_count",
]

__docformat__ = "google"

# ── Tunables (centralisés) ───────────────────────────────────────────────────
PREFIX_BONUS_PER_CHAR = 2   # +2 par char de préfixe commun
PREFIX_BONUS_CAP = 8        # plafond bonus de préfixe
RHYME_PENALTY = 12          # pénalité si tokens courts riment mais diffèrent en préfixe
LEN_GAP_PENALTY_PER = 1.5   # pénalité par char d'écart de longueur
LEN_GAP_PENALTY_CAP = 10    # plafond pénalité d'écart de longueur
OVERLAP_FUZZ_THRESHOLD = 85 # seuil par défaut pour l'overlap fuzzy


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """
    Does: Lowercase, trim, map hyphens/underscores to spaces, collapse spaces.
    """
    if s is None:
        return ""
    s = str(s).lower().strip().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ─────────────────────────────────────────────────────────────────────────────
# 1) Token-Level Scoring
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_token_score(a: str, b: str) -> float:
    """
    Does: Compute hybrid similarity (ratio/partial) + prefix bonus − rhyme/length penalties.
    Returns: Score in [0,100].
    """
    a = _norm(a)
    b = _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0

    ratio = rf_fuzz.ratio(a, b)
    partial = rf_fuzz.partial_ratio(a, b)

    # Prefix bonus
    cpl = _common_prefix_len(a, b)
    prefix_bonus = min(PREFIX_BONUS_CAP, cpl * PREFIX_BONUS_PER_CHAR) if cpl >= 2 else 0

    # Rhyme penalty for short tokens that only share ending
    rhyme_penalty = RHYME_PENALTY if rhyming_conflict(a, b) else 0

    # Length gap penalty
    len_gap_penalty = min(LEN_GAP_PENALTY_CAP, abs(len(a) - len(b)) * LEN_GAP_PENALTY_PER)

    score = (ratio + partial) / 2 + prefix_bonus - rhyme_penalty - len_gap_penalty
    score = max(0.0, min(100.0, round(score)))
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 2) Rhyming Conflict Detection
# ─────────────────────────────────────────────────────────────────────────────

def rhyming_conflict(a: str, b: str) -> bool:
    """
    Does: True if two short tokens rhyme (same last 2–3 chars) but start differently.
    """
    a = _norm(a)
    b = _norm(b)
    if not a or not b:
        return False

    # court et similaires par terminaison (2 ou 3 dernières lettres)
    if len(a) <= 6 and len(b) <= 6 and a[:1] != b[:1]:
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
            return True
        if len(a) >= 3 and len(b) >= 3 and a[-3:] == b[-3:]:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3) Token List Overlap
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_token_overlap_count(
    a_tokens: Sequence[str],
    b_tokens: Sequence[str],
    *,
    threshold: int = OVERLAP_FUZZ_THRESHOLD,
) -> int:
    """
    Does: Count overlaps between two token lists (exact or fuzzy ≥ threshold), consuming matches to avoid double count.
    Returns: Integer overlap count.
    """
    if not a_tokens or not b_tokens:
        return 0

    # Normalise une seule fois
    a_norm: List[str] = [_norm(t) for t in a_tokens if _norm(t)]
    b_norm: List[str] = [_norm(t) for t in b_tokens if _norm(t)]

    count = 0
    used = [False] * len(b_norm)

    for a in a_norm:
        # exact first
        try:
            exact_idx = next(j for j, (bb, u) in enumerate(zip(b_norm, used)) if not u and a == bb)
        except StopIteration:
            exact_idx = None

        if exact_idx is not None:
            used[exact_idx] = True
            count += 1
            continue

        # fuzzy fallback: pick best unused
        best_j = -1
        best_score = 0
        for j, (bb, u) in enumerate(zip(b_norm, used)):
            if u:
                continue
            s = rf_fuzz.ratio(a, bb)
            if s > best_score:
                best_score = s
                best_j = j

        if best_j >= 0 and best_score >= threshold:
            used[best_j] = True
            count += 1

    return count
