"""
scoring.py
==========

Fuzzy scoring logic for comparing tokens with context-aware adjustments.
Includes:
- Hybrid fuzzy score with prefix bonus and rhyme penalty
- Rhyming conflict detection
- Soft fuzzy overlap count for multi-token inputs
"""

from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rf_fuzz

# ─────────────────────────────────────────────────────────────
# 1. Token-Level Scoring
# ─────────────────────────────────────────────────────────────

def fuzzy_token_score(a: str, b: str) -> float:
    """
    Does: Computes a fuzzy similarity score between two tokens using fuzz ratios.
    - Adds bonus for prefix match and subtracts penalty for short rhyming pairs.
    Returns: A score from 0 to 100 reflecting similarity, capped within range.
    """
    partial = fuzz.partial_ratio(a, b)
    ratio = fuzz.ratio(a, b)

    # Prefix bonus (true only if both share start)
    bonus = 8 if a[:3] == b[:3] else 0

    # Penalize short rhyme matches like "ink/pink"
    penalty = 12 if (
        len(a) <= 5 and len(b) <= 5 and
        a[-2:] == b[-2:] and a[0] != b[0]
    ) else 0

    score = (partial + ratio) / 2 + bonus - penalty
    return max(0, min(100, round(score)))

# ─────────────────────────────────────────────────────────────
# 2. Rhyming Conflict Detection
# ─────────────────────────────────────────────────────────────
def rhyming_conflict(a: str, b: str) -> bool:
    """
    Does: Checks if two short tokens rhyme but start differently (e.g. 'ink' vs 'pink').
    Returns: True if likely a misleading rhyme match.
    """
    return (
        len(a) <= 5 and len(b) <= 5 and
        a[-2:] == b[-2:] and a[0] != b[0]
    )

# ─────────────────────────────────────────────────────────────
# 3. Token List Overlap
# ─────────────────────────────────────────────────────────────
def fuzzy_token_overlap_count(a_tokens, b_tokens):
    """
       Does: Counts how many tokens in one list match tokens in another, using exact or fuzzy match (≥85).
       Returns: Integer count of overlapping tokens across both lists, including soft suffix variants.
       """
    count = 0
    for a in a_tokens:
        for b in b_tokens:
            if a == b:
                count += 1
                break
            if rf_fuzz.ratio(a, b) >= 85:  # soft ~ softy, pink ~ pinky
                count += 1
                break
    return count
