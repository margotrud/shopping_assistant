"""
conflict_rules.py
=================

Simple logic for identifying token-level or alias-level conflicts.
Covers negation (e.g. 'no shimmer') and substring embedding.
"""

from __future__ import annotations
import re

from color_sentiment_extractor.extraction.general.token.normalize import normalize_token


# ─────────────────────────────────────────────────────────────
# 1) Negation conflict
# ─────────────────────────────────────────────────────────────

def _strip_negation(s: str) -> str | None:
    """
    Does: If s starts with a negation marker, return the negated base token; else None.
    """
    if not s:
        return None
    x = normalize_token(s, keep_hyphens=True)
    # Collapse multiple spaces (normalize_token le fait déjà en général)
    x = re.sub(r"\s+", " ", x).strip()

    # EN: no/not/without + optional hyphen
    for neg in ("no", "not", "without"):
        m = re.match(rf"^{neg}[- ]+(.+)$", x)
        if m:
            return m.group(1).strip()

    # FR: sans <token>
    m = re.match(r"^sans[- ]+(.+)$", x)
    if m:
        return m.group(1).strip()

    # FR: pas de <token>
    m = re.match(r"^pas[- ]+de[- ]+(.+)$", x)
    if m:
        return m.group(1).strip()

    return None


def is_negation_conflict(a: str, b: str) -> bool:
    """
    Does: True if one string is a simple negation of the other (e.g. 'no shimmer' vs 'shimmer').
    """
    a_norm = normalize_token(a, keep_hyphens=True)
    b_norm = normalize_token(b, keep_hyphens=True)

    a_base = _strip_negation(a_norm)
    b_base = _strip_negation(b_norm)

    if a_base is not None:
        return a_base == b_norm
    if b_base is not None:
        return b_base == a_norm
    return False


# ─────────────────────────────────────────────────────────────
# 2) Embedded alias conflict
# ─────────────────────────────────────────────────────────────

def is_embedded_alias_conflict(longer: str, shorter: str, min_len: int = 3) -> bool:
    """
    Does: True if shorter is embedded morphologically inside a single-token longer (e.g. 'rose'⊂'rosewood').
    """
    if not longer or not shorter:
        return False

    L = normalize_token(longer, keep_hyphens=True)
    S = normalize_token(shorter, keep_hyphens=True)

    if L == S or len(S) < min_len:
        return False

    # Only consider *single token* embeddings: no spaces/underscores/hyphens in the longer form
    if re.search(r"[\s_\-]", L):
        return False

    # Look for occurrences where at least one neighbor is a letter (morphological embedding),
    # i.e., not a clean word boundary match inside a phrase.
    for m in re.finditer(re.escape(S), L):
        start, end = m.span()
        left = L[start - 1] if start > 0 else ""
        right = L[end] if end < len(L) else ""
        left_is_alpha = bool(re.match(r"[A-Za-z]", left))
        right_is_alpha = bool(re.match(r"[A-Za-z]", right))
        if left_is_alpha or right_is_alpha:
            return True

    return False
