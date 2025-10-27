# src/color_sentiment_extractor/extraction/color/fuzzy/conflict_rules.py

"""
conflict_rules.py.

Does: Detect simple conflicts between tokens/aliases: negation (e.g., "no shimmer") 
and single-token morphological embedding.

Returns: Boolean checks for negation conflicts and embedded-alias conflicts.
Used by: Fuzzy/alias validation and conflict resolution steps in 
extraction pipelines.
"""

from __future__ import annotations

import logging
import re

from color_sentiment_extractor.extraction.general.token.normalize import normalize_token

__all__ = [
    "is_negation_conflict",
    "is_embedded_alias_conflict",
]

__docformat__ = "google"

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Precompiled patterns (case-insensitive) ──────────────────────────────────
# EN: "no|not|without <token>", FR: "sans <token>", "pas de <token>"
_NEGATION_PATTERNS = [
    re.compile(r"^(?:no|not|without)[- ]+(.+)$", flags=re.IGNORECASE),
    re.compile(r"^sans[- ]+(.+)$", flags=re.IGNORECASE),
    re.compile(r"^pas[- ]+de[- ]+(.+)$", flags=re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1) Negation conflict
# ─────────────────────────────────────────────────────────────────────────────


def _strip_negation(s: str) -> str | None:
    """Does: If s starts with a negation marker (EN/FR), return the negated phrase; else None."""
    if not s:
        return None
    x = normalize_token(s, keep_hyphens=True)
    x = re.sub(r"\s+", " ", x).strip()

    for pat in _NEGATION_PATTERNS:
        m = pat.match(x)
        if m:
            # keep hyphens normalized; caller may recover base
            return m.group(1).strip()

    return None


def is_negation_conflict(a: str, b: str) -> bool:
    """
    Does: True if one string is a simple negation of the other (e.g., "no shimmer" vs "shimmer"),
          comparing also on recovered bases for robustness ("shimmers"/"shimmery" vs "shimmer").
    """
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    if not a or not b:
        return False

    a_norm = normalize_token(a, keep_hyphens=True)
    b_norm = normalize_token(b, keep_hyphens=True)

    a_base_phrase = _strip_negation(a_norm)
    b_base_phrase = _strip_negation(b_norm)

    # Compare on surface OR recovered base(s)
    if a_base_phrase is not None:
        a_base_surface = normalize_token(a_base_phrase, keep_hyphens=True)
        a_base_recovered = (
            recover_base(a_base_surface, use_cache=True, debug=False) or a_base_surface
        )

        b_base_recovered = recover_base(b_norm, use_cache=True, debug=False) or b_norm
        return a_base_surface == b_norm or a_base_recovered == b_base_recovered

    if b_base_phrase is not None:
        b_base_surface = normalize_token(b_base_phrase, keep_hyphens=True)
        b_base_recovered = (
            recover_base(b_base_surface, use_cache=True, debug=False) or b_base_surface
        )

        a_base_recovered = recover_base(a_norm, use_cache=True, debug=False) or a_norm
        return b_base_surface == a_norm or b_base_recovered == a_base_recovered

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2) Embedded alias conflict
# ─────────────────────────────────────────────────────────────────────────────


def is_embedded_alias_conflict(longer: str, shorter: str, min_len: int = 3) -> bool:
    """
    Does: True if `shorter` is morphologically embedded inside a *single-token* `longer`
          (e.g., "rose" ⊂ "rosewood"), excluding multi-token/hyphenated/underscore forms.
    """
    if not longer or not shorter:
        return False

    L = normalize_token(longer, keep_hyphens=True)
    S = normalize_token(shorter, keep_hyphens=True)

    if L == S or len(S) < min_len:
        return False

    # Only consider single-token embeddings
    if re.search(r"[\s_\-]", L):
        return False

    # Morphological embedding: at least one neighbor around the match is alphabetic (accent-safe)
    for m in re.finditer(re.escape(S), L):
        start, end = m.span()
        left = L[start - 1] if start > 0 else ""
        right = L[end] if end < len(L) else ""
        # str.isalpha() handles Unicode letters with accents, unlike [A-Za-z]
        if (left and left.isalpha()) or (right and right.isalpha()):
            return True

    return False
