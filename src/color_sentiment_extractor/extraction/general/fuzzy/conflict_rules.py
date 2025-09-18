"""
conflict_rules.py
=================

Simple logic for identifying token-level or alias-level conflicts.
Covers negation (e.g. 'no shimmer') and substring embedding.
"""


from color_sentiment_extractor.extraction.general.token.normalize import normalize_token

# ─────────────────────────────────────────────────────────────
# 1. Negation conflict
# ─────────────────────────────────────────────────────────────

def is_negation_conflict(a: str, b: str) -> bool:
    """
    Does: Returns True if one token is the exact negation of the other (e.g. 'no shimmer' vs 'shimmer').
    Returns: Boolean flag.
    """
    a = normalize_token(a, keep_hyphens=True)
    b = normalize_token(b, keep_hyphens=True)
    return (a.startswith("no ") and a[3:] == b) or (b.startswith("no ") and b[3:] == a)


# ─────────────────────────────────────────────────────────────
# 2. Embedded alias conflict
# ─────────────────────────────────────────────────────────────

def is_embedded_alias_conflict(longer: str, shorter: str) -> bool:
    """
    Does: Returns True if shorter alias is embedded in the longer string (e.g. 'rose' in 'rosewood').
    Used to prevent overlap or fuzzy alias collisions.
    """
    return shorter in longer and shorter != longer
