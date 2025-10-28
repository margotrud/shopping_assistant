# extraction/general/token/normalize.py
# ──────────────────────────────────────────────────────────────
# Shared utilities for token normalization and analysis
# ──────────────────────────────────────────────────────────────
"""
normalize.

Does: Provide safe singularization and deterministic token normalization
      (lowercasing, spacing, optional hyphen preservation) with light
      Unicode hygiene and domain-aware last-noun handling.
Returns: singularize(), normalize_token(), get_tokens_and_counts().
Used by: Tokenization pipelines and color/modifier extraction stages.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from color_sentiment_extractor.extraction.general.vocab.cosmetic_nouns import COSMETIC_NOUNS

__all__ = [
    "singularize",
    "normalize_token",
    "get_tokens_and_counts",
]

# Invariants (remain identical when singularized)
INVARIANT_SINGULARS: set[str] = {"series", "species"}

# Hyphen-aware token pattern (supports multi-segment hyphens: "blue-grey-green")
_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*", re.IGNORECASE)

# Safe plural endings we collapse in _singularize_word
_ES_PLURAL_RE = re.compile(r"(ches|shes|xes|zes|sses|oes)$", re.IGNORECASE)

# Common “fancy” Unicode punctuation we want to normalize early
_FANCY_HYPHENS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}  # ‐ - ‒ – — −
_FANCY_QUOTES = {"\u2018", "\u2019", "\u201b", "\u2032", "\u02bc"}  # ‘ ’ ‛ ′ ʼ


# ──────────────────────────────────────────────────────────────
# 0) Light Unicode hygiene
# ──────────────────────────────────────────────────────────────


def _unicode_hygiene(s: str) -> str:
    """
    Does: Apply light Unicode normalization:
          - NFKC fold
          - map fancy hyphens to ASCII '-'
          - map curly quotes to ASCII "'"
    Returns: Cleaned string (best-effort).
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in _FANCY_HYPHENS:
        s = s.replace(ch, "-")
    for ch in _FANCY_QUOTES:
        s = s.replace(ch, "'")
    return s


# ──────────────────────────────────────────────────────────────
# 1) SINGULARIZATION (safe rules + backward-compat API)
# ──────────────────────────────────────────────────────────────


def _singularize_word(w: str) -> str:
    """
    Does: Safe singular for English-like tokens:
          …ies→…y; …(ch|sh|x|z|ss|o)es→…; else …s→….
    Returns: Singularized token, lowercase.
    """
    if not isinstance(w, str):
        return ""
    w = w.lower().strip()
    if w in INVARIANT_SINGULARS or len(w) <= 3:
        return w

    # berries → berry
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"

    # boxes/patches/bushes/glosses/zeroes/heroes → box/patch/bush/gloss/zero/hero
    if w.endswith("es") and _ES_PLURAL_RE.search(w):
        return w[:-2]

    # nudes → nude ; 'ss' handled above
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]

    return w


def singularize(text: str) -> str:
    """
    Does: Back-compat: singularize a word; if phrase, only the last token.
    Returns: Input with final token singularized using safe rules.
    """
    if not isinstance(text, str):
        return ""
    s = _unicode_hygiene(text).strip()
    if not s:
        return s
    parts = s.split()
    parts[-1] = _singularize_word(parts[-1])
    return " ".join(parts)


def _singularize_phrase_if_cosmetic_last(text: str, *, keep_hyphens: bool) -> str:
    """
    Does: Singularize last cosmetic noun; if hyphenated and keep_hyphens=True,
          singularize its last hyphen segment only.
    Returns: Phrase with safe singularization applied when relevant.
    """
    parts = text.split()
    if not parts:
        return text

    last = parts[-1]

    def _is_cosmetic(tok: str) -> bool:
        # Cleanup to letters and hyphens; then check against vocab
        t = re.sub(r"[^a-z\-]", "", tok.lower())
        return t in COSMETIC_NOUNS or _singularize_word(t) in COSMETIC_NOUNS

    if keep_hyphens and "-" in last:
        segs = last.split("-")
        if segs and _is_cosmetic(segs[-1]):
            segs[-1] = _singularize_word(segs[-1])
            parts[-1] = "-".join(segs)
            return " ".join(parts)
        return text

    if _is_cosmetic(last):
        parts[-1] = _singularize_word(last)
        return " ".join(parts)

    return text


# ──────────────────────────────────────────────────────────────
# 2) TOKEN NORMALIZATION
# ──────────────────────────────────────────────────────────────


def normalize_token(token: str, keep_hyphens: bool = False) -> str:
    """
    Does: Normalize `token`:
          - Unicode hygiene (NFKC; map fancy hyphens/quotes)
          - lowercase + trim
          - '_' → space; collapse internal whitespace
          - hyphens kept & tightened (keep_hyphens=True) OR converted to spaces
          - singularize last cosmetic noun (domain-aware)
    Returns: Normalized token/phrase.
    """
    if not isinstance(token, str):
        return ""
    s = _unicode_hygiene(token).lower().strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s)

    if keep_hyphens:
        # Tighten hyphens (avoid spaces around)
        s = re.sub(r"\s*-\s*", "-", s)
        s = _singularize_phrase_if_cosmetic_last(s, keep_hyphens=True)
    else:
        # Convert hyphens to spaces, then re-collapse spacing
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        s = _singularize_phrase_if_cosmetic_last(s, keep_hyphens=False)

    return s


# ──────────────────────────────────────────────────────────────
# 3) TOKEN ANALYSIS
# ──────────────────────────────────────────────────────────────


def get_tokens_and_counts(text: str, keep_hyphens: bool = False) -> dict[str, int]:
    """
    Does: Extract tokens (hyphen-aware), normalize each, split on spaces,
          and count occurrences.
    Returns: Dict[token → count].
    """
    if not isinstance(text, str):
        return {}
    cleaned = _unicode_hygiene(text).lower()
    raw_tokens = _TOKEN_RE.findall(cleaned)

    normed = [normalize_token(t, keep_hyphens=keep_hyphens) for t in raw_tokens]

    flat: list[str] = []
    for n in normed:
        flat.extend(n.split())

    return dict(Counter(flat))
