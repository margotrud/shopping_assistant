# ──────────────────────────────────────────────────────────────
# Chatbot/extraction/general/token/normalize.py
# Shared utilities for token normalization and analysis
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from collections import Counter
from typing import Dict

from color_sentiment_extractor.extraction.general.vocab.cosmetic_nouns import COSMETIC_NOUNS

INVARIANT_SINGULARS = {"series", "species"}  # restent identiques au singulier


# ──────────────────────────────────────────────────────────────
# 1) SINGULARIZATION (safe rules + backward-compat API)
# ──────────────────────────────────────────────────────────────

def _singularize_word(w: str) -> str:
    """
    Does: Safe singular for English-like tokens (…ies→…y; …(ch|sh|x|z|ss|o)es→…; else …s→…).
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
    if w.endswith("es"):
        if re.search(r"(ches|shes|xes|zes|sses|oes)$", w):
            return w[:-2]

    # nudes → nude ; but don't slice 'ss' plurals twice (handled above)
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
    s = text.strip()
    if not s:
        return s
    parts = s.split()
    parts[-1] = _singularize_word(parts[-1])
    return " ".join(parts)


def _singularize_phrase_if_cosmetic_last(text: str, *, keep_hyphens: bool) -> str:
    """
    Does: Singularize last cosmetic noun; if hyphenated and keep_hyphens, singularize its last segment only.
    Returns: Phrase with safe singularization applied when relevant.
    """
    parts = text.split()
    if not parts:
        return text

    last = parts[-1]
    # Helper to check membership against vocab after basic cleanup
    def _is_cosmetic(tok: str) -> bool:
        t = re.sub(r"[^a-z\-]", "", tok.lower())
        return t in COSMETIC_NOUNS or _singularize_word(t) in COSMETIC_NOUNS

    if keep_hyphens and "-" in last:
        segs = last.split("-")
        # On singularise la dernière sous-partie seulement si c’est un nom cosmétique
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
    Does: Lowercases, trims, '_'→space; hyphens kept/tightened (keep_hyphens) or spaced; singularize last cosmetic noun.
    Returns: Normalized token/phrase.
    """
    if not isinstance(token, str):
        return ""
    s = token.lower().strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    if keep_hyphens:
        s = re.sub(r"\s*-\s*", "-", s)  # tighten hyphens
        s = _singularize_phrase_if_cosmetic_last(s, keep_hyphens=True)
    else:
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        s = _singularize_phrase_if_cosmetic_last(s, keep_hyphens=False)
    return s


# ──────────────────────────────────────────────────────────────
# 3) TOKEN ANALYSIS
# ──────────────────────────────────────────────────────────────

# Supporte plusieurs segments hyphénés: "blue-grey-green"
_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

def get_tokens_and_counts(text: str, keep_hyphens: bool = False) -> Dict[str, int]:
    """
    Does: Extracts tokens (hyphen-aware), normalizes each, splits on spaces, and counts occurrences.
    Returns: Dict[token → count].
    """
    if not isinstance(text, str):
        return {}
    raw = _TOKEN_RE.findall(text.lower())
    normed = [normalize_token(t, keep_hyphens=keep_hyphens) for t in raw]
    flat: list[str] = []
    for n in normed:
        flat.extend(n.split())
    return dict(Counter(flat))
