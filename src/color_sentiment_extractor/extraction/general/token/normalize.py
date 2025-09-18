# ──────────────────────────────────────────────────────────────
# Chatbot/extraction/general/token/normalize.py
# Shared utilities for token normalization and analysis
# ──────────────────────────────────────────────────────────────

import re
from collections import Counter
from extraction.general.vocab.cosmetic_nouns import COSMETIC_NOUNS

INVARIANT_SINGULARS = {"series", "species"}  # restent identiques au singulier


# ──────────────────────────────────────────────────────────────
# 1) SINGULARIZATION (safe rules + backward-compat API)
# ──────────────────────────────────────────────────────────────

def _singularize_word(w: str) -> str:
    """
    Does: Safe singular for English-like tokens (ies→y; …es→…; else …s→…).
    Returns: Singularized token, lowercase.
    """
    if not isinstance(w, str):
        return ""
    w = w.lower().strip()
    if w in INVARIANT_SINGULARS:
        return w

    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"                  # berries -> berry
    if re.search(r"(sh|ch|x|z|ss|oes)$", w):
        # glosses/boxes/patches/bushes/zeroes -> gloss/box/patch/bush/zero
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]                         # nudes -> nude
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


def _singularize_phrase_if_cosmetic_last(text: str) -> str:
    """
    Does: Singularize last token only if it's a cosmetic noun (by vocab).
    Returns: Phrase with safe singularization applied when relevant.
    """
    parts = text.split()
    if not parts:
        return text
    last = parts[-1]
    last_clean = re.sub(r"[^a-z\-]", "", last.lower())
    if last_clean in COSMETIC_NOUNS or _singularize_word(last_clean) in COSMETIC_NOUNS:
        parts[-1] = _singularize_word(last)
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# 2) TOKEN NORMALIZATION
# ──────────────────────────────────────────────────────────────
def normalize_token(token: str, keep_hyphens: bool = False) -> str:
    """
    Does: Lowercases, trims, '_'→space; hyphens kept/tightened (keep_hyphens) or spaced.
    Also singularizes the last cosmetic noun (e.g., 'soft-pinks'→'soft pink').
    Returns: Normalized token/phrase.
    """
    if not isinstance(token, str):
        return ""
    s = token.lower().strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    if keep_hyphens:
        s = re.sub(r"\s*-\s*", "-", s)       # tighten hyphens
    else:
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
    return _singularize_phrase_if_cosmetic_last(s)

# ──────────────────────────────────────────────────────────────
# 3) TOKEN ANALYSIS
# ──────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)?")

def get_tokens_and_counts(text: str, keep_hyphens: bool = False) -> dict[str, int]:
    if not isinstance(text, str):
        return {}
    raw = _TOKEN_RE.findall(text.lower())
    normed = [normalize_token(t, keep_hyphens=keep_hyphens) for t in raw]
    flat = []
    for n in normed:
        flat.extend(n.split())
    return dict(Counter(flat))
