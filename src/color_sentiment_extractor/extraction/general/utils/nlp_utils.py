"""
nlp_utils.py

Does: Minimal NLP helpers for antonym checking (WordNet) and token lemmatization (spaCy).
Returns: are_antonyms() → bool, lemmatize_token() → str.
Used by: sentiment routing, extraction filters, and negation handling.
"""

from functools import lru_cache
from typing import Optional, Set, Iterable

from nltk.corpus import wordnet
from color_sentiment_extractor.extraction.general.token import normalize_token

__all__ = ["are_antonyms", "lemmatize_token"]

# ── Antonym checking (WordNet) ───────────────────────────────────────────────
def _candidates(word: str) -> Iterable[str]:
    """Does: Generate lookup candidates (normalized + raw lowercase). Returns: iterable of str."""
    candidates: list[str] = []
    try:
        norm = normalize_token(word, keep_hyphens=True)
    except Exception:
        norm = ""
    raw = (word or "").lower().strip()
    for w in (norm, raw):
        if w and (not candidates or w != candidates[-1]):
            candidates.append(w)
    return candidates

@lru_cache(maxsize=10_000)
def _normalized_antonyms(word: str) -> Set[str]:
    """Does: Return normalized antonyms for word via WordNet. Returns: set[str]."""
    out: Set[str] = set()
    try:
        for cand in _candidates(word):
            for syn in wordnet.synsets(cand):
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        out.add(normalize_token(ant.name().replace("_", ""), keep_hyphens=True))
    except LookupError:
        return set()
    return out

@lru_cache(maxsize=10_000)
def are_antonyms(word1: str, word2: str) -> bool:
    """
    Does: True iff one word is in the normalized antonyms of the other.
    Returns: bool
    """
    variants1, variants2 = list(_candidates(word1)), list(_candidates(word2))
    if not variants1 or not variants2:
        return False
    return any(v2 in _normalized_antonyms(v1) for v1 in variants1 for v2 in variants2) \
        or any(v1 in _normalized_antonyms(v2) for v2 in variants2 for v1 in variants1)

# ── Lemmatization (spaCy if available) ───────────────────────────────────────
_nlp: Optional[object] = None  # Will hold spaCy Language or False

def _get_spacy():
    """Does: Load spaCy en_core_web_sm once (lightweight). Returns: pipeline or None if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp or None
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
        except Exception:
            _nlp = False
    except Exception:
        _nlp = False
    return _nlp or None

@lru_cache(maxsize=20_000)
def lemmatize_token(token: str) -> str:
    """Does: Lemmatize token with spaCy if available, else return normalized form. Returns: str."""
    if not isinstance(token, str) or not token:
        return ""
    text = normalize_token(token, keep_hyphens=True)
    if not text:
        return ""
    nlp = _get_spacy()
    if not nlp:
        return text
    doc = nlp(text)
    return (doc[0].lemma_ or doc[0].text) if doc else text
