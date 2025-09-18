# Chatbot/extraction/general/utils/nlp_utils.py
"""
Minimal NLP helpers for the extraction pipeline: antonym check (WordNet) and token lemmatization (spaCy).
Relies on the project-wide normalizer and uses caching for performance.
"""


from functools import lru_cache
from typing import Optional, Set, Iterable

from nltk.corpus import wordnet
from ..token.normalize import normalize_token  # import relatif: general/utils -> general/token/normalize.py



# ──────────────────────────────────────────────────────────────
# Antonym checking (WordNet)
# ──────────────────────────────────────────────────────────────

def _candidates(word: str) -> Iterable[str]:
    """Generate lookup candidates: normalized then raw lowercase (dedup, non-empty)."""
    candidates = []
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
    """
    Does: Union of normalized antonyms for all candidate forms (normalized + raw lowercase).
    Returns: set[str].
    """
    out: Set[str] = set()
    try:
        for cand in _candidates(word):
            # Interroge WordNet pour chaque forme candidate
            for syn in wordnet.synsets(cand):
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        # normalise les antonymes exactement comme le pipeline (tolère underscores)
                        out.add(normalize_token(ant.name().replace("_", ""), keep_hyphens=True))
    except LookupError:
        return set()
    return out

@lru_cache(maxsize=10_000)
def are_antonyms(word1: str, word2: str) -> bool:
    """
    Does: True iff l’un des mots apparaît dans l’ensemble des antonymes normalisés de l’autre,
          en essayant à la fois la forme normalisée et la forme brute (lowercase).
    """
    # génère aussi les variantes du “côté comparé”
    variants1 = list(_candidates(word1))
    variants2 = list(_candidates(word2))
    if not variants1 or not variants2:
        return False

    # test symétrique : (chaque var2 ∈ antonymes(var1)) ou (chaque var1 ∈ antonymes(var2))
    for a in variants1:
        ants = _normalized_antonyms(a)
        if any(v2 in ants for v2 in variants2):
            return True
    for b in variants2:
        ants = _normalized_antonyms(b)
        if any(v1 in ants for v1 in variants1):
            return True
    return False

# ──────────────────────────────────────────────────────────────
# Lemmatization (spaCy if available)
# ──────────────────────────────────────────────────────────────

_nlp: Optional[object] = None  # None=unknown, False=unavailable, else spaCy pipeline

def _get_spacy():
    """
    Does: Load spaCy 'en_core_web_sm' (disabled: ner, parser, textcat) once; return None if unavailable.
    Returns: spaCy Language pipeline | None.
    """
    global _nlp
    if _nlp is not None:
        return _nlp or None
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
        except Exception:
            # No model installed → explicitly disable lemmatization instead of using a misleading blank model
            _nlp = False
    except Exception:
        _nlp = False
    return _nlp or None

@lru_cache(maxsize=20_000)
def lemmatize_token(token: str) -> str:
    """
    Does: Lemmatize a single token via spaCy when available; otherwise return the normalized token unchanged.
    Returns: str (lemma).
    """
    if not isinstance(token, str) or not token:
        return ""
    text = normalize_token(token, keep_hyphens=True)
    if not text:
        return ""
    nlp = _get_spacy()
    if not nlp:
        return text
    doc = nlp(text)
    if not doc:
        return text
    lemma = doc[0].lemma_ or doc[0].text
    return lemma or text

