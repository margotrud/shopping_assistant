# src/color_sentiment_extractor/extraction/general/sentiment/core.py

"""
Sentiment Detection & Clause Splitting.
--------------------------------------
Does:
- Hybrid sentiment detection (VADER first, BART MNLI fallback)
- Optional negation scope with negspaCy (NegEx) if available
- Soft-negation override (e.g. "not too shiny")
- Clause splitting (e.g. "I like pink but not red")
- Single source of truth for BART candidate labels (no duplication)
- Lazy-loading of heavy components (spaCy, VADER, BART)
- Logging (no print), configurable VADER thresholds, LRU cache
- High-level API: analyze_sentence_sentiment(sentence) → list[ClauseResult]
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
)

# VADER (lazy download & init in get_vader())
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

from color_sentiment_extractor.extraction.general.token import (
    normalize_token,  # kept for reuse elsewhere
)

__docformat__ = "google"

# ─────────────────────────────────────────────
# Typage & surface publique
# ─────────────────────────────────────────────
Sentiment = Literal["positive", "negative", "neutral"]


class ClauseResult(TypedDict):
    clause: str
    polarity: Sentiment
    separator: str | None


# result type for zero-shot pipeline
class ZeroShotResult(TypedDict):
    labels: list[str]
    scores: list[float]


# Buckets pour classify_segments_by_sentiment_no_neutral()
class SentimentBuckets(TypedDict):
    positive: list[str]
    negative: list[str]


__all__ = [
    "analyze_sentence_sentiment",
    "classify_segments_by_sentiment_no_neutral",
    "detect_sentiment",
    "map_sentiment",
    "is_negated_or_soft",
]

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


def _dbg(enabled: bool, *args: Any) -> None:
    """Internal debug helper that respects logger config."""
    if enabled:
        logger.debug(" ".join(str(a) for a in args))


# ─────────────────────────────────────────────
# Config via ENV: thresholds & feature switches
# ─────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in {"0", "false", "False", ""}


VADER_POS_TH: float = _env_float("SENT_POS_TH", 0.05)
VADER_NEG_TH: float = _env_float("SENT_NEG_TH", -0.05)
USE_NEGEX: bool = _env_bool("USE_NEGEX", True)
ALLOW_NLTK_DOWNLOAD: bool = _env_bool("ALLOW_NLTK_DOWNLOAD", False)
USE_FAKE_BART: bool = _env_bool("BART_MNLI_FAKE", False)

# ─────────────────────────────────────────────
# Single source of truth for BART candidates
# (order is contractuel)
# ─────────────────────────────────────────────
_BART_CANDIDATES: list[tuple[str, Sentiment]] = [
    ("I like this", "positive"),
    ("I dislike this", "negative"),
    ("I'm unsure or neutral", "neutral"),
]
_CANDIDATE_LABELS: list[str] = [t for (t, _) in _BART_CANDIDATES]
_CANDIDATE_SENTIMENTS: list[Sentiment] = [s for (_, s) in _BART_CANDIDATES]


def _map_bart_label_to_sentiment(label: str) -> Sentiment:
    """Maps the top BART textual label back to the intended sentiment without a duplicate dict."""
    try:
        i = _CANDIDATE_LABELS.index(label)
        return _CANDIDATE_SENTIMENTS[i]
    except ValueError:
        lab_low = label.lower()
        for idx, cand in enumerate(_CANDIDATE_LABELS):
            if cand.lower() == lab_low:
                return _CANDIDATE_SENTIMENTS[idx]
        return "neutral"


# ─────────────────────────────────────────────
# Lazy loaders (avoid heavy init at import time)
# ─────────────────────────────────────────────
_nlp = None
_vader = None
_sentiment_pipeline = None
_negex_ready: bool | None = None  # tri-state: None (unknown), True (available),
# False (not installed)

_SENTIMENT_MODEL_NAME = "facebook/bart-large-mnli"


def get_nlp():
    """Safe loader: prefer en_core_web_sm, fallback to blank('en')."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("spaCy model 'en_core_web_sm' not found; falling back to blank('en').")
            _nlp = spacy.blank("en")
    return _nlp


def get_vader():
    """Lazily initialize VADER. Respects ALLOW_NLTK_DOWNLOAD to avoid downloading in prod."""
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError as err:  # ← capture l'erreur d'origine
            if ALLOW_NLTK_DOWNLOAD:
                logger.info("VADER lexicon not found. Downloading because ALLOW_NLTK_DOWNLOAD=1.")
                nltk.download("vader_lexicon", quiet=True)
            else:
                msg = (
                    "VADER lexicon not found and ALLOW_NLTK_DOWNLOAD=0. "
                    "Please pre-install NLTK data."
                )
                logger.error(msg)
                raise RuntimeError(msg) from err  # ← chaînage explicite
        _vader = SentimentIntensityAnalyzer()
    return _vader


class _FakeZeroShot:
    """Tiny offline replacement for tests. Returns a deterministic label order."""

    def __call__(self, text: str, labels: list[str]) -> ZeroShotResult:
        low = text.lower()
        # naive rules just for tests
        if any(k in low for k in ("love", "like", "fan of", "great", "good", "amazing")):
            order = [labels[0], labels[2], labels[1]]  # positive > neutral > negative
        elif any(k in low for k in ("hate", "dislike", "awful", "bad", "terrible")):
            order = [labels[1], labels[2], labels[0]]  # negative > neutral > positive
        else:
            order = [labels[2], labels[0], labels[1]]  # neutral > positive > negative
        # Scores là juste pour debug/test -> floats
        return ZeroShotResult(labels=order, scores=[0.9, 0.08, 0.02])


def get_sentiment_pipeline():
    """Zero-shot classifier with device auto-detection (GPU if available),
    or fake offline pipeline.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        if USE_FAKE_BART:
            logger.info("Using _FakeZeroShot pipeline (BART_MNLI_FAKE=1).")
            _sentiment_pipeline = _FakeZeroShot()
        else:
            try:
                import torch  # optional

                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1
            _sentiment_pipeline = pipeline(
                "zero-shot-classification",
                model=_SENTIMENT_MODEL_NAME,
                device=device,
            )
    return _sentiment_pipeline


def ensure_negex():
    """
    Try to attach negspaCy 'negex' pipe if available and USE_NEGEX is True.
    Returns the active nlp with negex (or plain nlp if not available).
    """
    global _negex_ready
    nlp = get_nlp()
    if not USE_NEGEX:
        _negex_ready = False
        return nlp

    if _negex_ready is False:
        return nlp
    if _negex_ready is True and "negex" in nlp.pipe_names:
        return nlp

    # Probe availability
    try:
        # We do a TYPE_CHECKING split so mypy won't yell about missing import,
        # but at runtime we still import the real thing.
        if TYPE_CHECKING:
            from negspacy.negation import Negex as _Negex  # noqa: F401
        else:
            from negspacy.negation import Negex  # noqa: F401
        if "negex" not in nlp.pipe_names:
            # Prefer general English rules for non-clinical data
            nlp.add_pipe("negex", config={"language": "en"})
        _negex_ready = True
    except Exception:
        _negex_ready = False
    return nlp


# ─────────────────────────────────────────────
# Precompiled regex
# ─────────────────────────────────────────────
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPLIT_RE = re.compile(r"([.;,])")


# ─────────────────────────────────────────────
# LRU cache for repeated VADER scoring
# ─────────────────────────────────────────────
@lru_cache(maxsize=2048)
def _vader_score(text: str) -> float:
    """Cache VADER scores with light whitespace normalization (preserve case)."""
    try:
        key = _WHITESPACE_RE.sub(" ", text.strip())
        return get_vader().polarity_scores(key)["compound"]
    except Exception:
        logger.exception("VADER scoring failed")
        return 0.0


# ─────────────────────────────────────────────
# Public API: classify segments into pos/neg
# ─────────────────────────────────────────────
def classify_segments_by_sentiment_no_neutral(
    has_splitter: bool, segments: list[str]
) -> SentimentBuckets:
    """
    Classifies segments into positive/negative using VADER + BART (fallback).
    If a segment resolves to 'neutral', prefer negation cues to decide negative,
    else default to positive.
    """
    classification: SentimentBuckets = {"positive": [], "negative": []}
    for seg in segments:
        try:
            base = detect_sentiment(seg)  # hybrid (VADER → BART)
            mapped = map_sentiment(base, seg)  # negation override + VADER re-check
            final: Sentiment = mapped
            if final == "neutral":
                hn, sn = is_negated_or_soft(seg, debug=False)
                final = "negative" if (hn or sn) else "positive"
            # Only two buckets in this function:
            classification["negative" if final == "negative" else "positive"].append(seg)
        except Exception:
            logger.exception("SENTIMENT ERROR on segment: %r", seg, exc_info=True)
    return classification


# ─────────────────────────────────────────────
# Hybrid sentiment detection (with DI for tests)
# ─────────────────────────────────────────────
def detect_sentiment(
    text: str, vader: SentimentIntensityAnalyzer | None = None, bart: Any | None = None
) -> Sentiment:
    """Detects sentiment using a hybrid approach.

    1) VADER for quick lexical scoring (handles 'love', 'fan of', 'hate', etc.)
    2) BART MNLI fallback for ambiguous cases (candidate labels set above)

    Returns:
        'positive' | 'negative' | 'neutral'

    DI hooks:
      - vader: pass a fake/fixture to avoid loading the real model in tests
      - bart: pass a fake/fixture (callable like HF pipeline) in tests.
    """
    try:
        v = vader or get_vader()
        score = _vader_score(text) if vader is None else v.polarity_scores(text)["compound"]
        if score >= VADER_POS_TH:
            return "positive"
        elif score <= VADER_NEG_TH:
            return "negative"

        # Fallback to BART if near-neutral
        pipe = bart or get_sentiment_pipeline()
        # Deterministic: rely on candidate labels order; single-label decision.
        result = pipe(text, _CANDIDATE_LABELS)
        top_label = result["labels"][0]
        return _map_bart_label_to_sentiment(top_label)

    except Exception:
        logger.exception("detect_sentiment failed")
        return "neutral"


def map_sentiment(predicted: Sentiment, text: str) -> Sentiment:
    """
    Post-processes the predicted sentiment by:
    - Overriding with negation (hard/soft) if present (negation wins),
    - Returning predicted if already non-neutral,
    - For true neutrals, re-checking with VADER and deciding,
    - Falling back to 'neutral' if still ambiguous.
    """
    hard_neg, soft_neg = is_negated_or_soft(text, debug=False)
    if hard_neg or soft_neg:
        return "negative"

    if predicted != "neutral":
        return predicted

    # Neutral case → try VADER one more time
    score = _vader_score(text)
    if score >= VADER_POS_TH:
        return "positive"
    elif score <= VADER_NEG_TH:
        return "negative"

    return "neutral"


# ─────────────────────────────────────────────
# Clause Splitting
# ─────────────────────────────────────────────
def contains_sentiment_splitter_with_segments(text: str) -> tuple[bool, list[str]]:
    """
    Detects whether the sentence contains a clause-level sentiment split
    and returns segmented parts. Uses dependency parsing and punctuation fallback.

    Returns:
        (has_split: bool, segments: List[str])
    """
    doc = get_nlp()(text)

    # 1) Skip pattern: negation + "or" + no punctuation
    if should_skip_split_due_to_or_negation(doc):
        return False, [text.strip()]

    # 2) Try dependency-based splitter
    index = find_clause_splitter_index(doc)

    # 2a) Guard: if the split is caused by a sentence-initial ADV (advmod at i=0),
    #            only allow it when it's a configured discourse adverb.
    if index == 0 and len(doc) > 0 and doc[0].dep_ == "advmod":
        try:
            # Correct package path; lazy import to avoid heavy imports at module load
            from color_sentiment_extractor.extraction.general.utils.load_config import load_config

            discourse_adverbs: frozenset[str] = load_config("discourse_adverbs", mode="set")
        except Exception:
            discourse_adverbs = frozenset()
        if normalize_token(doc[0].text, keep_hyphens=True) not in discourse_adverbs:
            index = None  # ignore this splitter and fall back later

    if index is not None:
        parts = split_text_on_index(doc, index)
        # If the split produces only one non-empty part (e.g., splitter at start/end),
        # treat it as no reliable split for sentiment purposes.
        if len(parts) >= 2:
            return True, parts
        else:
            return False, [text.strip()]

    # 3) Fallback: punctuation-based split
    return fallback_split_on_punctuation(text)


def should_skip_split_due_to_or_negation(doc) -> bool:
    has_neg = any(tok.dep_ == "neg" for tok in doc)
    # leaner than normalize_token for this single check:
    has_or = any(tok.lower_ == "or" for tok in doc)
    has_punct = any(tok.text in {".", ",", ";"} for tok in doc)
    return has_neg and has_or and not has_punct


def find_clause_splitter_index(doc) -> int | None:
    """
    Heuristics to find a clause-level splitter token (cc/mark/discourse).
    Skips conjunctions that connect two tone-like descriptors (see is_tone_conjunction).
    """
    for i, tok in enumerate(doc):
        if tok.dep_ == "cc" and tok.lower_ in {"and", "or"}:
            if is_tone_conjunction(doc, i, debug=False):
                continue
        if tok.dep_ in {"cc", "mark", "discourse"}:
            return i
        # Early adverbial marker at sentence start (rare but handled)
        if tok.dep_ == "advmod" and i == 0:
            return i
    return None


def is_tone_conjunction(doc, index: int, antonym_fn=None, debug: bool = False) -> bool:
    """
    Avoid splitting when a conjunction connects two tone-like tokens.
    If `antonym_fn` is provided, it will be used to reject pairs of ADJ that are antonyms.
    """

    def dbg(*args):
        _dbg(debug, "[is_tone_conjunction]", *args)

    # Bounds
    if index < 0 or index >= len(doc):
        dbg(f"Index {index} out of bounds → False")
        return False

    tok = doc[index]
    prev = doc[index - 1] if index > 0 else None
    next_ = doc[index + 1] if index + 1 < len(doc) else None

    dbg(f"doc='{doc.text}'")
    dbg(f"center idx={index} text='{tok.text}' pos={tok.pos_} tag={tok.tag_} morph={tok.morph}")

    if not prev or not next_:
        dbg("Missing prev/next neighbor → False")
        return False

    center_text = tok.lower_
    conj_like = (
        center_text in {"and", "or", "&", "/"} or tok.pos_ == "CCONJ" or tok.tag_ in {"CC", "CCONJ"}
    )
    dbg(f"conj_like={conj_like}")

    def is_adj_or_noun_like(t):
        low, pos, tag, morph, dep = t.lower_, t.pos_, t.tag_, t.morph, t.dep_

        if pos == "ADV":
            return (
                False,
                {"text": t.text, "pos": pos, "reason": "pos==ADV → reject"},
            )

        is_adj = pos == "ADJ"
        is_nounish = pos in {"NOUN", "PROPN"}
        verbform = morph.get("VerbForm")
        is_participle = pos in {"VERB", "AUX"} and (
            tag == "VBN" or ("Part" in verbform if verbform else False)
        )
        is_past_ed_as_adj = pos in {"VERB", "AUX"} and tag == "VBD" and low.endswith("ed")

        # Reject plain verbs (unless post overrides)
        if pos in {"VERB", "AUX"} and not (is_participle or is_past_ed_as_adj):
            return (
                False,
                {
                    "text": t.text,
                    "pos": pos,
                    "tag": tag,
                    "morph": str(morph),
                    "reason": "plain VERB/AUX → reject",
                },
            )

        has_adj_suffix = low.endswith("ish") or (low.endswith("y") and not low.endswith("ly"))
        likely_descr = (
            t.is_alpha
            and len(low) <= 8
            and pos
            not in {
                "PRON",
                "DET",
                "ADP",
                "AUX",
                "ADV",
                "PART",
                "SCONJ",
                "PUNCT",
                "SYM",
                "NUM",
            }
            and dep not in {"advmod"}
        )

        like = (
            is_adj
            or is_nounish
            or is_participle
            or is_past_ed_as_adj
            or has_adj_suffix
            or likely_descr
        )
        info = {
            "text": t.text,
            "lemma": t.lemma_,
            "pos": pos,
            "tag": tag,
            "morph": str(morph),
            "dep": dep,
            "adj": is_adj,
            "nounish": is_nounish,
            "participle": is_participle,
            "past_ed_as_adj": is_past_ed_as_adj,
            "suffix_adj": has_adj_suffix,
        }

        return like, info

    prev_ok, prev_info = is_adj_or_noun_like(prev)
    next_ok, next_info = is_adj_or_noun_like(next_)
    dbg("prev:", prev_info)
    dbg("next:", next_info)

    result = bool(conj_like and prev_ok and next_ok)

    # Antonym gate via dependency injection
    if result and prev.pos_ == "ADJ" and next_.pos_ == "ADJ":
        if antonym_fn is None:
            dbg("antonym gate: are_antonyms not available → skipping")
        else:
            try:
                if antonym_fn(prev.lemma_.lower(), next_.lemma_.lower()):
                    dbg("antonym gate: neighbors are antonyms → False")
                    return False
            except Exception as e:
                dbg(f"antonym check failed ({e}); ignoring")

    # Post overrides (unchanged)
    if conj_like and not result:
        p, n = prev.text.lower(), next_.text.lower()
        if center_text == "or":
            if (
                prev.is_alpha
                and len(p) >= 5
                and not p.endswith(("ing", "ly"))
                and next_.is_alpha
                and len(n) >= 5
                and not n.endswith(("ing", "ly"))
            ):
                dbg("override: 'X or Y' base-form noun/adj-like → True")
                return True
        if center_text == "and":
            if (
                prev.is_alpha
                and p.endswith("y")
                and not p.endswith("ly")
                and len(p) >= 5
                and next_.is_alpha
                and n.endswith("y")
                and not n.endswith("ly")
                and len(n) >= 5
            ):
                dbg("override: paired '-y' nouns/adjectives in 'X and Y' → True")
                return True

    dbg(f"→ result={result}")
    return result


def split_text_on_index(doc, i: int) -> list[str]:
    """
    Does: Splits a spaCy doc around token i.
          If i is first → return segments from the right (split on ; or ,),
          if i is last → return segments from the left (split on ; or ,),
          else return [left_text, right_text].

    Returns: List[str] of trimmed segments; empty segments are removed.
    """
    # i = premier token → on prend ce qu'il y a APRÈS
    if i == 0:
        remaining = " ".join(tok.text for tok in doc[i + 1 :]).strip()
        return [seg.strip() for seg in re.split(r"[;,]", remaining) if seg.strip()]

    # i = dernier token → on prend ce qu'il y a AVANT
    if i == len(doc) - 1:
        remaining = " ".join(tok.text for tok in doc[:i]).strip()
        return [seg.strip() for seg in re.split(r"[;,]", remaining) if seg.strip()]

    # i au milieu → split en deux morceaux (avant / après)
    left = doc[:i].text.strip()
    right = doc[i + 1 :].text.strip()
    return [left, right]


def fallback_split_on_punctuation(text: str) -> tuple[bool, list[str]]:
    """
    Split on punctuation as a last resort. If only one segment is found,
    return no-split.
    """
    segments = [
        s.strip() for s in _PUNCT_SPLIT_RE.split(text) if s and s.strip() not in {".", ";", ","}
    ]
    # The splitting above keeps separators in the list; we filtered them out.
    return (True, segments) if len(segments) >= 2 else (False, [text.strip()])


# ─────────────────────────────────────────────
# Sentence-level API with separators
# ─────────────────────────────────────────────
def _split_sentence_with_separators(text: str, _doc=None) -> tuple[list[str], list[str | None]]:
    """
    Returns (clauses, separators). Heuristic.

    - use dependency splitter if found → separator = token text at split.
    - else split on punctuation and keep punctuation as separator tokens.
    - if no split → single clause, separator=None.
    """
    nlp = get_nlp()
    doc = _doc if _doc is not None else nlp(text)

    # 1) Try dependency-based split
    idx = find_clause_splitter_index(doc)
    if idx is not None:
        sep = doc[idx].text.strip()
        left = doc[:idx].text.strip()
        right = doc[idx + 1 :].text.strip()
        # Guard: if one side is empty, treat as unreliable and fallback to punctuation
        if left and right:
            return [left, right], [sep, None]
        # else fall through to punctuation fallback

    # 2) Rescue split (handles cases where is_tone_conjunction suppressed a valid split)
    #    Example: "Choose matte or choose glossy" → split on 'or'
    #    Only trigger when at least one neighbor of 'or' is VERB/AUX to avoid descriptor pairs.
    for j, tok in enumerate(doc):
        if tok.lower_ == "or":
            prev_tok = doc[j - 1] if j - 1 >= 0 else None
            next_tok = doc[j + 1] if j + 1 < len(doc) else None
            if prev_tok is None or next_tok is None:
                continue
            if (prev_tok.pos_ in {"VERB", "AUX"}) or (next_tok.pos_ in {"VERB", "AUX"}):
                left = doc[:j].text.strip()
                right = doc[j + 1 :].text.strip()
                if left and right:  # only accept if both sides are non-empty
                    return [left, right], ["or", None]

    # 3) Fallback to punctuation (preserve separators)
    parts = _PUNCT_SPLIT_RE.split(text)
    clauses: list[str] = []
    seps: list[str | None] = []
    current = ""
    for part in parts:
        if part and part.strip() in {".", ";", ","}:
            if current.strip():
                clauses.append(current.strip())
                seps.append(part.strip())
                current = ""
        else:
            current += part
    if current.strip():
        clauses.append(current.strip())
        seps.append(None)
    if not clauses:
        return [text.strip()], [None]
    return clauses, seps


def analyze_sentence_sentiment(sentence: str) -> list[ClauseResult]:
    """High-level API: takes a full sentence and returns.

    [{"clause": str, "polarity": "positive|negative|neutral", "separator": str|None}, ...]
    """
    try:
        nlp = get_nlp()
        doc = nlp(sentence)
        clauses, separators = _split_sentence_with_separators(sentence, _doc=doc)
        out: list[ClauseResult] = []
        for i, clause in enumerate(clauses):
            base = detect_sentiment(clause)
            pol = map_sentiment(base, clause)
            sep = separators[i] if i < len(separators) else None
            out.append({"clause": clause, "polarity": pol, "separator": sep})
        return out
    except Exception:
        logger.exception("analyze_sentence_sentiment failed: %r", sentence)
        # Fallback: single neutral clause
        return [{"clause": sentence, "polarity": "neutral", "separator": None}]


# ─────────────────────────────────────────────
# Negation Detection (negspaCy optional + soft negation)
# ─────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _is_negated_or_soft_cached(text: str) -> tuple[bool, bool]:
    """Cached core for is_negated_or_soft when debug=False."""
    return _is_negated_or_soft_core(text, debug=False)


def is_negated_or_soft(text: str, debug: bool = False) -> tuple[bool, bool]:
    """Returns (hard_negation, soft_negation), mutually exclusive.

    - Soft motif: (neg cue) + 'too' + ADJ
      neg cue := Polarity=Neg OR lemma in {'not','no'} OR PRON/ADV starting with 'no'
    - Hard cues (outside any soft motif):
        R1: token.dep_ == 'neg'
        R2: token.morph.Polarity == 'Neg'
        R3: 'no' as DET (dep=det) of ADJ/NOUN/PROPN (e.g., 'no good', 'no problem')
        R4: PRON starting with 'no' as core argument (nsubj/obj/attr, etc.)
    """
    if not debug:
        return _is_negated_or_soft_cached(text)
    return _is_negated_or_soft_core(text, debug=True)


def _is_negated_or_soft_core(text: str, debug: bool = False) -> tuple[bool, bool]:
    _dbg(debug, f"\n[INPUT] {text!r}")

    nlp = ensure_negex()
    doc = nlp(text)

    if debug:
        _dbg(True, "\n[TOKENS DEBUG]")
        for i, tok in enumerate(doc):
            _dbg(
                True,
                f"{i:02d} | text={tok.text!r:12} lemma={tok.lemma_!r:12} "
                f"dep={tok.dep_:10} pos={tok.pos_:6} tag={tok.tag_:6} "
                f"morph={tok.morph} Polarity={tok.morph.get('Polarity')}",
            )

    # negspaCy (reference only)
    try:
        negex_flag = bool(getattr(doc._, "negex", False))
    except Exception as e:
        negex_flag = False
        _dbg(debug, f"[negspaCy] Could not retrieve negex flag: {e}")
    _dbg(debug, f"[negspaCy span flag] {negex_flag}")

    # --- SOFT motif: (neg cue) + 'too' + ADJ ---
    soft_neg = False
    soft_shield_idxs: set[int] = set()
    soft_start: int | None = None
    soft_end: int | None = None
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i + 1], doc[i + 2]
        neg_cue = (
            t1.morph.get("Polarity") == ["Neg"]
            or t1.lemma_.lower() in {"not", "no"}
            or (t1.pos_ in {"PRON", "ADV"} and t1.lower_.startswith("no"))
        )
        motif = neg_cue and (t2.lower_ == "too") and (t3.pos_ == "ADJ")
        if debug:
            why = (
                "Polarity"
                if t1.morph.get("Polarity") == ["Neg"]
                else ("lemma" if t1.lemma_.lower() in {"not", "no"} else "pron/adv_no*")
            )
            _dbg(
                True,
                f"[SOFT CHECK] window=({t1.text}, {t2.text}, {t3.text}) "
                f"→ neg_cue={why} match={motif}",
            )
        if motif:
            soft_neg = True
            soft_start, soft_end = i, i + 2
            soft_shield_idxs.update({i, i + 1, i + 2})
            if debug:
                _dbg(
                    True,
                    f"[SOFT SHIELD] shielding indices {sorted(soft_shield_idxs)} "
                    f"for tokens {[doc[j].text for j in sorted(soft_shield_idxs)]}",
                )
            break

    # helper pour checker si un index est hors du span soft
    def outside_soft(idx: int) -> bool:
        if not soft_neg:
            return True
        assert soft_start is not None and soft_end is not None
        return (idx < soft_start) or (idx > soft_end)

    # --- HARD cues (check only OUTSIDE soft window if soft exists) ---
    fired: tuple[str, int, str] | None = None

    # R1: dep=neg
    for i, tok in enumerate(doc):
        if tok.dep_ == "neg" and outside_soft(i):
            fired = ("R1_dep_neg", i, tok.text)
            break
    # R2: Polarity=Neg
    if not fired:
        for i, tok in enumerate(doc):
            if tok.morph.get("Polarity") == ["Neg"] and outside_soft(i):
                fired = ("R2_morph_Polarity", i, tok.text)
                break
    # R3: 'no' determiner of ADJ/NOUN/PROPN
    if not fired:
        for i, tok in enumerate(doc):
            if tok.lemma_ == "no" and tok.pos_ == "DET" and tok.dep_ == "det" and outside_soft(i):
                if tok.head.pos_ in {"ADJ", "NOUN", "PROPN"}:
                    fired = ("R3_det_no", i, tok.text)
                    break
    # R4: negative PRON as core argument
    if not fired:
        CORE_ARGS = {"nsubj", "nsubjpass", "obj", "dobj", "pobj", "attr"}
        for i, tok in enumerate(doc):
            if (
                tok.pos_ == "PRON"
                and tok.lower_.startswith("no")
                and tok.dep_ in CORE_ARGS
                and outside_soft(i)
            ):
                fired = ("R4_neg_pron_core", i, tok.text)
                break

    # --- Decision: either hard or soft ---
    if fired:
        hard_neg, soft_neg = True, False
    elif soft_neg:
        hard_neg, soft_neg = False, True
    else:
        hard_neg, soft_neg = False, False

    if debug:
        _dbg(True, f"[RESULT] hard_neg={hard_neg}, soft_neg={soft_neg}")
        if fired:
            _dbg(
                True,
                f"[HARD-NEG RULE FIRED] {fired[0]} by token #{fired[1]} -> {fired[2]!r}",
            )

    return hard_neg, soft_neg


# Backwards-compatible helpers retained (if you call them elsewhere)
def is_negated(text: str) -> bool:
    return is_negated_or_soft(text)[0]


def is_softly_negated(text: str) -> bool:
    return is_negated_or_soft(text)[1]
