"""
Sentiment Detection & Clause Splitting
--------------------------------------
Handles:
- Hybrid sentiment detection (VADER first, BART MNLI fallback)
- Optional negation scope with negspaCy (NegEx) if available
- Soft-negation override (e.g. "not too shiny")
- Clause splitting (e.g. "I like pink but not red")
- Single source of truth for BART candidate labels (no duplication)
- Lazy-loading of heavy components (spaCy, VADER, BART)
- Logging (no print), configurable VADER thresholds, LRU cache
- High-level API: analyze_sentence_sentiment(sentence) → [{clause, polarity, separator}]
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import List, Dict, Tuple, Optional

import spacy
from transformers import pipeline

# VADER (lazy download & init in get_vader())
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from color_sentiment_extractor.extraction.general.token.normalize import normalize_token

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

def _dbg(enabled: bool, *args):
    """Internal debug helper that respects logger config."""
    if enabled:
        logger.debug(" ".join(str(a) for a in args))

# ─────────────────────────────────────────────
# Config: VADER thresholds (tunable)
# ─────────────────────────────────────────────
VADER_POS_TH: float = 0.05
VADER_NEG_TH: float = -0.05

# ─────────────────────────────────────────────
# Single source of truth for BART candidates
# ─────────────────────────────────────────────
_BART_CANDIDATES: List[Tuple[str, str]] = [
    ("I like this", "positive"),
    ("I dislike this", "negative"),
    ("I'm unsure or neutral", "neutral"),
]
_CANDIDATE_LABELS: List[str] = [t for (t, _) in _BART_CANDIDATES]
_CANDIDATE_SENTIMENTS: List[str] = [s for (_, s) in _BART_CANDIDATES]

def _map_bart_label_to_sentiment(label: str) -> str:
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
_negex_ready = None  # tri-state: None (unknown), True (available), False (not installed)

_SENTIMENT_MODEL_NAME = "facebook/bart-large-mnli"

def get_nlp():
    """Safe loader: prefer en_core_web_sm, fallback to blank('en')"""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("spaCy model 'en_core_web_sm' not found; falling back to blank('en').")
            _nlp = spacy.blank("en")
    return _nlp

def get_vader():
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        _vader = SentimentIntensityAnalyzer()
    return _vader

def get_sentiment_pipeline():
    """Zero-shot classifier with device auto-detection (GPU if available)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
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
    Try to attach negspaCy 'negex' pipe if available.
    Returns the active nlp with negex (or plain nlp if not available).
    """
    global _negex_ready
    nlp = get_nlp()
    if _negex_ready is False:
        return nlp
    if _negex_ready is True and "negex" in nlp.pipe_names:
        return nlp
    # Probe availability
    try:
        from negspacy.negation import Negex  # type: ignore
        if "negex" not in nlp.pipe_names:
            # Prefer general English rules for non-clinical data
            nlp.add_pipe("negex", config={"language": "en"})
        _negex_ready = True
    except Exception:
        _negex_ready = False
    return nlp

# ─────────────────────────────────────────────
# LRU cache for repeated VADER scoring
# ─────────────────────────────────────────────
@lru_cache(maxsize=2048)
def _vader_score(text: str) -> float:
    """Cache VADER scores with light whitespace normalization (preserve case)."""
    try:
        key = re.sub(r"\s+", " ", text.strip())
        return get_vader().polarity_scores(key)["compound"]
    except Exception:
        logger.exception("VADER scoring failed")
        return 0.0

# ─────────────────────────────────────────────
# Public API: classify segments into pos/neg
# ─────────────────────────────────────────────
def classify_segments_by_sentiment_no_neutral(has_splitter: bool, segments: List[str]) -> Dict[str, List[str]]:
    """
    Classifies segments into positive/negative using VADER + BART (fallback).
    If a segment resolves to 'neutral', prefer negation cues to decide negative,
    else default to positive.
    """
    classification = {"positive": [], "negative": []}
    for seg in segments:
        try:
            base = detect_sentiment(seg)          # hybrid (VADER → BART)
            mapped = map_sentiment(base, seg)     # negation override + VADER re-check
            final = mapped
            if final == "neutral":
                hn, sn = is_negated_or_soft(seg, debug=False)
                final = "negative" if (hn or sn) else "positive"
            classification[final].append(seg)
        except Exception:
            logger.exception("SENTIMENT ERROR on segment: %r", seg)
    return classification

# ─────────────────────────────────────────────
# Hybrid sentiment detection (with DI for tests)
# ─────────────────────────────────────────────
def detect_sentiment(text: str, vader: Optional[SentimentIntensityAnalyzer] = None, bart=None) -> str:
    """
    Detects sentiment using a hybrid approach:
    1) VADER for quick lexical scoring (handles 'love', 'fan of', 'hate', etc.)
    2) BART MNLI fallback for ambiguous cases (candidate labels set above)
    Returns: 'positive' | 'negative' | 'neutral'

    DI hooks:
      - vader: pass a fake/fixture to avoid loading the real model in tests
      - bart: pass a fake/fixture (callable like HF pipeline) in tests
    """
    try:
        v = vader or get_vader()
        score = (_vader_score(text) if vader is None else v.polarity_scores(text)["compound"])
        if score >= VADER_POS_TH:
            return "positive"
        elif score <= VADER_NEG_TH:
            return "negative"

        # Fallback to BART if near-neutral
        pipe = bart or get_sentiment_pipeline()
        result = pipe(text, _CANDIDATE_LABELS)
        top_label = result["labels"][0]
        return _map_bart_label_to_sentiment(top_label)

    except Exception:
        logger.exception("detect_sentiment failed")
        return "neutral"

def map_sentiment(predicted: str, text: str) -> str:
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
def contains_sentiment_splitter_with_segments(text: str) -> Tuple[bool, List[str]]:
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
    index = find_splitter_index(doc)

    # 2a) Guard: if the split is caused by a sentence-initial ADV (advmod at i=0),
    #            only allow it when it's a configured discourse adverb.
    if index == 0 and len(doc) > 0 and doc[0].dep_ == "advmod":
        try:
            # Correct package path; lazy import to avoid heavy imports at module load
            from color_sentiment_extractor.extraction.general.utils.load_config import load_config  # type: ignore
            discourse_adverbs = load_config("discourse_adverbs", mode="set")
        except Exception:
            discourse_adverbs = set()
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
    has_or = any(normalize_token(tok.text, keep_hyphens=True) == "or" for tok in doc)
    has_punct = any(tok.text in {".", ",", ";"} for tok in doc)
    return has_neg and has_or and not has_punct

def find_splitter_index(doc) -> Optional[int]:
    """
    Heuristics to find a clause-level splitter token.
    Prioritizes coordinating conjunctions / markers, but skips when joining tones.
    """
    for i, tok in enumerate(doc):
        if tok.dep_ == "cc" and tok.text.lower() in {"and", "or"}:
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
    def dbg(*args): _dbg(debug, "[is_tone_conjunction]", *args)

    # Bounds
    if index < 0 or index >= len(doc):
        dbg(f"Index {index} out of bounds → False"); return False

    tok = doc[index]
    prev = doc[index - 1] if index > 0 else None
    next_ = doc[index + 1] if index + 1 < len(doc) else None

    dbg(f"doc='{doc.text}'")
    dbg(f"center idx={index} text='{tok.text}' pos={tok.pos_} tag={tok.tag_} morph={tok.morph}")

    if not prev or not next_:
        dbg("Missing prev/next neighbor → False"); return False

    center_text = tok.text.lower()
    conj_like = (center_text in {"and", "or", "&", "/"} or tok.pos_ == "CCONJ" or tok.tag_ in {"CC", "CCONJ"})
    dbg(f"conj_like={conj_like}")

    def is_adj_or_noun_like(t):
        low, pos, tag, morph, dep = t.text.lower(), t.pos_, t.tag_, t.morph, t.dep_

        if pos == "ADV":
            return False, {"text": t.text, "pos": pos, "reason": "pos==ADV → reject"}

        is_adj, is_nounish = (pos == "ADJ"), (pos in {"NOUN", "PROPN"})
        verbform = morph.get("VerbForm")
        is_participle     = (pos in {"VERB","AUX"}) and (tag == "VBN" or ("Part" in verbform if verbform else False))
        is_past_ed_as_adj = (pos in {"VERB","AUX"}) and tag == "VBD" and low.endswith("ed")

        # Reject plain verbs (unless post overrides)
        if pos in {"VERB","AUX"} and not (is_participle or is_past_ed_as_adj):
            return False, {"text": t.text, "pos": pos, "tag": tag, "morph": str(morph), "reason": "plain VERB/AUX → reject"}

        has_adj_suffix = low.endswith("ish") or (low.endswith("y") and not low.endswith("ly"))
        likely_descr = (
            t.is_alpha and len(low) <= 8 and
            pos not in {"PRON","DET","ADP","AUX","ADV","PART","SCONJ","PUNCT","SYM","NUM"} and
            dep not in {"advmod"}
        )

        like = is_adj or is_nounish or is_participle or is_past_ed_as_adj or has_adj_suffix or likely_descr
        info = {
            "text": t.text, "lemma": t.lemma_, "pos": pos, "tag": tag, "morph": str(morph), "dep": dep,
            "adj": is_adj, "nounish": is_nounish, "participle": is_participle,
            "past_ed_as_adj": is_past_ed_as_adj, "suffix_adj": has_adj_suffix,
            "likely_descriptive": likely_descr, "adj_or_noun_like": like,
        }
        return like, info

    prev_ok, prev_info = is_adj_or_noun_like(prev)
    next_ok, next_info = is_adj_or_noun_like(next_)
    dbg("prev:", prev_info); dbg("next:", next_info)

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
            if (prev.is_alpha and len(p) >= 5 and not p.endswith(("ing","ly"))
                and next_.is_alpha and len(n) >= 5 and not n.endswith(("ing","ly"))):
                dbg("override: 'X or Y' base-form noun/adj-like → True"); return True
        if center_text == "and":
            if (prev.is_alpha and p.endswith("y") and not p.endswith("ly") and len(p) >= 5
                and next_.is_alpha and n.endswith("y") and not n.endswith("ly") and len(n) >= 5):
                dbg("override: paired '-y' nouns/adjectives in 'X and Y' → True"); return True

    dbg(f"→ result={result}")
    return result


def split_text_on_index(doc, i: int) -> List[str]:
    """
    Does: Splits a spaCy doc around token i; if i is first→return segments from the right (split on ; or ,),
          if i is last→return segments from the left (split on ; or ,), else return [left_text, right_text].
    Returns: List[str] of trimmed segments; empty segments are removed.
    """
    # i = premier token → on prend ce qu'il y a APRÈS
    if i == 0:
        remaining = " ".join(tok.text for tok in doc[i + 1:]).strip()
        return [seg.strip() for seg in re.split(r"[;,]", remaining) if seg.strip()]

    # i = dernier token → on prend ce qu'il y a AVANT
    if i == len(doc) - 1:
        remaining = " ".join(tok.text for tok in doc[:i]).strip()
        return [seg.strip() for seg in re.split(r"[;,]", remaining) if seg.strip()]

    # i au milieu → split en deux morceaux (avant / après)
    left = doc[:i].text.strip()
    right = doc[i + 1:].text.strip()
    return [left, right]

def fallback_split_on_punctuation(text: str) -> Tuple[bool, List[str]]:
    """
    Split on punctuation as a last resort. If only one segment is found,
    return no-split.
    """
    segments = [s.strip() for s in re.split(r"[.;,]", text) if s.strip()]
    return (True, segments) if len(segments) >= 2 else (False, [text.strip()])

# ─────────────────────────────────────────────
# Sentence-level API with separators
# ─────────────────────────────────────────────
def _split_sentence_with_separators(text: str, _doc=None) -> Tuple[List[str], List[Optional[str]]]:
    """
    Returns (clauses, separators). Heuristic:
      - use dependency splitter if found → separator = token text at split
      - else split on punctuation and keep punctuation as separator tokens
      - if no split → single clause, separator=None
    """
    nlp = get_nlp()
    doc = _doc if _doc is not None else nlp(text)

    # 1) Try dependency-based split
    idx = find_splitter_index(doc)
    if idx is not None:
        sep = doc[idx].text.strip()
        left = doc[:idx].text.strip()
        right = doc[idx + 1:].text.strip()
        # Guard: if one side is empty, treat as unreliable and fallback to punctuation
        if left and right:
            return [left, right], [sep, None]
        # else fall through to punctuation fallback

    # 2) Rescue split (handles cases where is_tone_conjunction suppressed a valid split)
    #    Example: "Choose matte or choose glossy" → split on 'or'
    #    Only trigger when at least one neighbor of 'or' is VERB/AUX to avoid descriptor pairs.
    for j, tok in enumerate(doc):
        if tok.text.lower() == "or":
            prev_tok = doc[j - 1] if j - 1 >= 0 else None
            next_tok = doc[j + 1] if j + 1 < len(doc) else None
            if prev_tok is None or next_tok is None:
                continue
            if (prev_tok.pos_ in {"VERB", "AUX"}) or (next_tok.pos_ in {"VERB", "AUX"}):
                left = doc[:j].text.strip()
                right = doc[j + 1:].text.strip()
                if left and right:  # only accept if both sides are non-empty
                    return [left, right], ["or", None]

    # 3) Fallback to punctuation
    parts = re.split(r"([.;,])", text)
    clauses: List[str] = []
    seps: List[Optional[str]] = []
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

def analyze_sentence_sentiment(sentence: str) -> List[Dict[str, Optional[str]]]:
    """
    High-level API: takes a full sentence and returns:
      [{"clause": str, "polarity": "positive|negative|neutral", "separator": str|None}, ...]
    """
    try:
        nlp = get_nlp()
        doc = nlp(sentence)
        clauses, separators = _split_sentence_with_separators(sentence, _doc=doc)
        out: List[Dict[str, Optional[str]]] = []
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
def is_negated_or_soft(text: str, debug: bool = False) -> Tuple[bool, bool]:
    """
    Returns (hard_negation, soft_negation), mutually exclusive.
      - Soft motif: (neg cue) + 'too' + ADJ
        neg cue := Polarity=Neg OR lemma in {'not','no'} OR PRON/ADV starting with 'no'
      - Hard cues (outside any soft motif):
          R1: token.dep_ == 'neg'
          R2: token.morph.Polarity == 'Neg'
          R3: 'no' as DET (dep=det) of ADJ/NOUN/PROPN (e.g., 'no good', 'no problem')
          R4: PRON starting with 'no' as core argument (nsubj/obj/attr, etc.)
    """
    _dbg(debug, f"\n[INPUT] {text!r}")

    nlp = ensure_negex()
    doc = nlp(text)

    if debug:
        _dbg(True, "\n[TOKENS DEBUG]")
        for i, tok in enumerate(doc):
            _dbg(True,
                 f"{i:02d} | text={tok.text!r:12} lemma={tok.lemma_!r:12} "
                 f"dep={tok.dep_:10} pos={tok.pos_:6} tag={tok.tag_:6} "
                 f"morph={tok.morph} Polarity={tok.morph.get('Polarity')}")

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
    soft_start = soft_end = None
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i + 1], doc[i + 2]
        neg_cue = (
            t1.morph.get("Polarity") == ["Neg"]
            or t1.lemma_.lower() in {"not", "no"}
            or (t1.pos_ in {"PRON", "ADV"} and t1.text.lower().startswith("no"))
        )
        motif = neg_cue and (t2.lower_ == "too") and (t3.pos_ == "ADJ")
        if debug:
            why = (
                "Polarity"
                if t1.morph.get("Polarity") == ["Neg"]
                else ("lemma" if t1.lemma_.lower() in {"not", "no"} else "pron/adv_no*")
            )
            _dbg(True, f"[SOFT CHECK] window=({t1.text}, {t2.text}, {t3.text}) "
                       f"→ neg_cue={why} match={motif}")
        if motif:
            soft_neg = True
            soft_start, soft_end = i, i + 2
            soft_shield_idxs.update({i, i+1, i+2})
            if debug:
                _dbg(True, f"[SOFT SHIELD] shielding indices {sorted(soft_shield_idxs)} "
                           f"for tokens {[doc[j].text for j in sorted(soft_shield_idxs)]}")
            break

    # --- HARD cues (check only OUTSIDE soft window if soft exists) ---
    fired = None
    def outside_soft(idx: int) -> bool:
        return not soft_neg or idx < soft_start or idx > soft_end

    # R1: dep=neg
    for i, tok in enumerate(doc):
        if tok.dep_ == "neg" and outside_soft(i):
            fired = ("R1_dep_neg", i, tok.text); break
    # R2: Polarity=Neg
    if not fired:
        for i, tok in enumerate(doc):
            if tok.morph.get("Polarity") == ["Neg"] and outside_soft(i):
                fired = ("R2_morph_Polarity", i, tok.text); break
    # R3: 'no' determiner of ADJ/NOUN/PROPN
    if not fired:
        for i, tok in enumerate(doc):
            if tok.lemma_ == "no" and tok.pos_ == "DET" and tok.dep_ == "det" and outside_soft(i):
                if tok.head.pos_ in {"ADJ", "NOUN", "PROPN"}:
                    fired = ("R3_det_no", i, tok.text); break
    # R4: negative PRON as core argument
    if not fired:
        CORE_ARGS = {"nsubj", "nsubjpass", "obj", "dobj", "pobj", "attr"}
        for i, tok in enumerate(doc):
            if tok.pos_ == "PRON" and tok.text.lower().startswith("no") and tok.dep_ in CORE_ARGS and outside_soft(i):
                fired = ("R4_neg_pron_core", i, tok.text); break

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
            _dbg(True, f"[HARD-NEG RULE FIRED] {fired[0]} by token #{fired[1]} -> {fired[2]!r}")

    return hard_neg, soft_neg

# Backwards-compatible helpers retained (if you call them elsewhere)
def is_negated(text: str) -> bool:
    return is_negated_or_soft(text)[0]

def is_softly_negated(text: str) -> bool:
    return is_negated_or_soft(text)[1]
