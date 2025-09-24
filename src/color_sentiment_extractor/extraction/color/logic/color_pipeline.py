"""
color_pipeline.py
=================

Pipeline logic for resolving color-related tokens after extraction.
Includes fallback resolution, LLM simplification, and RGB mapping.

Used By:
--------
- Compound extraction recovery
- Standalone color fallback
- Segment-level RGB aggregation
"""

# =============================================================================
# Imports
# =============================================================================

from typing import Set, Tuple, Optional, List, Dict

import spacy

from color_sentiment_extractor.extraction.color.logic.rgb_pipeline import process_color_phrase, _sanitize_simplified
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

nlp = spacy.load("en_core_web_sm")

from color_sentiment_extractor.extraction.color.extraction.compound import extract_compound_phrases
from color_sentiment_extractor.extraction.color.extraction.standalone import (
    extract_standalone_phrases,
    extract_lone_tones,
)

from color_sentiment_extractor.extraction.color.vocab import known_tones
from color_sentiment_extractor.extraction.color.constants import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.general.utils.load_config import load_config
expression_map = load_config("expression_definition", mode="validated_dict")
KNOWN_TONES = known_tones
import re
# =============================================================================
# color Phrase Extraction
# =============================================================================

def extract_all_descriptive_color_phrases(
    text: str,
    known_tones: Set[str],
    known_modifiers: Set[str],
    all_webcolor_names: Set[str],
    expression_map: dict,
    llm_client=None,
    debug: bool = False
) -> List[str]:
    """
    Does: Full pipeline to extract all valid descriptive color phrases from raw input.
          Combines compound, standalone, and lone tone extraction strategies.
    Returns: List of extracted color phrases (lowercased, deduplicated).
    """
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)

    phrases = set()
    segments = []  # Will hold (modifier, tone) tuples

    extract_compound_phrases(
        tokens=tokens,
        compounds=phrases,
        raw_compounds=segments,
        known_color_tokens=known_tones,  # or whatever is correct in your context
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        all_webcolor_names=all_webcolor_names,
        raw_text=text,
        debug=debug
    )

    phrases.update(extract_standalone_phrases(tokens, known_modifiers, known_tones, expression_map, llm_client, debug))
    phrases.update(extract_lone_tones(tokens, known_tones, debug))

    return list(set(map(str.lower, phrases)))

from color_sentiment_extractor.extraction.color.constants import BLOCKED_TOKENS
from fuzzywuzzy import fuzz

def extract_phrases_from_segment(
    segment: str,
    known_modifiers: set,
    known_tones: set,
    all_webcolor_names: set,
    expression_map: dict,
    llm_client=None,
    cache=None,
    debug=True
):
    """
       Does: Extracts valid color-related phrases from a text segment by identifying
             (modifier + tone) or standalone tone matches, validating them via
             base-token recovery and fuzzy confidence checks, and filtering out
             cosmetic noun endings, blocked token pairs, and redundant single tokens.
       Returns: A set of validated color phrases found in the segment, preserving only
                high-confidence, non-conflicting matches.
       """
    raw_results = extract_all_descriptive_color_phrases(
        segment,
        known_tones,
        known_modifiers,
        all_webcolor_names,
        expression_map,
        llm_client,
        debug
    )

    def is_high_confidence(raw, base):
        raw, base = raw.lower(), base.lower()
        blocked = (raw, base) in BLOCKED_TOKENS or (base, raw) in BLOCKED_TOKENS
        ratio = fuzz.ratio(raw, base)

        # Disallow fuzzy if blocked or too weak
        if raw != base and (blocked or ratio < 80):
            if debug:
                print(f"[🚫 BLOCKED FUZZY] raw='{raw}' → base='{base}' | ratio={ratio} | blocked={blocked}")
            return False

        if debug:
            print(f"[🔍 CONFIDENCE CHECK] raw='{raw}' → base='{base}' | ratio={ratio} | blocked={blocked}")
        return True

    valid_results = set()

    for phrase in raw_results:
        tokens = phrase.split()
        if debug:
            print(f"\n[🧪 CHECKING PHRASE] '{phrase}' | Tokens: {tokens}")

        if len(tokens) == 1:
            base = recover_base(tokens[0], known_modifiers=known_modifiers, known_tones=KNOWN_TONES, debug=False, fuzzy_fallback=False)
            blocked = (tokens[0], base) in BLOCKED_TOKENS or (base, tokens[0]) in BLOCKED_TOKENS
            if debug:
                print(f"[🔍 SINGLE TOKEN] '{tokens[0]}' → base='{base}' | blocked={blocked}")
            if (
                    base in known_tones and
                    is_high_confidence(tokens[0], base) and
                    not blocked and
                    tokens[0].lower() == base.lower()  # 💥 NEW: avoid fuzzy match if raw != base
            ):
                valid_results.add(phrase)



        elif len(tokens) == 2:
            base1 = recover_base(
                tokens[0],
                known_modifiers=known_modifiers,
                known_tones=known_tones,
            )

            base2 = recover_base(
                tokens[1],
                known_modifiers=known_modifiers,
                known_tones=known_tones,
            )

            valid_roles = (
                (base1 in known_modifiers and base2 in known_tones) or
                (base1 in known_tones and base2 in known_modifiers) or
                (base1 in known_tones and base2 in known_tones)
            )

            if debug:
                print(f"[🔍 PAIR] '{tokens[0]}' + '{tokens[1]}' → base1='{base1}', base2='{base2}'")

            if (
                valid_roles and
                is_high_confidence(tokens[0], base1) and
                is_high_confidence(tokens[1], base2)
            ):
                valid_results.add(phrase)

    # 🔒 Filter 1: remove phrases that end in cosmetic nouns
    valid_results = {
        phrase for phrase in valid_results
        if phrase.split()[-1] not in COSMETIC_NOUNS
    }

    # 🔒 Filter 2: remove standalone tokens used inside compound phrases
    compound_tokens = set()
    for phrase in valid_results:
        tokens = phrase.split()
        if len(tokens) > 1:
            compound_tokens.update(tokens)

    valid_results = {
        phrase for phrase in valid_results
        if len(phrase.split()) > 1 or phrase not in compound_tokens
    }

    if debug:
        print(f"\n✅ FINAL VALID RESULTS: {valid_results}")

    # # 🔒 Filter 3: keep only phrases that exist in original segment
    # valid_results = {
    #     phrase for phrase in valid_results
    #     if phrase in segment.lower()
    # }

    return valid_results

# =============================================================================
# Single Phrase RGB Resolution
# =============================================================================


# =============================================================================
# Fallback Resolution for Missed Tokens
# =============================================================================



# =============================================================================
# Segment-Level and Batch Processing
# =============================================================================

def process_segment_colors(
    color_phrases: List[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client=None,
    cache=None,
    debug: bool = False
) -> Tuple[List[str], List[Optional[Tuple[int, int, int]]]]:
    """
    Does: Processes a list of raw color phrases through simplification + RGB resolution pipeline.
    Returns: Tuple of (list of simplified phrases, list of RGB tuples or None).
    """
    simplified: List[str] = []
    rgb_list: List[Optional[Tuple[int, int, int]]] = []

    for phrase in color_phrases:
        try:
            simple, rgb = process_color_phrase(
                phrase,
                known_modifiers,
                known_tones,
                llm_client=llm_client,
                cache=cache,
                debug=debug
            )


        except Exception as e:
            if debug:
                import inspect, sys, traceback
                print("[DEBUG] process_color_phrase.__module__ =", process_color_phrase.__module__)
                print("[DEBUG] process_color_phrase file       =", inspect.getsourcefile(process_color_phrase))
                print("[DEBUG] _sanitize_simplified.__module__ =",
                      _sanitize_simplified.__module__ if '_sanitize_simplified' in globals() else None)
                print("[DEBUG] re in sys.modules?               =", 're' in sys.modules)
                print("[DEBUG] re in globals()?                 =", 're' in globals())
                print(f"[⛔ process_color_phrase ERROR] '{phrase}': {e}")
                traceback.print_exc()  # <<<<<< imprime le fichier+ligne exacts

            simple, rgb = "", None

        # 👇 Garde-fous : ne jamais renvoyer None côté 'simple'
        if not simple:  # couvre None, "" ou valeur falsy
            if debug:
                print(f"[⛔ SKIP] No simplification returned for phrase '{phrase}'")
            simple = ""   # évite les .split() sur None plus loin

        # Normalise aussi rgb si besoin
        if rgb is False:  # au cas où la pipeline renverrait False
            rgb = None

        simplified.append(simple)
        rgb_list.append(rgb)

    return simplified, rgb_list


def aggregate_color_phrase_results(
    segments: List[str],
    known_modifiers: Set[str],
    all_webcolor_names: Set[str],
    llm_client,
    cache=None,
    debug: bool = True
) -> Tuple[Set[str], List[str], Dict[str, Tuple[int, int, int]]]:
    """
    Does: Aggregates simplified tone names and RGB values from color phrases in all segments.
    Returns: (set of simplified tone names, list of all simplified phrases, RGB map for each phrase).
    """
    from color_sentiment_extractor.extraction.general.utils.load_config import load_config
    from color_sentiment_extractor.extraction.color.vocab import known_tones as KNOWN_TONES
    from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

    expression_map = load_config("expression_definition")

    def tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"[^\w\-]+", text.lower()) if t]

    def build_allowed_tokens(seg: str) -> Set[str]:
        raw_tokens = tokenize(seg)
        allowed: Set[str] = set(raw_tokens)
        for t in list(raw_tokens):
            if "-" in t:
                allowed.add(t.replace("-", " "))
        for t in raw_tokens:
            base = recover_base(t, known_modifiers=known_modifiers, known_tones=KNOWN_TONES, debug=False, fuzzy_fallback=False)
            if base:
                allowed.add(base)
                if "-" in base:
                    allowed.add(base.replace("-", " "))
        return allowed

    def phrase_ok(phrase: str, allowed: Set[str]) -> bool:
        words = phrase.lower().split()
        return all(w in allowed for w in words)

    tone_set: Set[str] = set()
    all_phrases: List[str] = []
    rgb_map: Dict[str, Tuple[int, int, int]] = {}

    for seg in segments:
        allowed = build_allowed_tokens(seg)

        extracted = extract_phrases_from_segment(
            seg,
            known_modifiers=known_modifiers,
            known_tones=KNOWN_TONES,
            all_webcolor_names=all_webcolor_names,
            expression_map=expression_map,
            debug=debug
        )

        color_phrases = {p for p in extracted if phrase_ok(p, allowed)}

        simplified, rgb_list = process_segment_colors(
            color_phrases=color_phrases,
            known_modifiers=known_modifiers,
            known_tones=KNOWN_TONES,
            llm_client=llm_client,
            cache=cache,
            debug=debug
        )

        for original, simple, rgb in zip(color_phrases, simplified, rgb_list):
            if simple != original and original in KNOWN_TONES:
                simple = original
            if simple != original and original not in KNOWN_TONES:
                rgb = None
            if phrase_ok(simple, allowed):
                all_phrases.append(simple)
                tone_set.add(simple)
                rgb_map[simple] = rgb

    return tone_set, all_phrases, rgb_map
