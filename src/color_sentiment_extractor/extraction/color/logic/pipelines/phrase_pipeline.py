"""
phrase_pipeline.py
==================

Pipeline logic for resolving color-related tokens after extraction.
Includes phrase extraction (compound/standalone), confidence checks,
and RGB mapping via the rgb_pipeline.

Used By:
--------
- Compound extraction recovery
- Standalone color fallback
- Segment-level RGB aggregation
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import spacy

# Prefer rapidfuzz (faster). Fallback to fuzzywuzzy if not installed.
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    from fuzzywuzzy import fuzz  # type: ignore

from color_sentiment_extractor.extraction.color.constants import (
    BLOCKED_TOKENS,
    COSMETIC_NOUNS,
)
from color_sentiment_extractor.extraction.color.logic.pipelines.rgb_pipeline import (
    process_color_phrase,
)
from color_sentiment_extractor.extraction.color.strategies.compound import (
    extract_compound_phrases,
)
from color_sentiment_extractor.extraction.color.strategies.standalone import (
    extract_lone_tones,
    extract_standalone_phrases,
)
from color_sentiment_extractor.extraction.general.token.base_recovery import (
    recover_base,
)
from color_sentiment_extractor.extraction.general.utils.load_config import load_config

# =============================================================================
# Globals
# =============================================================================

logger = logging.getLogger(__name__)

# Load spaCy *once* at module import for performance
nlp = spacy.load("en_core_web_sm")

# Config used by some extractors (validated dict)
expression_map = load_config("expression_definition", mode="validated_dict")


# =============================================================================
# Color Phrase Extraction
# =============================================================================
def extract_all_descriptive_color_phrases(
    text: str,
    known_tones: Set[str],
    known_modifiers: Set[str],
    all_webcolor_names: Set[str],
    expression_map: dict,
    llm_client=None,
    debug: bool = False,
) -> List[str]:
    """
    Does: Full pipeline to extract all valid descriptive color phrases from raw input.
          Combines compound, standalone, and lone tone extraction strategies.
    Returns: Sorted list of extracted color phrases (lowercased, deduplicated).
    """
    tokens = nlp(text)

    phrases: Set[str] = set()
    segments: List[Tuple[str, str]] = []  # (modifier, tone) if you need them later

    extract_compound_phrases(
        tokens=tokens,
        compounds=phrases,
        raw_compounds=segments,
        known_color_tokens=known_tones,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        all_webcolor_names=all_webcolor_names,
        raw_text=text,
        debug=debug,
    )

    phrases.update(
        extract_standalone_phrases(
            tokens, known_modifiers, known_tones, expression_map, llm_client, debug
        )
    )
    phrases.update(extract_lone_tones(tokens, known_tones, debug))

    # Deterministic output
    return sorted({p.lower() for p in phrases})


def extract_phrases_from_segment(
    segment: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    expression_map: dict,
    llm_client=None,
    cache=None,
    debug: bool = False,
) -> Set[str]:
    """
    Does: Extracts valid color-related phrases from a text segment by identifying
          (modifier + tone) or standalone tone matches, validating them via
          base-token recovery and confidence checks, and filtering out
          cosmetic noun endings, blocked token pairs, and redundant singles.

    Returns: A set of validated color phrases found in the segment.
    """
    raw_results = extract_all_descriptive_color_phrases(
        segment,
        known_tones,
        known_modifiers,
        all_webcolor_names,
        expression_map,
        llm_client,
        debug,
    )

    def is_high_confidence(raw: str, base: str) -> bool:
        raw_l, base_l = raw.lower(), base.lower()
        blocked = (raw_l, base_l) in BLOCKED_TOKENS or (base_l, raw_l) in BLOCKED_TOKENS
        ratio = fuzz.ratio(raw_l, base_l)
        if raw_l != base_l and (blocked or ratio < 80):
            if debug:
                logger.debug(
                    "[BLOCKED FUZZY] raw=%r base=%r ratio=%s blocked=%s",
                    raw_l,
                    base_l,
                    ratio,
                    blocked,
                )
            return False
        if debug:
            logger.debug(
                "[CONF CHECK] raw=%r base=%r ratio=%s blocked=%s",
                raw_l,
                base_l,
                ratio,
                blocked,
            )
        return True

    valid_results: Set[str] = set()

    for phrase in raw_results:
        tokens = phrase.split()
        if debug:
            logger.debug("[CHECK PHRASE] %r tokens=%s", phrase, tokens)

        if len(tokens) == 1:
            base = recover_base(
                tokens[0],
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
            )
            blocked = (tokens[0], base) in BLOCKED_TOKENS or (base, tokens[0]) in BLOCKED_TOKENS
            if (
                base in known_tones
                and is_high_confidence(tokens[0], base)
                and not blocked
                and tokens[0] == base  # avoid keeping fuzzy-altered tone names
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
                (base1 in known_modifiers and base2 in known_tones)
                or (base1 in known_tones and base2 in known_modifiers)
                or (base1 in known_tones and base2 in known_tones)
            )

            if debug:
                logger.debug(
                    "[PAIR] %r + %r → base1=%r base2=%r",
                    tokens[0],
                    tokens[1],
                    base1,
                    base2,
                )

            if (
                valid_roles
                and is_high_confidence(tokens[0], base1)
                and is_high_confidence(tokens[1], base2)
            ):
                valid_results.add(phrase)

    # 🔒 Filter 1: remove phrases that end in cosmetic nouns
    valid_results = {
        phrase for phrase in valid_results if phrase.split()[-1] not in COSMETIC_NOUNS
    }

    # 🔒 Filter 2: remove standalone tokens used inside compound phrases
    compound_tokens = {t for p in valid_results for t in p.split() if len(p.split()) > 1}
    valid_results = {
        p for p in valid_results if len(p.split()) > 1 or p not in compound_tokens
    }

    if debug:
        logger.debug("[FINAL VALID RESULTS] %s", valid_results)

    return valid_results


# =============================================================================
# Segment-Level Processing
# =============================================================================
def process_segment_colors(
    color_phrases: List[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client=None,
    cache=None,
    debug: bool = False,
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
                debug=debug,
            )
        except Exception as e:  # pragma: no cover (diagnostic path)
            if debug:
                import inspect
                import sys
                import traceback

                logger.exception("[process_color_phrase ERROR] %r: %s", phrase, e)
                logger.debug(
                    "process_color_phrase.__module__=%s file=%s",
                    process_color_phrase.__module__,
                    inspect.getsourcefile(process_color_phrase),
                )
                logger.debug("re in sys.modules? %s", "re" in sys.modules)
            simple, rgb = "", None

        # Guard: never return None for simplified
        if not simple:
            if debug:
                logger.debug("[SKIP] No simplification returned for phrase %r", phrase)
            simple = ""

        if rgb is False:  # just in case upstream returns False
            rgb = None

        simplified.append(simple)
        rgb_list.append(rgb)

    return simplified, rgb_list


def aggregate_color_phrase_results(
    segments: List[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    llm_client=None,
    cache=None,
    debug: bool = False,
) -> Tuple[Set[str], List[str], Dict[str, Tuple[int, int, int]]]:
    """
    Does: Aggregates simplified tone names and RGB values from color phrases in all segments.
    Returns: (set of simplified tone names, list of all simplified phrases, RGB map for each phrase).
    """

    def tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^\w\-]+", text.lower()) if t]

    def build_allowed_tokens(seg: str) -> Set[str]:
        raw_tokens = tokenize(seg)
        allowed: Set[str] = set(raw_tokens)
        # allow both hyphenated and spaced variants
        for t in list(raw_tokens):
            if "-" in t:
                allowed.add(t.replace("-", " "))
        for t in raw_tokens:
            base = recover_base(
                t,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
            )
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
            known_tones=known_tones,
            all_webcolor_names=all_webcolor_names,
            expression_map=expression_map,
            llm_client=llm_client,
            debug=debug,
        )

        color_phrases = {p for p in extracted if phrase_ok(p, allowed)}

        simplified, rgb_list = process_segment_colors(
            color_phrases=sorted(color_phrases),  # stable order
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            llm_client=llm_client,
            cache=cache,
            debug=debug,
        )

        for original, simple, rgb in zip(sorted(color_phrases), simplified, rgb_list):
            # If simplification changes a known tone, trust original tone
            if simple != original and original in known_tones:
                simple = original
            # If original wasn't a known tone and simplification changed it,
            # distrust the RGB (it may belong to a different phrase now)
            if simple != original and original not in known_tones:
                rgb = None

            if phrase_ok(simple, allowed):
                all_phrases.append(simple)
                tone_set.add(simple)
                if rgb is not None:
                    rgb_map[simple] = rgb

    return tone_set, all_phrases, rgb_map
