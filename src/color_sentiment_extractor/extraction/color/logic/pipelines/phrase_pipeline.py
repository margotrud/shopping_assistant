"""
phrase_pipeline.py.
==================

Does: Extract, validate, simplify, and map RGB for descriptive color phrases across
text segments.
Returns: Extracted phrases (per segment), simplified tones, and a phrase→RGB map.
Used By: color.logic.pipelines.rgb_pipeline, higher-level sentiment/aggregation flows.
"""

from __future__ import annotations

# ── Imports ───────────────────────────────────────────────────────────────────
import logging
import re
from functools import lru_cache
from typing import (
    Protocol,
    cast,
    runtime_checkable,
)

import spacy
from spacy.language import Language

# Prefer rapidfuzz; fallback to fuzzywuzzy.
try:
    from rapidfuzz import fuzz as _fuzz

    ratio = _fuzz.ratio
except Exception:  # pragma: no cover
    from fuzzywuzzy import fuzz as _fuzz  # type: ignore

    ratio = _fuzz.ratio

from color_sentiment_extractor.extraction.color import BLOCKED_TOKENS, COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.strategies import (
    extract_compound_phrases,
    extract_lone_tones,
    extract_standalone_phrases,
)
from color_sentiment_extractor.extraction.general.token import recover_base

# import the canonical LLM protocol used across the project
from color_sentiment_extractor.extraction.llm.types import (
    LLMClientProto as CoreLLMClientProto,
)

from .rgb_pipeline import process_color_phrase

# ── Types & Globals ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]


@runtime_checkable
class LLMClientProto(Protocol):
    """
    Structural contract for any LLM client we pass around.
    We only assume: given a phrase, it can maybe return an RGB tuple.
    """

    def query_rgb(self, phrase: str) -> RGB | None: ...

    # If rgb_pipeline.process_color_phrase actually calls a different method,
    # update this Protocol to match that method signature.


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Does: Lazily load spaCy 'en_core_web_sm' and cache it. Returns: Language."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:  # pragma: no cover
        raise ValueError(
            "spaCy model 'en_core_web_sm' not available. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from e


def _blocked(a: str, b: str) -> bool:
    """Does: Case-insensitive (a,b)/(b,a) BLOCKED_TOKENS check. Returns: bool."""
    al, bl = a.lower(), b.lower()
    return (al, bl) in BLOCKED_TOKENS or (bl, al) in BLOCKED_TOKENS


# ── Public API ────────────────────────────────────────────────────────────────
def extract_all_descriptive_color_phrases(
    text: str,
    known_tones: set[str],
    known_modifiers: set[str],
    all_webcolor_names: set[str],
    expression_map: dict,
    llm_client: LLMClientProto | None = None,
    nlp: Language | None = None,
    debug: bool = False,
) -> list[str]:
    """
    Does: Extract compound, standalone, and lone-tone color phrases from raw text.
    Returns: Sorted, lowercased, deduplicated list of extracted phrases.
    Used By: extract_phrases_from_segment, aggregate_color_phrase_results.
    """
    nlp = nlp or get_nlp()
    tokens = nlp(text)

    phrases: set[str] = set()
    raw_compounds: list[tuple[str, str]] = []  # accumulateur (modifier, tone)

    # Compounds
    extract_compound_phrases(
        tokens=tokens,
        compounds=phrases,
        raw_compounds=raw_compounds,
        known_color_tokens=known_tones,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        all_webcolor_names=all_webcolor_names,
        raw_text=text,
        debug=debug,
    )

    # Standalone + Lone tones
    phrases.update(
        extract_standalone_phrases(
            tokens,
            known_modifiers,
            known_tones,
            expression_map,
            llm_client,
            debug,
        )
    )
    phrases.update(extract_lone_tones(tokens, known_tones, debug))

    return sorted({p.lower() for p in phrases})


def extract_phrases_from_segment(
    segment: str,
    known_modifiers: set[str],
    known_tones: set[str],
    all_webcolor_names: set[str],
    expression_map: dict,
    llm_client: LLMClientProto | None = None,
    cache=None,
    nlp: Language | None = None,
    debug: bool = False,
) -> set[str]:
    """
    Does: Extract and validate phrases via base recovery, fuzzy threshold, and filters.
    Returns: Set of validated phrases (no cosmetic-noun endings, no redundant singles).
    Used By: aggregate_color_phrase_results.
    """
    raw_results = extract_all_descriptive_color_phrases(
        text=segment,
        known_tones=known_tones,
        known_modifiers=known_modifiers,
        all_webcolor_names=all_webcolor_names,
        expression_map=expression_map,
        llm_client=llm_client,
        nlp=nlp,
        debug=debug,
    )

    def is_high_confidence(raw: str, base: str) -> bool:
        raw_l, base_l = raw.lower(), base.lower()
        r = ratio(raw_l, base_l)
        blocked = _blocked(raw_l, base_l)
        if raw_l != base_l and (blocked or r < 80):
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[phrase_pipeline][BLOCKED FUZZY] raw=%r base=%r ratio=%s blocked=%s",
                    raw_l,
                    base_l,
                    r,
                    blocked,
                )
            return False
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[phrase_pipeline][CONF CHECK] raw=%r base=%r ratio=%s blocked=%s",
                raw_l,
                base_l,
                r,
                blocked,
            )
        return True

    valid_results: set[str] = set()

    for phrase in raw_results:
        tokens = phrase.split()
        if debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[phrase_pipeline][CHECK PHRASE] %r tokens=%s", phrase, tokens)

        if len(tokens) == 1:
            base = recover_base(
                tokens[0],
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                debug=False,
                fuzzy_fallback=False,
            )
            if (
                base in known_tones
                and base is not None
                and is_high_confidence(tokens[0], base)
                and not _blocked(tokens[0], base)
                and tokens[0] == base
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

            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[phrase_pipeline][PAIR] %r + %r → base1=%r base2=%r",
                    tokens[0],
                    tokens[1],
                    base1,
                    base2,
                )

            paired = (
                valid_roles
                and base1 is not None
                and base2 is not None
                and is_high_confidence(tokens[0], base1)
                and is_high_confidence(tokens[1], base2)
            )

            if paired:
                valid_results.add(phrase)

    # Filter 1: remove phrases ending with cosmetic nouns
    valid_results = {phrase for phrase in valid_results if phrase.split()[-1] not in COSMETIC_NOUNS}

    # Filter 2: remove single tokens already present in compounds
    compound_tokens = {t for p in valid_results for t in p.split() if len(p.split()) > 1}
    valid_results = {p for p in valid_results if len(p.split()) > 1 or p not in compound_tokens}

    if debug and logger.isEnabledFor(logging.DEBUG):
        logger.debug("[phrase_pipeline][FINAL VALID RESULTS] %s", valid_results)

    return valid_results


def process_segment_colors(
    color_phrases: list[str],
    known_modifiers: set[str],
    known_tones: set[str],
    llm_client: LLMClientProto | None = None,
    cache=None,
    debug: bool = False,
) -> tuple[list[str], list[RGB | None]]:
    """
    Does: Simplify phrases and resolve RGB for each extracted phrase.
    Returns: (simplified phrases list, RGB list aligned; None if unresolved).
    Used By: aggregate_color_phrase_results.
    """
    simplified: list[str] = []
    rgb_list: list[RGB | None] = []

    for phrase in color_phrases:
        try:
            # NOTE: we pass llm_client directly, no cast(object, ...)
            # If process_color_phrase expects the core LLM proto,
            # we just cast to that proto instead of 'object'.
            simple, rgb = process_color_phrase(
                phrase,
                known_modifiers,
                known_tones,
                llm_client=cast(CoreLLMClientProto | None, llm_client),
                cache=cache,
                debug=debug,
            )
        except Exception as e:  # pragma: no cover
            if debug:
                logger.exception("[phrase_pipeline][process_color_phrase ERROR] %r: %s", phrase, e)
            simple, rgb = "", None

        if not simple:
            if debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[phrase_pipeline][SKIP] No simplification for %r", phrase)
            simple = ""

        if rgb is False:
            rgb = None

        simplified.append(simple)
        rgb_list.append(rgb)

    return simplified, rgb_list


def aggregate_color_phrase_results(
    segments: list[str],
    known_modifiers: set[str],
    known_tones: set[str],
    all_webcolor_names: set[str],
    expression_map: dict,
    llm_client: LLMClientProto | None = None,
    cache=None,
    nlp: Language | None = None,
    debug: bool = False,
) -> tuple[set[str], list[str], dict[str, RGB]]:
    """
    Does: Aggregate simplified tones and RGB across segments with guardrails.
    Returns: (tone set, all simplified phrases, phrase→RGB dict).
    Used By: downstream color/sentiment aggregation.
    """

    def tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"[^\w\-]+", text.lower()) if t]

    def build_allowed_tokens(seg: str) -> set[str]:
        raw_tokens = tokenize(seg)
        allowed: set[str] = set(raw_tokens)
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

    def phrase_ok(phrase: str, allowed: set[str]) -> bool:
        words = phrase.lower().split()
        return all(w in allowed for w in words)

    tone_set: set[str] = set()
    all_phrases: list[str] = []
    rgb_map: dict[str, RGB] = {}

    nlp = nlp or get_nlp()

    for seg in segments:
        allowed = build_allowed_tokens(seg)

        extracted = extract_phrases_from_segment(
            segment=seg,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            all_webcolor_names=all_webcolor_names,
            expression_map=expression_map,
            llm_client=llm_client,
            cache=cache,
            nlp=nlp,
            debug=debug,
        )

        color_phrases = {p for p in extracted if phrase_ok(p, allowed)}

        simplified, rgb_list = process_segment_colors(
            color_phrases=sorted(color_phrases),
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            llm_client=llm_client,
            cache=cache,
            debug=debug,
        )

        for original, simple, rgb in zip(sorted(color_phrases), simplified, rgb_list, strict=False):
            # Guard: if LLM "simplified" a known tone into something else, keep original tone
            if simple != original and original in known_tones:
                simple = original

            # Guard: if simplifier changed a non-tone token, drop the RGB (unsafe hallucination)
            if simple != original and original not in known_tones:
                rgb = None

            if simple and phrase_ok(simple, allowed):
                all_phrases.append(simple)
                tone_set.add(simple)
                if rgb is not None:
                    rgb_map[simple] = rgb

    return tone_set, all_phrases, rgb_map
