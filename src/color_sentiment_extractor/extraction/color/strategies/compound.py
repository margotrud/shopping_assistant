from __future__ import annotations

"""
compound.py

Does: Resolve and validate (modifier, tone) pairs from adjacent/split/glued tokens with suffix/base recovery and optional LLM fallback.
Returns: Mutates `compounds: set[str]` (e.g., "dusty rose") and `raw_compounds: list[tuple[str,str]]` holding (modifier, tone).
Used by: Color phrase extraction pipelines; RGB pipelines downstream.
"""

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple
import logging
import spacy
from spacy.tokens import Doc, Token

__all__ = [
    "attempt_mod_tone_pair",
    "extract_from_adjacent",
    "extract_from_split",
    "extract_from_glued",
    "extract_compound_phrases",
]

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy model once (lazy); fallback to blank English if model missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")  # tokenization-only fallback


from color_sentiment_extractor.extraction.color.constants import (
    COSMETIC_NOUNS,
    SEMANTIC_CONFLICTS,
)
from color_sentiment_extractor.extraction.color.recovery import (
    _attempt_simplify_token,
    is_blocked_modifier_tone_pair,
    is_known_tone,
    match_suffix_fallback,
    resolve_modifier_token,
)
from color_sentiment_extractor.extraction.color.suffix import build_y_variant
from color_sentiment_extractor.extraction.color.token import (
    split_glued_tokens,
    split_tokens_to_parts,
)
from color_sentiment_extractor.extraction.general.token import (
    normalize_token,
    recover_base,
    singularize,
)
from color_sentiment_extractor.extraction.general.token.suffix import (
    build_augmented_suffix_vocab,
)


def _surface_modifier(
    raw_mod: str,
    mod_canonical: Optional[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
) -> str:
    """Pick best display form for modifier."""
    raw = raw_mod.lower()

    if raw in known_modifiers:
        return raw

    if raw.endswith(("y", "ish")):
        base_raw = recover_base(
            raw,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            fuzzy_fallback=False,
        ) or None
        if base_raw and mod_canonical and base_raw == mod_canonical:
            return raw

    sug = match_suffix_fallback(raw, known_modifiers, known_tones)
    if sug and sug.endswith(("y", "ish")) and sug in known_modifiers:
        return sug

    if mod_canonical:
        y = build_y_variant(mod_canonical)
        if y and y in known_modifiers:
            return y
        if mod_canonical in known_modifiers:
            return mod_canonical

    return raw


def _is_invalid_suffixy_tone(
    tone: str, known_tones: Set[str], all_webcolor_names: Set[str]
) -> bool:
    """Reject tones like 'pinkish' if not explicitly known."""
    return (
        tone.endswith(("y", "ish"))
        and (tone not in known_tones)
        and (tone not in all_webcolor_names)
    )


def _safe_add_compound(
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    mod: str,
    tone: str,
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool,
) -> None:
    """Add phrase 'mod tone' if valid."""
    if _is_invalid_suffixy_tone(tone, known_tones, all_webcolor_names):
        if debug:
            log.debug("[⛔ INVALID SUFFIXY TONE] '%s' not in known tone lists", tone)
        return
    phrase = f"{mod} {tone}"
    if phrase not in compounds:
        compounds.add(phrase)
        raw_compounds.append((mod, tone))
        if debug:
            log.debug("[✅ COMPOUND DETECTED] → '%s'", phrase)


def is_plausible_modifier(
    token: str, known_modifiers: Set[str], known_color_tokens: Set[str]
) -> bool:
    """
    Heuristic: looks like a modifier (or can be reduced to one) and not total garbage.
    """
    if token in known_modifiers or token in known_color_tokens:
        return True
    if not token.isalpha() or len(token) < 3:
        return False
    base = recover_base(
        token,
        known_modifiers=known_modifiers,
        known_tones=known_color_tokens,
        fuzzy_fallback=False,
    )
    return bool(base and base in known_modifiers)


def attempt_mod_tone_pair(
    mod_candidate: str,
    tone_candidate: str,
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    llm_client,
    debug: bool = False,
) -> None:
    """
    Try to resolve modifier+tone with recovery / fallback / LLM.
    Writes to compounds/raw_compounds if confident.
    """
    if debug:
        log.debug("──────────── 🧪 attempt_mod_tone_pair ────────────")
        log.debug("[🔍 MODIFIER CANDIDATE] '%s'", mod_candidate)
        log.debug("[🔍 TONE CANDIDATE]     '%s'", tone_candidate)

    resolved: Dict[str, str] = {}
    for role, candidate in [("modifier", mod_candidate), ("tone", tone_candidate)]:
        if debug:
            log.debug("[🔎 RESOLUTION START] Role=%s | Candidate='%s'", role, candidate)

        # direct resolver, no fuzz
        result = resolve_modifier_token(
            candidate,
            known_modifiers,
            known_tones=known_tones,
            fuzzy=False,
            debug=debug,
        )
        if debug and result:
            log.debug("[✅ RESOLVED VIA TOKEN] '%s' → '%s'", candidate, result)

        # accept direct tone as-is if known
        if not result and role == "tone" and (
            candidate in known_tones or candidate in all_webcolor_names
        ):
            result = candidate
            if debug:
                log.debug("[⚠️ DIRECT TONE ACCEPTED] '%s' is known", candidate)

        # LLM fallback
        if not result:
            if debug:
                log.debug("[💡 LLM FALLBACK] simplify '%s' for role '%s'", candidate, role)
            result = _attempt_simplify_token(
                candidate,
                known_modifiers,
                known_tones,
                llm_client,
                role=role,
                debug=debug,
            )
            if result:
                if debug:
                    log.debug("[✨ SIMPLIFIED] '%s' → '%s' via LLM", candidate, result)
                if (
                    role == "modifier"
                    and result != candidate
                    and candidate not in known_modifiers
                    and result not in known_modifiers
                    and frozenset([candidate, result]) in SEMANTIC_CONFLICTS
                ):
                    if debug:
                        log.debug("[🧨 FUZZY CONFLICT BLOCKED] '%s' → '%s'", candidate, result)
                    return
            else:
                if debug:
                    log.debug("[⛔ LLM REJECTED] no simplification for '%s'", candidate)
                return

        resolved[role] = result
        if debug:
            log.debug("[✅ FINAL %s] '%s' → '%s'", role.upper(), candidate, result)

    mod, tone = resolved["modifier"], resolved["tone"]

    if debug:
        log.debug("[🎯 FINAL RESOLUTION] mod='%s' tone='%s'", mod, tone)

    # don't allow rewriting modifier to something we still don't trust
    if mod_candidate != mod and mod not in known_modifiers:
        if debug:
            log.debug("[⛔ MODIFIER REWRITE BLOCKED] '%s' → '%s' not trusted", mod_candidate, mod)
        return

    _safe_add_compound(compounds, raw_compounds, mod, tone, known_tones, all_webcolor_names, debug)


def extract_from_split(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_color_tokens: Set[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool = False,
    *,
    aug_vocab: Optional[Set[str]] = None,
) -> None:
    """
    Recover modifier+tone from tokens that might be internally splittable
    ('dustyrose' -> ['dusty', 'rose'], etc.).
    """
    for token in tokens:
        text = token.text.lower()
        pos = getattr(token, "pos_", "") or ""
        if pos not in {"ADJ", "NOUN", "PROPN"}:
            if debug:
                log.debug("[⏩ POS SKIP] '%s' pos=%s", text, pos)
            continue

        if debug:
            log.debug("[🔍 TOKEN] '%s'", text)

        if text in known_modifiers or text in known_tones:
            if debug:
                log.debug("[⏩ SKIP] '%s' is a known tone/modifier", text)
            continue
        if any(text == c.replace(" ", "") for c in compounds):
            if debug:
                log.debug("[⏩ SKIP] '%s' matches existing compound (glued form)", text)
            continue

        if debug:
            log.debug("[🔍 ATTEMPT SPLIT] '%s'", text)

        parts = split_tokens_to_parts(text, known_modifiers | known_tones, debug=debug)

        if not parts or len(parts) not in {2, 3} or (len(parts) == 1 and parts[0] in known_tones):
            if debug:
                log.debug("[⛔ SPLIT FAILED] → %s", parts)
                log.debug("[🔁 FALLBACK] binary left|right splits for '%s'", text)

            if len(text) < 4:
                continue

            for i in range(2, len(text) - 1):
                left = text[:i]
                right = text[i:]

                # Right is tone -> left must behave like modifier
                if right in known_tones:
                    left_mod = match_suffix_fallback(left, known_modifiers, known_tones) or left
                    left_base = recover_base(
                        left_mod,
                        known_modifiers=known_modifiers,
                        known_tones=known_tones,
                        fuzzy_fallback=True,
                    )

                    if debug:
                        log.debug("[🔍 FALLBACK CHECK] '%s' + '%s'", left, right)
                        log.debug("[🔁 NORMALIZED LEFT] '%s' → '%s'", left, left_mod)
                        log.debug("[🔁 RECOVERED BASE] '%s' → '%s'", left_mod, left_base)

                    if is_plausible_modifier(left_mod, known_modifiers, known_color_tokens):
                        surface_left = _surface_modifier(
                            left, left_base, known_modifiers, known_tones
                        )
                        _safe_add_compound(
                            compounds,
                            raw_compounds,
                            surface_left,
                            right,
                            known_tones,
                            all_webcolor_names,
                            debug,
                        )
                        break

                # Left is tone -> right must act like modifier (reverse order)
                elif left in known_tones:
                    right_mod = match_suffix_fallback(right, known_modifiers, known_tones) or right
                    right_base = recover_base(
                        right_mod,
                        known_modifiers=known_modifiers,
                        known_tones=known_tones,
                        fuzzy_fallback=True,
                    )

                    if debug:
                        log.debug("[🔍 FALLBACK REVERSED] '%s' + '%s'", left, right)
                        log.debug("[🔁 NORMALIZED RIGHT] '%s' → '%s'", right, right_mod)
                        log.debug("[🔁 RECOVERED BASE] '%s' → '%s'", right_mod, right_base)

                    if (
                        (right_mod in known_modifiers)
                        or (right_mod in known_color_tokens)
                        or (right_base in known_modifiers)
                    ):
                        surface_right = _surface_modifier(
                            right, right_base, known_modifiers, known_tones
                        )
                        _safe_add_compound(
                            compounds,
                            raw_compounds,
                            surface_right,
                            left,
                            known_tones,
                            all_webcolor_names,
                            debug,
                        )
                        break

            continue

        # Light-normalize split parts
        parts = [match_suffix_fallback(p, known_modifiers, known_tones) or p for p in parts]

        # 2-part case
        if len(parts) == 2:
            first, second = parts
            first_is_tone = is_known_tone(first, known_tones, all_webcolor_names)
            first_canon_mod = resolve_modifier_token(
                first, known_modifiers, known_tones, debug=debug
            )
            second_is_tone = is_known_tone(second, known_tones, all_webcolor_names)

            if debug:
                log.debug(
                    "[🔍 2-PART] '%s'→tone=%s,mod=%s | '%s'→tone=%s",
                    first,
                    first_is_tone,
                    bool(first_canon_mod),
                    second,
                    second_is_tone,
                )

            if (first_canon_mod or first_is_tone) and second_is_tone:
                if first_canon_mod:
                    surface_first = _surface_modifier(
                        first, first_canon_mod, known_modifiers, known_tones
                    )
                    _safe_add_compound(
                        compounds,
                        raw_compounds,
                        surface_first,
                        second,
                        known_tones,
                        all_webcolor_names,
                        debug,
                    )
                else:
                    if not first_is_tone:  # avoid "red purple"
                        _safe_add_compound(
                            compounds,
                            raw_compounds,
                            first,
                            second,
                            known_tones,
                            all_webcolor_names,
                            debug,
                        )
            else:
                if debug:
                    log.debug("[❌ INVALID 2-PART] '%s %s'", first, second)

        # 3-part case
        elif len(parts) == 3:
            first, second, third = parts

            first_canon_mod = resolve_modifier_token(
                first, known_modifiers, known_tones, debug=debug
            )
            second_canon_mod = resolve_modifier_token(
                second, known_modifiers, known_tones, debug=debug
            )
            third_canon_mod = resolve_modifier_token(
                third, known_modifiers, known_tones, debug=debug
            )

            first_is_tone = is_known_tone(first, known_tones, all_webcolor_names)
            second_is_tone = is_known_tone(second, known_tones, all_webcolor_names)
            third_is_tone = is_known_tone(third, known_tones, all_webcolor_names)

            if debug:
                log.debug(
                    "[🔍 3-PART] '%s %s %s' | mod=[%s,%s,%s] tone=[%s,%s,%s]",
                    first,
                    second,
                    third,
                    bool(first_canon_mod),
                    bool(second_canon_mod),
                    bool(third_canon_mod),
                    first_is_tone,
                    second_is_tone,
                    third_is_tone,
                )

            def surf(token_: str, canon_: Optional[str]) -> str:
                return (
                    _surface_modifier(token_, canon_, known_modifiers, known_tones)
                    if canon_
                    else token_
                )

            valid_1 = first_canon_mod and second_is_tone and third_canon_mod
            valid_2 = first_canon_mod and second_canon_mod and third_is_tone
            valid_3 = first_canon_mod and second_is_tone and third_is_tone

            if valid_1:
                a_out = surf(first, first_canon_mod)
                tone_tok = third if third_is_tone else second
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    a_out,
                    tone_tok,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )
            elif valid_2:
                a_out = surf(first, first_canon_mod)
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    a_out,
                    third,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )
            elif valid_3:
                a_out = surf(first, first_canon_mod)
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    a_out,
                    third,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )
            else:
                if debug:
                    log.debug("[❌ INVALID 3-PART] '%s %s %s'", first, second, third)
                # merged tone fallback
                if first_canon_mod:
                    merged_tone = f"{second} {third}"
                    if merged_tone in known_tones:
                        a_out = surf(first, first_canon_mod)
                        _safe_add_compound(
                            compounds,
                            raw_compounds,
                            a_out,
                            merged_tone,
                            known_tones,
                            all_webcolor_names,
                            debug,
                        )


def extract_from_glued(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_color_tokens: Set[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool = False,
    *,
    aug_vocab: Optional[Set[str]] = None,
) -> None:
    """
    Recover from glued tokens like 'dustyrose'.
    """
    for token in tokens:
        raw = token.text.lower()

        pos = getattr(token, "pos_", "") or ""
        if pos not in {"ADJ", "NOUN", "PROPN"}:
            if debug:
                log.debug("[⏩ POS SKIP] '%s' pos=%s", raw, pos)
            continue

        if not raw.isalpha() or raw in known_color_tokens:
            if debug:
                log.debug("[⛔ SKIP] token '%s' is known or non-alpha", raw)
            continue

        parts = split_glued_tokens(
            raw,
            known_color_tokens,
            frozenset(known_modifiers),
            debug=debug,
            vocab=aug_vocab,
        )

        # try backup split if the main didn't convince us
        if not parts or len(parts) not in {2, 3} or (len(parts) == 1 and parts[0] in known_tones):
            fallback_parts = split_tokens_to_parts(
                raw, known_modifiers | known_tones, debug=debug
            )
            if fallback_parts and len(fallback_parts) == 3:
                a, b, c = fallback_parts
                tone_b = is_known_tone(b, known_tones, all_webcolor_names)
                tone_c = is_known_tone(c, known_tones, all_webcolor_names)

                def normalize_mod(token_: str) -> Optional[str]:
                    if token_ in known_modifiers:
                        return token_
                    if token_.endswith("y") and token_[:-1] in known_modifiers:
                        return token_[:-1]
                    if token_.endswith("ish") and token_[:-3] in known_modifiers:
                        return token_[:-3]
                    return None

                mod_a = normalize_mod(a)
                if mod_a and tone_b and tone_c:
                    surface_a = _surface_modifier(
                        a, mod_a, known_modifiers, known_tones
                    )
                    _safe_add_compound(
                        compounds,
                        raw_compounds,
                        surface_a,
                        c,
                        known_tones,
                        all_webcolor_names,
                        debug,
                    )
            continue

        if len(parts) == 2:
            a, b = parts
            mod_canon = resolve_modifier_token(a, known_modifiers, known_tones, debug=debug)
            is_tone_b = is_known_tone(b, known_tones, all_webcolor_names)

            # we only take it if b is a tone (avoid mod+mod)
            if mod_canon and is_tone_b:
                surface_mod = _surface_modifier(a, mod_canon, known_modifiers, known_tones)
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    surface_mod,
                    b,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )

        elif len(parts) == 3:
            a, b, c = parts

            mod_a = resolve_modifier_token(a, known_modifiers, known_tones, debug=debug)
            mod_b = resolve_modifier_token(b, known_modifiers, known_tones, debug=debug)
            mod_c = resolve_modifier_token(c, known_modifiers, known_tones, debug=debug)

            tone_a = is_known_tone(a, known_tones, all_webcolor_names)
            tone_b = is_known_tone(b, known_tones, all_webcolor_names)
            tone_c = is_known_tone(c, known_tones, all_webcolor_names)

            if debug:
                log.debug(
                    "[🔍 GLUED 3-PART] '%s %s %s' | mod=[%s,%s,%s] tone=[%s,%s,%s]",
                    a, b, c, bool(mod_a), bool(mod_b), bool(mod_c),
                    tone_a, tone_b, tone_c,
                )

            def surf(token_: str, canon_: Optional[str]) -> str:
                return (
                    _surface_modifier(token_, canon_, known_modifiers, known_tones)
                    if canon_
                    else token_
                )

            valid = (
                (mod_a and tone_b and tone_c)
                or (mod_a and mod_b and tone_c)
                or (mod_a and tone_b and mod_c)
            )
            if valid:
                a_out = surf(a, mod_a)
                tone_tok = c if tone_c else (b if tone_b else c)
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    a_out,
                    tone_tok,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )


def extract_from_adjacent(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool = False,
) -> None:
    """Scan adjacent tokens: 'soft pink', 'barely-there pink', etc."""
    tokens_list = list(tokens)
    n = len(tokens_list)

    for i in range(n - 1):
        raw_mod = tokens_list[i].text.lower()
        raw_tone = singularize(tokens_list[i + 1].text.lower())

        if debug:
            log.debug("[🔍 ADJACENT PAIR] '%s' + '%s'", raw_mod, raw_tone)

        if raw_mod in COSMETIC_NOUNS or raw_tone in COSMETIC_NOUNS:
            if debug:
                log.debug("[⛔ COSMETIC BLOCK] skip '%s %s'", raw_mod, raw_tone)
            continue

        mod_canon = resolve_modifier_token(raw_mod, known_modifiers, known_tones, debug=debug)
        tone = raw_tone if raw_tone in known_tones else None

        if mod_canon and tone:
            surface_mod = _surface_modifier(raw_mod, mod_canon, known_modifiers, known_tones)
            _safe_add_compound(
                compounds,
                raw_compounds,
                surface_mod,
                tone,
                known_tones,
                all_webcolor_names,
                debug,
            )

        # 2-word modifier + tone
        if i < n - 2:
            m1 = tokens_list[i].text.lower()
            m2 = tokens_list[i + 1].text.lower()
            combined_mod_raw = f"{m1}-{m2}"
            raw_tone2 = singularize(tokens_list[i + 2].text.lower())

            if debug:
                log.debug("[🔍 2-WORD MODIFIER] '%s' + '%s'", combined_mod_raw, raw_tone2)

            if m1 in COSMETIC_NOUNS or m2 in COSMETIC_NOUNS or raw_tone2 in COSMETIC_NOUNS:
                if debug:
                    log.debug("[⛔ COSMETIC BLOCK] skip '%s %s'", combined_mod_raw, raw_tone2)
                continue

            mod2_canon = resolve_modifier_token(combined_mod_raw, known_modifiers, known_tones, debug=debug)
            tone2 = raw_tone2 if raw_tone2 in known_tones else None

            if mod2_canon and tone2:
                surface_mod2 = _surface_modifier(
                    combined_mod_raw, mod2_canon, known_modifiers, known_tones
                )
                _safe_add_compound(
                    compounds,
                    raw_compounds,
                    surface_mod2,
                    tone2,
                    known_tones,
                    all_webcolor_names,
                    debug,
                )


def extract_compound_phrases(
    tokens: Iterable[Token] | Doc,
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_color_tokens: Set[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    raw_text: str = "",
    debug: bool = False,
) -> None:
    """
    Orchestrator: run adjacent/split/glued passes and then final cleanup.
    """
    if raw_text:
        raw_text = raw_text.replace("-", " ")
        tokens = get_nlp()(raw_text)
    tokens = list(tokens)

    # Build shared augmented vocab once for this batch (helps glued splitting)
    aug_vocab = build_augmented_suffix_vocab(known_color_tokens, known_modifiers)
    if debug:
        try:
            log.debug("[DBG] aug_vocab size=%d", len(aug_vocab))
        except Exception:
            pass

    extract_from_adjacent(
        tokens,
        compounds,
        raw_compounds,
        known_modifiers,
        known_tones,
        all_webcolor_names,
        debug,
    )
    extract_from_split(
        tokens,
        compounds,
        raw_compounds,
        known_color_tokens,
        known_modifiers,
        known_tones,
        all_webcolor_names,
        debug=debug,
        aug_vocab=aug_vocab,
    )
    extract_from_glued(
        tokens,
        compounds,
        raw_compounds,
        known_color_tokens,
        known_modifiers,
        known_tones,
        all_webcolor_names,
        debug=debug,
        aug_vocab=aug_vocab,
    )

    # fuzzy-ish fallback on adjacent tokens where left isn't in known_modifiers yet
    tokens_list = list(tokens)
    for i in range(len(tokens_list) - 1):
        left = normalize_token(tokens_list[i].text, keep_hyphens=True)
        right = normalize_token(tokens_list[i + 1].text, keep_hyphens=True)

        if left in known_modifiers or left in known_tones or len(left) < 3:
            continue
        if right not in known_tones or not right.isalpha():
            continue

        mod_canon = resolve_modifier_token(
            left,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            fuzzy=True,
            debug=debug,
        )

        if mod_canon:
            surface_mod = _surface_modifier(left, mod_canon, known_modifiers, known_tones)
            _safe_add_compound(
                compounds,
                raw_compounds,
                surface_mod,
                right,
                known_tones,
                all_webcolor_names,
                debug,
            )

    # final: remove semantically blocked pairs
    blocked = {(m, t) for (m, t) in raw_compounds if is_blocked_modifier_tone_pair(m, t)}
    if blocked:
        for mod, tone in blocked:
            compounds.discard(f"{mod} {tone}")
        raw_compounds[:] = [(m, t) for (m, t) in raw_compounds if (m, t) not in blocked]
        if debug:
            for mod, tone in blocked:
                log.debug("[⛔ BLOCKED PAIR REMOVED] '%s %s'", mod, tone)
