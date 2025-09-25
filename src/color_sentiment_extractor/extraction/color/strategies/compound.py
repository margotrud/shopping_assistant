# src/color_sentiment_extractor/extraction/color/extraction/compound.py
from __future__ import annotations

"""
compound.py

Does:
    Resolve and validate (modifier, tone) pairs as compound color expressions.
    Strategies:
      - adjacent tokens
      - smart splits
      - glued token recovery (+ suffix/base normalization)
      - LLM-based simplification fallback
Returns:
    - Updates `compounds` (set[str] like "dusty rose")
    - Updates `raw_compounds` (list[tuple[str, str]] of (modifier, tone))
Notes:
    - No global spacy model load (lazy via get_nlp()).
    - Consistent typing & logging-friendly debug.
"""

from functools import lru_cache
from typing import Iterable, List, Set, Tuple, Optional, Dict

import logging
import spacy
from spacy.tokens import Token, Doc

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Lazy spaCy model ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy model once (lazy)."""
    return spacy.load("en_core_web_sm")

# ── Domain imports ───────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color import (
    SEMANTIC_CONFLICTS,
    COSMETIC_NOUNS,
)
from color_sentiment_extractor.extraction.color.recovery.llm_recovery import (
    _attempt_simplify_token,
)
from color_sentiment_extractor.extraction.color.recovery.modifier_resolution import (
    resolve_modifier_token,
    match_suffix_fallback,
    is_blocked_modifier_tone_pair,
    is_known_tone,
)
from color_sentiment_extractor.extraction.color.token.split import (
    split_tokens_to_parts,
    split_glued_tokens,
)
from color_sentiment_extractor.extraction.color.suffix.rules import build_y_variant

from color_sentiment_extractor.extraction.general.token.base_recovery import (
    recover_base,
)
from color_sentiment_extractor.extraction.general.token.normalize import (
    singularize,
    normalize_token,
)
from color_sentiment_extractor.extraction.general.token.suffix.recovery import (
    build_augmented_suffix_vocab,
)

# =============================================================================
# Helper: choisir la forme "surface" d'un modificateur pour affichage
# =============================================================================
def _surface_modifier(
    raw_mod: str,
    mod_canonical: Optional[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
) -> str:
    """
    Retourne la meilleure forme 'surface' pour le modificateur sans l'aplatir
    si la surface fournie est déjà cohérente. Priorités:
      1) garder la surface si elle est un mod valide
      2) si surface est suffixée (-y/-ish) et sa base == canonique → garder surface
      3) sinon, utiliser match_suffix_fallback si ça produit une forme suffixée valide
      4) sinon, variante -y de la base canonique si valide
      5) sinon, la base canonique si valide
      6) fallback: surface d'origine
    """
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


# =============================================================================
# 1) Compound builder
# =============================================================================
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
    Resolve (modifier, tone) into a validated compound, with LLM fallback and conflict checks.
    Mutates `compounds` (set of "mod tone") and `raw_compounds` ([(mod,tone), ...]) in-place.
    Rejects semantically invalid pairs and suffixy tones not in known lists.
    """
    if debug:
        log.debug("──────────── 🧪 attempt_mod_tone_pair ────────────")
        log.debug("[🔍 MODIFIER CANDIDATE] '%s'", mod_candidate)
        log.debug("[🔍 TONE CANDIDATE]     '%s'", tone_candidate)

    resolved: Dict[str, str] = {}
    for role, candidate, is_tone_role in [
        ("modifier", mod_candidate, False),
        ("tone", tone_candidate, True),
    ]:
        if debug:
            log.debug("[🔎 RESOLUTION START] Role=%s | Candidate='%s'", role, candidate)

        # Pre-block: conflict pair like {'warm','cool'} with existing known modifiers
        if role == "modifier" and any(
            frozenset([candidate, known]) in SEMANTIC_CONFLICTS for known in known_modifiers
        ):
            if debug:
                log.debug("[🧨 SEMANTIC BLOCK PRE-RESOLVE] '%s' conflicts with known modifier", candidate)
            return

        result = resolve_modifier_token(
            candidate,
            known_modifiers,
            known_tones=known_tones,
            fuzzy=False,      # ← was allow_fuzzy=False
            debug=debug,
        )
        if debug and result:
            log.debug("[✅ RESOLVED VIA TOKEN] '%s' → '%s'", candidate, result)

        # accept direct tone if known
        if not result and role == "tone" and (
            candidate in known_tones or candidate in all_webcolor_names
        ):
            result = candidate
            if debug:
                log.debug("[⚠️ DIRECT TONE ACCEPTED] '%s' is known", candidate)

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
                # Check semantic conflict after simplification
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

    if mod_candidate != mod and mod not in known_modifiers:
        if debug:
            log.debug("[⛔ MODIFIER REWRITE BLOCKED] '%s' → '%s' not trusted", mod_candidate, mod)
        return

    if (tone.endswith(("y", "ish"))) and (tone not in known_tones) and (tone not in all_webcolor_names):
        if debug:
            log.debug("[⛔ INVALID SUFFIXY TONE] '%s' not in known tone lists", tone)
        return

    compound = f"{mod} {tone}"
    if compound not in compounds:
        compounds.add(compound)
        raw_compounds.append((mod, tone))
        if debug:
            log.debug("[✅ COMPOUND DETECTED] → '%s'", compound)


# =============================================================================
# 2) Split helpers
# =============================================================================
def is_plausible_modifier(
    token: str, known_modifiers: Set[str], known_color_tokens: Set[str]
) -> bool:
    """
    Conservative plausibility: must be alpha, len>=3, and either known
    or recoverable to a known modifier without fuzz.
    """
    if token in known_modifiers or token in known_color_tokens:
        return True
    if not token.isalpha() or len(token) < 3:
        return False
    base = recover_base(token, known_modifiers, known_color_tokens, fuzzy_fallback=False)
    return bool(base and base in known_modifiers)


# =============================================================================
# 3) Compound Recovery from Split Tokens
# =============================================================================
def extract_from_split(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_color_tokens: Set[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool = True,
    *,
    aug_vocab: Optional[Set[str]] = None,
) -> None:
    """
    Recovers glued compound tokens like 'dustyroseglow' or 'sunsetcoral' by splitting them
    into valid 2- or 3-part phrases. Conserve la forme 'surface' du modificateur.
    """
    for token in tokens:
        text = token.text.lower()
        pos = getattr(token, "pos_", None)
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
                continue  # nothing meaningful to split

            # ------- Fallback binaire gauche|droite -------
            for i in range(2, len(text) - 1):
                left = text[:i]
                right = text[i:]

                # Right is known tone → left must become a valid modifier
                if right in known_tones:
                    left_mod = match_suffix_fallback(left, known_modifiers, known_tones) or left
                    left_base = recover_base(
                        left_mod, known_modifiers, known_tones, fuzzy_fallback=True
                    )

                    if debug:
                        log.debug("[🔍 FALLBACK CHECK] '%s' + '%s'", left, right)
                        log.debug("[🔁 NORMALIZED LEFT] '%s' → '%s'", left, left_mod)
                        log.debug("[🔁 RECOVERED BASE] '%s' → '%s'", left_mod, left_base)

                    if is_plausible_modifier(left_mod, known_modifiers, known_color_tokens):
                        surface_left = _surface_modifier(
                            left, left_base, known_modifiers, known_tones
                        )
                        phrase = f"{surface_left} {right}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append((surface_left, right))
                            if debug:
                                log.debug("[✅ FALLBACK SPLIT] '%s' → '%s'", text, phrase)
                        break

                # Left is known tone → right must become a valid modifier
                elif left in known_tones:
                    right_mod = match_suffix_fallback(right, known_modifiers, known_tones) or right
                    right_base = recover_base(
                        right_mod, known_modifiers, known_tones, fuzzy_fallback=True
                    )

                    if debug:
                        log.debug("[🔍 FALLBACK REVERSED] '%s' + '%s'", left, right)
                        log.debug("[🔁 NORMALIZED RIGHT] '%s' → '%s'", right, right_mod)
                        log.debug("[🔁 RECOVERED BASE] '%s' → '%s'", right_mod, right_base)

                    if (right_mod in known_modifiers) or (right_mod in known_color_tokens) or (
                        right_base in known_modifiers
                    ):
                        surface_right = _surface_modifier(
                            right, right_base, known_modifiers, known_tones
                        )
                        phrase = f"{left} {surface_right}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append((left, surface_right))
                            if debug:
                                log.debug("[✅ FALLBACK SPLIT REVERSED] '%s' → '%s'", text, phrase)
                        break

            continue

        # Normalisation légère (-y/-ish éventuels) avant validation
        parts = [match_suffix_fallback(p, known_modifiers, known_tones) or p for p in parts]

        # ------- 2-part -------
        if len(parts) == 2:
            first, second = parts
            first_is_tone = is_known_tone(first, known_tones, all_webcolor_names)
            first_canon_mod = resolve_modifier_token(
                first, known_modifiers, known_tones, debug=debug
            )
            second_is_tone = is_known_tone(second, known_tones, all_webcolor_names)
            second_canon_mod = resolve_modifier_token(
                second, known_modifiers, known_tones, debug=debug
            )

            if debug:
                log.debug(
                    "[🔍 2-PART] '%s'→tone=%s,mod=%s | '%s'→tone=%s,mod=%s",
                    first, first_is_tone, bool(first_canon_mod),
                    second, second_is_tone, bool(second_canon_mod),
                )

            phrase = None
            if (first_canon_mod or first_is_tone) and second_is_tone:
                if first_canon_mod:
                    surface_first = _surface_modifier(
                        first, first_canon_mod, known_modifiers, known_tones
                    )
                    phrase = f"{surface_first} {second}"
                    mod_tok = surface_first
                else:
                    phrase = f"{first} {second}"
                    mod_tok = first
                if phrase not in compounds:
                    compounds.add(phrase)
                    raw_compounds.append((mod_tok, second))
                    if debug:
                        log.debug("[✅ 2-PART COMPOUND] '%s' → '%s'", text, phrase)
            else:
                if debug:
                    log.debug("[❌ INVALID 2-PART] '%s %s'", first, second)

        # ------- 3-part -------
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
                    first, second, third,
                    bool(first_canon_mod), bool(second_canon_mod), bool(third_canon_mod),
                    first_is_tone, second_is_tone, third_is_tone,
                )

            def surf(token_: str, canon_: Optional[str]) -> str:
                return (
                    _surface_modifier(token_, canon_, known_modifiers, known_tones)
                    if canon_ else token_
                )

            valid_1 = first_canon_mod and second_is_tone and third_canon_mod
            valid_2 = first_canon_mod and second_canon_mod and third_is_tone
            valid_3 = first_canon_mod and second_is_tone and third_is_tone

            phrase = None
            mod_tok = None
            if valid_1:
                phrase = f"{surf(first, first_canon_mod)} {second} {surf(third, third_canon_mod)}"
                mod_tok = surf(first, first_canon_mod)
            elif valid_2:
                phrase = f"{surf(first, first_canon_mod)} {surf(second, second_canon_mod)} {third}"
                mod_tok = surf(first, first_canon_mod)
            elif valid_3:
                phrase = f"{surf(first, first_canon_mod)} {second} {third}"
                mod_tok = surf(first, first_canon_mod)

            if phrase:
                if phrase not in compounds and mod_tok:
                    compounds.add(phrase)
                    # on mappe (mod, tone) en choisissant le dernier segment tone:
                    tone_tok = third if third_is_tone else second if second_is_tone else third
                    raw_compounds.append((mod_tok, tone_tok))
                    if debug:
                        log.debug("[✅ 3-PART COMPOUND] '%s' → '%s'", text, phrase)
            else:
                if debug:
                    log.debug("[❌ INVALID 3-PART] '%s %s %s'", first, second, third)
                # merged tone fallback
                if first_canon_mod:
                    merged_tone = f"{second} {third}"
                    if merged_tone in known_tones:
                        a_out = surf(first, first_canon_mod)
                        phrase = f"{a_out} {merged_tone}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append((a_out, merged_tone))
                            if debug:
                                log.debug("[✅ MERGED TONE 3-PART] '%s' → '%s'", text, phrase)


# =============================================================================
# 4) Compound Recovery from Glued Tokens
# =============================================================================
def extract_from_glued(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_color_tokens: Set[str],
    known_modifiers: Set[str],
    known_tones: Set[str],
    all_webcolor_names: Set[str],
    debug: bool = True,
    *,
    aug_vocab: Optional[Set[str]] = None,
) -> None:
    """
    Extract compound color phrases from glued tokens like 'dustyrose' or 'greylavenderpink'.
    Conserve la forme 'surface' des modificateurs détectés.
    """
    for token in tokens:
        raw = token.text.lower()

        pos = getattr(token, "pos_", None)
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
            known_modifiers,
            debug=debug,
            vocab=aug_vocab,
        )

        # 3-part fallback via split_tokens_to_parts
        if not parts or len(parts) not in {2, 3} or (len(parts) == 1 and parts[0] in known_tones):
            fallback_parts = split_tokens_to_parts(raw, known_modifiers | known_tones, debug=debug)
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
                    surface_a = _surface_modifier(a, mod_a, known_modifiers, known_tones)
                    phrase = f"{surface_a} {b} {c}"
                    if phrase not in compounds:
                        compounds.add(phrase)
                        # map to (modifier, last tone candidate)
                        tone_tok = c if tone_c else b
                        raw_compounds.append((surface_a, tone_tok))
                        if debug:
                            log.debug("[✅ GLUED 3-PART FALLBACK] '%s' → '%s'", raw, phrase)
            continue

        # --- 2-part ---
        if len(parts) == 2:
            a, b = parts
            mod_canon = resolve_modifier_token(a, known_modifiers, known_tones, debug=debug)
            is_tone_b = is_known_tone(b, known_tones, all_webcolor_names)

            allow_tone_pair = False  # keep strict (avoid "red purple")
            mod_fallback = a in known_modifiers and is_tone_b
            mod_mod_pair = a in known_modifiers and (b in known_modifiers)

            if mod_canon and is_tone_b:
                surface_mod = _surface_modifier(a, mod_canon, known_modifiers, known_tones)
                compound = f"{surface_mod} {b}"
                mod_tok = surface_mod
            elif mod_fallback or mod_mod_pair or (allow_tone_pair and a in known_tones and is_tone_b):
                compound = f"{a} {b}"
                mod_tok = a
            else:
                continue

            if compound not in compounds:
                compounds.add(compound)
                raw_compounds.append((mod_tok, b))
                if debug:
                    log.debug("[✅ GLUED 2-PART COMPOUND] '%s' → '%s'", raw, compound)

        # --- 3-part ---
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
                    a, b, c, bool(mod_a), bool(mod_b), bool(mod_c), tone_a, tone_b, tone_c
                )

            def surf(token_: str, canon_: Optional[str]) -> str:
                return _surface_modifier(token_, canon_, known_modifiers, known_tones) if canon_ else token_

            valid = (
                (mod_a and tone_b and tone_c)
                or (mod_a and mod_b and tone_c)
                or (mod_a and tone_b and mod_c)
            )
            if valid:
                a_out = surf(a, mod_a)
                b_out = surf(b, mod_b) if mod_b else b
                c_out = surf(c, mod_c) if mod_c else c
                compound = f"{a_out} {b_out} {c_out}"
                if compound not in compounds:
                    compounds.add(compound)
                    # choose a tone token preference (last tone if available)
                    tone_tok = c if tone_c else b if tone_b else c
                    raw_compounds.append((a_out, tone_tok))
                    if debug:
                        log.debug("[✅ GLUED 3-PART COMPOUND] '%s' → '%s'", raw, compound)


# =============================================================================
# 5) Compound Extraction from Adjacent Tokens
# =============================================================================
def extract_from_adjacent(
    tokens: Iterable[Token],
    compounds: Set[str],
    raw_compounds: List[Tuple[str, str]],
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = True,
) -> None:
    """
    Extract compounds from adjacent tokens:
      - single modifier + tone (e.g. "soft pink")
      - two-word modifier + tone (e.g. "barely-there pink")
    Mutates compounds/raw_compounds in-place.
    """
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

        mod_canon = resolve_modifier_token(
            raw_mod, known_modifiers, known_tones, debug=debug
        )
        tone = raw_tone if raw_tone in known_tones else None

        if mod_canon and tone:
            surface_mod = _surface_modifier(raw_mod, mod_canon, known_modifiers, known_tones)
            phrase = f"{surface_mod} {tone}"
            if phrase not in compounds:
                compounds.add(phrase)
                raw_compounds.append((surface_mod, tone))
                if debug:
                    log.debug("[✅ ADJACENT COMPOUND] → '%s'", phrase)

        # ----- Extended 2-word modifier + tone -----
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

            mod2_canon = resolve_modifier_token(
                combined_mod_raw, known_modifiers, known_tones, debug=debug
            )
            tone2 = raw_tone2 if raw_tone2 in known_tones else None

            if mod2_canon and tone2:
                surface_mod2 = _surface_modifier(combined_mod_raw, mod2_canon, known_modifiers, known_tones)
                phrase2 = f"{surface_mod2} {tone2}"
                if phrase2 not in compounds:
                    compounds.add(phrase2)
                    raw_compounds.append((surface_mod2, tone2))
                    if debug:
                        log.debug("[✅ ADJACENT 2-WORD COMPOUND] → '%s'", phrase2)


# =============================================================================
# 6) Compound Extraction Orchestrator
# =============================================================================
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
    Orchestrate adjacent/split/glued extractors with fallback recovery and final filtering.
    Mutates `compounds` and `raw_compounds` (list of (modifier, tone)).
    """
    if raw_text:
        # normaliser les tirets pour faciliter les splits
        raw_text = raw_text.replace("-", " ")
        tokens = get_nlp()(raw_text)

    # 🔑 Build once per segment: vocab étendu (perf gain sur glued splits)
    aug_vocab = build_augmented_suffix_vocab(known_color_tokens, known_modifiers)
    if debug:
        try:
            log.debug("[DBG] aug_vocab size=%d", len(aug_vocab))
        except Exception:
            pass

    # Main extractors
    extract_from_adjacent(tokens, compounds, raw_compounds, known_modifiers, known_tones, debug)
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

    # Fallback: fuzzy match missed modifier + tone pairs
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
            fuzzy=True,   # ← was allow_fuzzy=True
            debug=debug,
        )

        if mod_canon:
            surface_mod = _surface_modifier(left, mod_canon, known_modifiers, known_tones)
            phrase = f"{surface_mod} {right}"
            if phrase not in compounds:
                compounds.add(phrase)
                raw_compounds.append((surface_mod, right))
                if debug:
                    log.debug("[🩹 FALLBACK PATCH] %s+%s → %s", left, right, phrase)

    # Remove blocked (modifier, tone) pairs
    blocked = {(m, t) for (m, t) in raw_compounds if is_blocked_modifier_tone_pair(m, t)}
    if blocked:
        for mod, tone in blocked:
            compounds.discard(f"{mod} {tone}")
        # remove from raw_compounds too
        raw_compounds[:] = [(m, t) for (m, t) in raw_compounds if (m, t) not in blocked]
        if debug:
            for mod, tone in blocked:
                log.debug("[⛔ BLOCKED PAIR REMOVED] '%s %s'", mod, tone)
