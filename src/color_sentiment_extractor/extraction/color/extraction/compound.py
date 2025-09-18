"""
compound.py

Does:
    Attempts to resolve and validate (modifier, tone) pairs as compound color expressions.
    Includes:
        - Core pair resolver with normalization and suppression logic
        - LLM-based fallback simplifier for noisy inputs

Returns:
    - Valid compound phrases (e.g. 'soft pink') added to provided sets/lists
    - Rejected or invalid pairs are skipped silently
"""
from __future__ import annotations

import spacy

from color_sentiment_extractor.extraction.color.constants import SEMANTIC_CONFLICTS, COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.recovery.llm_recovery import _attempt_simplify_token
from color_sentiment_extractor.extraction.color.recovery.modifier_resolution import (
    resolve_modifier_token,
    match_suffix_fallback,
    is_blocked_modifier_tone_pair,
    is_known_tone,
)
from color_sentiment_extractor.extraction.color.token.split import split_tokens_to_parts, split_glued_tokens
from color_sentiment_extractor.extraction.color.suffix.rules import build_y_variant

from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import singularize, normalize_token
from color_sentiment_extractor.extraction.general.token.suffix.recovery import build_augmented_suffix_vocab

nlp = spacy.load("en_core_web_sm")


# =====================================================================
# Helper: choisir la forme "surface" d'un modificateur pour affichage
# =====================================================================
def _surface_modifier(
    raw_mod: str,
    mod_canonical: str | None,
    known_modifiers: set,
    known_tones: set,
) -> str:
    """
    Retourne la meilleure forme 'surface' pour le modificateur sans l'aplatir
    en base si l'utilisateur a déjà tapé une forme suffixée ('dusty', 'pinkish', ...).
    Priorités:
      1) garder la surface si elle est déjà un mod valide
      2) si la surface est suffixée (-y/-ish) et que sa base == canonique → garder la surface
      3) sinon, ne prendre match_suffix_fallback(raw) QUE si ça produit une forme suffixée
      4) sinon, tenter la variante -y de la base canonique
      5) sinon, si le canonique est un mod valide, le retourner
      6) fallback: la surface telle quelle
    """
    raw = raw_mod.lower()

    # 1) surface brute déjà connue
    if raw in known_modifiers:
        return raw

    # 2) forme suffixée cohérente avec la base canonique → on la garde
    if raw.endswith("y") or raw.endswith("ish"):
        try:
            base_raw = recover_base(
                raw,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                fuzzy_fallback=False,
            )
        except Exception:
            base_raw = None
        if base_raw and mod_canonical and base_raw == mod_canonical:
            return raw

    # 3) suggérer une forme suffixée uniquement
    sug = match_suffix_fallback(raw, known_modifiers, known_tones)
    if sug and (sug.endswith("y") or sug.endswith("ish")) and (sug in known_modifiers):
        return sug

    # 4) -y variant de la base canonique
    if mod_canonical:
        y = build_y_variant(mod_canonical)
        if y and y in known_modifiers:
            return y

        # 5) canonique si valide
        if mod_canonical in known_modifiers:
            return mod_canonical

    # 6) fallback: garder la surface d'origine
    return raw


# =============================================
# 1. Compound Builder
# =============================================
def attempt_mod_tone_pair(
    mod_candidate: str,
    tone_candidate: str,
    compounds: set,
    raw_compounds: list,
    known_modifiers: set,
    known_tones: set,
    all_webcolor_names: set,
    llm_client,
    debug: bool = False,
):
    """
    Does:
        Resolves a (modifier, tone) pair into a validated compound.
        Uses token resolution, LLM fallback, and semantic conflict checks.
    """
    if debug:
        print("──────────── 🧪 attempt_mod_tone_pair ────────────")
        print(f"[🔍 MODIFIER CANDIDATE] '{mod_candidate}'")
        print(f"[🔍 TONE CANDIDATE]     '{tone_candidate}'")

    resolved: dict[str, str] = {}
    for role, candidate, is_tone_role in [
        ("modifier", mod_candidate, False),
        ("tone", tone_candidate, True),
    ]:
        if debug:
            print(f"\n[🔎 RESOLUTION START] Role: {role} | Candidate: '{candidate}'")

        # 🧨 Semantic pre-block: avoid fuzzy if candidate conflicts with any known modifier
        if (
            role == "modifier"
            and any(
                frozenset([candidate, known]) in SEMANTIC_CONFLICTS
                for known in known_modifiers
            )
        ):
            if debug:
                print(
                    f"[🧨 SEMANTIC BLOCK PRE-RESOLVE] '{candidate}' conflicts with known modifier"
                )
            return

        result = resolve_modifier_token(
            candidate,
            known_modifiers,
            known_tones=known_tones,
            allow_fuzzy=False,
            is_tone=is_tone_role,
            debug=debug,
        )

        if debug and result:
            print(f"[✅ RESOLVED VIA TOKEN] '{candidate}' → '{result}'")

        if not result and role == "tone" and (
            candidate in known_tones or candidate in all_webcolor_names
        ):
            result = candidate
            if debug:
                print(f"[⚠️ DIRECT TONE ACCEPTED] '{candidate}' is known")

        if not result:
            if debug:
                print(
                    f"[💡 LLM FALLBACK] Trying to simplify '{candidate}' for role '{role}'"
                )
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
                    print(f"[✨ SIMPLIFIED] '{candidate}' → '{result}' via LLM")

                # 🧨 Check semantic conflict after LLM simplification
                if (
                    role == "modifier"
                    and result != candidate
                    and candidate not in known_modifiers
                    and result not in known_modifiers
                    and frozenset([candidate, result]) in SEMANTIC_CONFLICTS
                ):
                    if debug:
                        print(
                            f"[🧨 FUZZY CONFLICT BLOCKED] '{candidate}' → '{result}' is semantically invalid"
                        )
                    return
            else:
                if debug:
                    print(
                        f"[⛔ LLM REJECTED] No simplification found for '{candidate}'"
                    )
                return

        resolved[role] = result
        if debug:
            print(f"[✅ FINAL {role.upper()}] '{candidate}' → '{result}'")

    mod, tone = resolved["modifier"], resolved["tone"]

    if debug:
        print("\n[🎯 FINAL RESOLUTION]")
        print(f"[🎯 MODIFIER] '{mod_candidate}' → '{mod}'")
        print(f"[🎯 TONE]     '{tone_candidate}' → '{tone}'")

    if mod_candidate != mod and mod not in known_modifiers:
        if debug:
            print(
                f"[⛔ MODIFIER REWRITE BLOCKED] '{mod_candidate}' → '{mod}' not trusted"
            )
        return

    if (tone.endswith("y") or tone.endswith("ish")) and tone not in known_tones and tone not in all_webcolor_names:
        if debug:
            print(f"[⛔ INVALID SUFFIXY TONE] '{tone}' not in known tone lists")
        return

    compound = f"{mod} {tone}"
    compounds.add(compound)
    raw_compounds.append(compound)

    if debug:
        print(f"[✅ COMPOUND DETECTED] → '{compound}'")
        print("────────────────────────────────────────────\n")


# =============================================
# 3. Compound Recovery from Split Tokens
# =============================================
def is_plausible_modifier(token, known_modifiers, known_color_tokens):
    """
    Check if a token is a plausible color modifier by validating against known modifiers/tones,
    ensuring it is alphabetic and not too short, with a soft fallback to accept unknown but valid words.
    """
    if token in known_modifiers or token in known_color_tokens:
        return True
    if not token.isalpha() or len(token) < 3:
        return False
    return True  # soft fallback


def extract_from_split(
    tokens,
    compounds,
    raw_compounds,
    known_color_tokens,
    known_modifiers,
    known_tones,
    all_webcolor_names,
    debug: bool = True,
    *,
    aug_vocab: set | None = None,
):
    """
    Recovers glued compound tokens like 'dustyroseglow' or 'sunsetcoral' by splitting them
    into valid 2- or 3-part phrases. Conserve la forme 'surface' du modificateur.
    """
    for token in tokens:
        text = token.text.lower()
        # ⛔ skip par POS (on garde adj/noms/properNouns)
        pos = getattr(token, "pos_", None)
        if pos not in {"ADJ", "NOUN", "PROPN"}:
            if debug:
                print(f"[⏩ POS SKIP] '{text}' pos={pos}")
            continue

        if debug:
            print(f"\n[🔍 TOKEN] '{text}'")

        if text in known_modifiers or text in known_tones:
            if debug:
                print(f"[⏩ SKIP] '{text}' is a known tone or modifier")
            continue
        if any(text == c.replace(" ", "") for c in compounds):
            if debug:
                print(f"[⏩ SKIP] '{text}' matches existing compound")
            continue

        if debug:
            print(f"[🔍 ATTEMPT SPLIT] Checking token: '{text}'")

        parts = split_tokens_to_parts(text, known_modifiers | known_tones, debug=debug)
        if not parts or len(parts) not in {2, 3} or (len(parts) == 1 and parts[0] in known_tones):
            if debug:
                print(f"[⛔ SPLIT FAILED] → {parts}")
                print(f"[🔁 FALLBACK] Attempting glue splits for '{text}'")

            # ------- Fallback binaire gauche|droite -------
            for i in range(2, len(text) - 1):
                left = text[:i]
                right = text[i:]

                # Cas: right est un TON connu -> left doit être (ou devenir) un MOD valide
                if right in known_tones:
                    left_mod = match_suffix_fallback(left, known_modifiers, known_tones) or left
                    left_base = recover_base(left_mod, known_modifiers, known_tones, fuzzy_fallback=True)

                    if debug:
                        print(f"[🔍 FALLBACK CHECK] '{left}' + '{right}'")
                        print(f"[🔁 NORMALIZED LEFT] '{left}' → '{left_mod}'")
                        print(f"[🔁 RECOVERED BASE] '{left_mod}' → '{left_base}'")
                        print(f"[📚 IN MODIFIERS?] {left_mod in known_modifiers}")
                        print(f"[📚 IN COLORS?] {left_mod in known_color_tokens}")
                        print(f"[📚 BASE IN MODIFIERS?] {left_base in known_modifiers}")

                    if right in known_tones and is_plausible_modifier(left_mod, known_modifiers, known_color_tokens):
                        surface_left = _surface_modifier(left, left_base, known_modifiers, known_tones)
                        phrase = f"{surface_left} {right}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append(phrase)
                            if debug:
                                print(f"[✅ FALLBACK SPLIT] '{text}' → '{phrase}'")
                        break

                # Cas: left est un TON connu -> right doit être (ou devenir) un MOD valide
                elif left in known_tones:
                    right_mod = match_suffix_fallback(right, known_modifiers, known_tones) or right
                    right_base = recover_base(right_mod, known_modifiers, known_tones, fuzzy_fallback=True)

                    if debug:
                        print(f"[🔍 FALLBACK REVERSED CHECK] '{left}' + '{right}'")
                        print(f"[🔁 NORMALIZED RIGHT] '{right}' → '{right_mod}'")
                        print(f"[🔁 RECOVERED BASE] '{right_mod}' → '{right_base}'")
                        print(f"[📚 IN MODIFIERS?] {right_mod in known_modifiers}")
                        print(f"[📚 IN COLORS?] {right_mod in known_color_tokens}")
                        print(f"[📚 BASE IN MODIFIERS?] {right_base in known_modifiers}")

                    if (right_mod in known_modifiers) or (right_mod in known_color_tokens) or (right_base in known_modifiers):
                        surface_right = _surface_modifier(right, right_base, known_modifiers, known_tones)
                        phrase = f"{left} {surface_right}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append(phrase)
                            if debug:
                                print(f"[✅ FALLBACK SPLIT REVERSED] '{text}' → '{phrase}'")
                        break

            continue

        # Normalisation légère (-y/-ish éventuels) avant validation
        parts = [match_suffix_fallback(p, known_modifiers, known_tones) or p for p in parts]

        # ------- Combinaison 2-part -------
        if len(parts) == 2:
            first, second = parts

            first_is_tone = is_known_tone(first, known_tones, all_webcolor_names)
            first_canon_mod = resolve_modifier_token(first, known_modifiers, known_tones, is_tone=False, debug=debug)
            second_is_tone = is_known_tone(second, known_tones, all_webcolor_names)
            second_canon_mod = resolve_modifier_token(second, known_modifiers, known_tones, is_tone=False, debug=debug)

            if debug:
                print(f"[🔍 FIRST VALID?] '{first}' → tone={first_is_tone}, modifier={bool(first_canon_mod)}")
                print(f"[🔍 SECOND VALID?] '{second}' → tone={second_is_tone}, modifier={bool(second_canon_mod)}")

            phrase = None
            if (first_canon_mod or first_is_tone) and second_is_tone:
                if first_canon_mod:
                    surface_first = _surface_modifier(first, first_canon_mod, known_modifiers, known_tones)
                    phrase = f"{surface_first} {second}"
                else:
                    phrase = f"{first} {second}"
            elif first_is_tone and second_canon_mod:
                surface_second = _surface_modifier(second, second_canon_mod, known_modifiers, known_tones)
                phrase = f"{first} {surface_second}"

            if phrase:
                if phrase not in compounds:
                    compounds.add(phrase)
                    raw_compounds.append(phrase)
                    if debug:
                        print(f"[✅ 2-PART COMPOUND ADDED] '{text}' → '{phrase}'")
            else:
                if debug:
                    print(f"[❌ INVALID 2-PART COMBINATION] '{first} {second}'")

        # ------- Combinaison 3-part -------
        elif len(parts) == 3:
            first, second, third = parts

            first_canon_mod = resolve_modifier_token(first, known_modifiers, known_tones, is_tone=False, debug=debug)
            second_canon_mod = resolve_modifier_token(second, known_modifiers, known_tones, is_tone=False, debug=debug)
            third_canon_mod = resolve_modifier_token(third, known_modifiers, known_tones, is_tone=False, debug=debug)

            first_is_tone = is_known_tone(first, known_tones, all_webcolor_names)
            second_is_tone = is_known_tone(second, known_tones, all_webcolor_names)
            third_is_tone = is_known_tone(third, known_tones, all_webcolor_names)

            if debug:
                print(f"[🔍 3-PART STRUCTURE] '{first}' mod={bool(first_canon_mod)}, tone={first_is_tone}")
                print(f"[                     '{second}' mod={bool(second_canon_mod)}, tone={second_is_tone}]")
                print(f"[                     '{third}' mod={bool(third_canon_mod)}, tone={third_is_tone}]")

            def surf(token_, canon_):
                return _surface_modifier(token_, canon_, known_modifiers, known_tones) if canon_ else token_

            valid_1 = first_canon_mod and second_is_tone and third_canon_mod
            valid_2 = first_canon_mod and second_canon_mod and third_is_tone
            valid_3 = first_canon_mod and second_is_tone and third_is_tone

            phrase = None
            if valid_1:
                phrase = f"{surf(first, first_canon_mod)} {second} {surf(third, third_canon_mod)}"
            elif valid_2:
                phrase = f"{surf(first, first_canon_mod)} {surf(second, second_canon_mod)} {third}"
            elif valid_3:
                phrase = f"{surf(first, first_canon_mod)} {second} {third}"

            if phrase:
                if phrase not in compounds:
                    compounds.add(phrase)
                    raw_compounds.append(phrase)
                    if debug:
                        print(f"[✅ 3-PART COMPOUND ADDED] '{text}' → '{phrase}'")
            else:
                if debug:
                    print(f"[❌ INVALID 3-PART COMBINATION] '{first} {second} {third}'")

                if first_canon_mod:
                    merged_tone = f"{second} {third}"
                    if merged_tone in known_tones:
                        phrase = f"{surf(first, first_canon_mod)} {merged_tone}"
                        if phrase not in compounds:
                            compounds.add(phrase)
                            raw_compounds.append(phrase)
                            if debug:
                                print(f"[✅ MERGED TONE 3-PART COMPOUND] '{text}' → '{phrase}'")
                    else:
                        if debug:
                            print(f"[❌ MERGED FAILED] '{merged_tone}' not a known tone")


# =============================================
# 4. Compound Recovery from Glued Tokens
# =============================================
def extract_from_glued(
    tokens,
    compounds,
    raw_compounds,
    known_color_tokens,
    known_modifiers,
    known_tones,
    all_webcolor_names,
    debug: bool = True,
    *,
    aug_vocab: set | None = None,
):
    """
    Extracts compound color phrases from glued tokens like 'dustyrose' or 'greylavenderpink'.
    Conserve la forme 'surface' des modificateurs détectés.
    """
    for token in tokens:
        raw = token.text.lower()

        pos = getattr(token, "pos_", None)
        if pos not in {"ADJ", "NOUN", "PROPN"}:
            if debug:
                print(f"[⏩ POS SKIP] '{raw}' pos={pos}")
            continue

        if not raw.isalpha() or raw in known_color_tokens:
            if debug:
                print(f"[⛔ SKIP] Token '{raw}' is known or non-alpha")
            continue

        # Splitter principal pour “glued” — passer le vocab étendu si dispo
        parts = split_glued_tokens(
            raw,
            known_color_tokens,
            known_modifiers,
            debug=debug,
            vocab=aug_vocab,
        )

        # Fallback: 3-part récursif via split_tokens_to_parts
        if not parts or len(parts) not in {2, 3} or (len(parts) == 1 and parts[0] in known_tones):
            fallback_parts = split_tokens_to_parts(raw, known_modifiers | known_tones, debug=debug)
            if fallback_parts and len(fallback_parts) == 3:
                a, b, c = fallback_parts
                tone_b = is_known_tone(b, known_tones, all_webcolor_names)
                tone_c = is_known_tone(c, known_tones, all_webcolor_names)

                def normalize_mod(token_):
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
                        raw_compounds.append(phrase)
                        if debug:
                            print(f"[✅ GLUED 3-PART FALLBACK] '{raw}' → '{phrase}'")
            continue

        # --- 2-part ---
        if len(parts) == 2:
            a, b = parts

            mod_canon = resolve_modifier_token(a, known_modifiers, known_tones, is_tone=False, debug=debug)
            is_tone_b = is_known_tone(b, known_tones, all_webcolor_names)

            allow_tone_pair = False  # ← mets True si tu veux tolérer "red purple"
            tone_pair = allow_tone_pair and (a in known_tones and b in known_tones)

            mod_fallback = a in known_modifiers and is_tone_b
            mod_mod_pair = a in known_modifiers and b in known_modifiers

            if mod_canon and is_tone_b:
                surface_mod = _surface_modifier(a, mod_canon, known_modifiers, known_tones)
                compound = f"{surface_mod} {b}"
            elif mod_fallback or mod_mod_pair or tone_pair:
                compound = f"{a} {b}"
            else:
                continue

            if compound not in compounds:
                compounds.add(compound)
                raw_compounds.append(compound)
                if debug:
                    print(f"[✅ GLUED 2-PART COMPOUND] '{raw}' → '{compound}'")


        # ─── 3-part ─────────────────────
        elif len(parts) == 3:
            a, b, c = parts

            mod_a = resolve_modifier_token(a, known_modifiers, known_tones, is_tone=False, debug=debug)
            mod_b = resolve_modifier_token(b, known_modifiers, known_tones, is_tone=False, debug=debug)
            mod_c = resolve_modifier_token(c, known_modifiers, known_tones, is_tone=False, debug=debug)

            tone_a = is_known_tone(a, known_tones, all_webcolor_names)
            tone_b = is_known_tone(b, known_tones, all_webcolor_names)
            tone_c = is_known_tone(c, known_tones, all_webcolor_names)

            if debug:
                print(f"[🔍 3-PART CHECK] '{a} {b} {c}'")
                print(f"    → mod=[{bool(mod_a)}, {bool(mod_b)}, {bool(mod_c)}]")
                print(f"    → tone=[{tone_a}, {tone_b}, {tone_c}]")

            def surf(token_, canon_):
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
                    raw_compounds.append(compound)
                    if debug:
                        print(f"[✅ GLUED 3-PART COMPOUND] '{raw}' → '{compound}'")


# =============================================
# 5. Compound Extraction from Adjacent Tokens
# =============================================
def extract_from_adjacent(
    tokens,
    compounds,
    raw_compounds,
    known_modifiers,
    known_tones,
    debug: bool = True,
):
    """
    Does: Extracts compound color phrases from adjacent token pairs or
          multi-word modifiers followed by a tone.
    Accepts:
      - single modifier + tone (e.g. "soft pink")
      - two-word modifier + tone (e.g. "barely-there pink")
    Returns: Updates compounds set and raw_compounds list in-place.
    """
    n = len(tokens)

    for i in range(n - 1):
        raw_mod = tokens[i].text.lower()
        raw_tone = singularize(tokens[i + 1].text.lower())

        if debug:
            print(f"\n[🔍 ADJACENT PAIR] '{raw_mod}' + '{raw_tone}'")

        # 🚫 Skip if either token is a cosmetic noun
        if raw_mod in COSMETIC_NOUNS or raw_tone in COSMETIC_NOUNS:
            if debug:
                print(f"[⛔ COSMETIC BLOCK] skip '{raw_mod} {raw_tone}'")
            continue

        # ----- Standard 1-word modifier + tone -----
        mod_canon = resolve_modifier_token(
            raw_mod, known_modifiers, known_tones, is_tone=False, debug=debug
        )
        tone = raw_tone if raw_tone in known_tones else None

        if mod_canon and tone:
            surface_mod = _surface_modifier(raw_mod, mod_canon, known_modifiers, known_tones)
            phrase = f"{surface_mod} {tone}"
            if phrase not in compounds:
                if debug:
                    print(f"[✅ ADJACENT COMPOUND ADDED] → '{phrase}'")
                compounds.add(phrase)
                raw_compounds.append(phrase)

        # ----- Extended 2-word modifier + tone -----
        if i < n - 2:
            m1 = tokens[i].text.lower()
            m2 = tokens[i + 1].text.lower()
            combined_mod_raw = f"{m1}-{m2}"
            raw_tone2 = singularize(tokens[i + 2].text.lower())

            if debug:
                print(f"[🔍 2-WORD MODIFIER CHECK] '{combined_mod_raw}' + '{raw_tone2}'")

            # 🚫 Skip if any part is a cosmetic noun
            if m1 in COSMETIC_NOUNS or m2 in COSMETIC_NOUNS or raw_tone2 in COSMETIC_NOUNS:
                if debug:
                    print(f"[⛔ COSMETIC BLOCK] skip '{combined_mod_raw} {raw_tone2}'")
                continue

            mod2_canon = resolve_modifier_token(
                combined_mod_raw, known_modifiers, known_tones, is_tone=False, debug=debug
            )
            tone2 = raw_tone2 if raw_tone2 in known_tones else None

            if mod2_canon and tone2:
                surface_mod2 = _surface_modifier(combined_mod_raw, mod2_canon, known_modifiers, known_tones)
                phrase2 = f"{surface_mod2} {tone2}"
                if phrase2 not in compounds:
                    if debug:
                        print(f"[✅ ADJACENT COMPOUND ADDED] → '{phrase2}'")
                    compounds.add(phrase2)
                    raw_compounds.append(phrase2)


# =============================================
# 6. Compound Extraction Orchestrator
# =============================================
def extract_compound_phrases(
    tokens,
    compounds,
    raw_compounds,
    known_color_tokens,
    known_modifiers,
    known_tones,
    all_webcolor_names,
    raw_text: str = "",
    debug: bool = False,
):
    """
    Orchestrates full compound color extraction via all strategies + fallback recovery.
    Applies all extractors and fuzzy recovery. Filters blocked combinations at the end.
    """
    if raw_text:
        raw_text = raw_text.replace("-", " ")
        tokens = nlp(raw_text)

    # 🔑 Build once per segment: vocab étendu (énorme perf gain sur glued splits)
    aug_vocab = build_augmented_suffix_vocab(known_color_tokens, known_modifiers)
    if debug:
        try:
            print(f"[DBG] aug_vocab size={len(aug_vocab)}")
        except Exception:
            pass

    # Main extractors (on passe aug_vocab aux extracteurs qui splittent les tokens collés)
    extract_from_adjacent(tokens, compounds, raw_compounds, known_modifiers, known_tones, debug)
    extract_from_split(
        tokens, compounds, raw_compounds,
        known_color_tokens, known_modifiers, known_tones, all_webcolor_names,
        debug=debug, aug_vocab=aug_vocab
    )
    extract_from_glued(
        tokens, compounds, raw_compounds,
        known_color_tokens, known_modifiers, known_tones, all_webcolor_names,
        debug=debug, aug_vocab=aug_vocab
    )

    # Fallback: fuzzy match missed modifier + tone pairs
    for i in range(len(tokens) - 1):
        left = normalize_token(tokens[i].text, keep_hyphens=True)
        right = normalize_token(tokens[i + 1].text, keep_hyphens=True)

        if left in known_modifiers or left in known_tones or len(left) < 3:
            continue
        if right not in known_tones or not right.isalpha():
            continue

        mod_canon = resolve_modifier_token(
            left,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            allow_fuzzy=True,
            is_tone=False,
            debug=debug,
        )

        if mod_canon:
            surface_mod = _surface_modifier(left, mod_canon, known_modifiers, known_tones)
            phrase = f"{surface_mod} {right}"
            if phrase not in compounds:
                compounds.add(phrase)
                raw_compounds.append((surface_mod, right))
                if debug:
                    print(f"[🩹 FALLBACK PATCH] {left}+{right} → {phrase}")

    # Remove blocked (modifier, tone) pairs
    blocked = {
        (m, t)
        for pair in raw_compounds
        if isinstance(pair, tuple) and len(pair) == 2
        for m, t in [pair]
        if is_blocked_modifier_tone_pair(m, t)
    }

    for mod, tone in blocked:
        phrase = f"{mod} {tone}"
        if phrase in compounds:
            compounds.discard(phrase)
        try:
            raw_compounds.remove((mod, tone))
        except ValueError:
            pass
        if debug:
            print(f"[⛔ BLOCKED PAIR REMOVED] '{phrase}'")
