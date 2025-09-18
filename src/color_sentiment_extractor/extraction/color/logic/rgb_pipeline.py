"""
rgb_pipeline.py
==========

Orchestration logic for resolving RGB from descriptive color phrases
using LLM-based generation, simplification, and fallback strategies.

Used By:
--------
- color resolution pipelines
- Modifier-tone RGB mapping
- User input parsing and RGB grounding
"""
import re
print("[DEBUG] rgb_pipeline loaded from:", __file__)

from typing import Optional, Tuple, Set

from color_sentiment_extractor.extraction.color.constants import SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.color.llm.llm_api_client import query_llm_for_rgb
from color_sentiment_extractor.extraction.color.recovery.llm_recovery import simplify_color_description_with_llm, simplify_phrase_if_needed
from color_sentiment_extractor.extraction.color.utils.rgb_distance import (
    fuzzy_match_rgb_from_known_colors,
    _try_simplified_match,
)
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.color.suffix.rules import build_y_variant
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token


# rgb_pipeline.py

def _project_to_known_vocab(s: str, known_modifiers: Set[str], known_tones: Set[str]) -> str:
    if not s:
        return ""
    toks = s.lower().strip().split()
    toks = [t for t in toks if (t in known_modifiers or t in known_tones)]
    if not toks:
        return ""

    # priorité: (modifier + tone) si possible, sinon tone, sinon modifier
    mod = next((t for t in toks if t in known_modifiers), None)
    tone = next((t for t in toks if t in known_tones), None)
    if tone and mod:
        return f"{mod} {tone}"
    return tone or mod


def _sanitize_simplified(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\- ]+", " ", s)  # lettres / espace / tiret
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    # drop tokens d’1 lettre (p.ex. 's') et tronque à 2 mots
    parts = [w for w in parts if len(w) >= 2][:2]
    return " ".join(parts)
def _normalize_modifier_tone(
    phrase: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False
) -> str:
    """
    Normalise (modifier, tone) sans aplatir inutilement les surfaces -y/-ish.
    Ex.: 'dusty rose' reste 'dusty rose' si 'rose' ∈ known_tones et base('dusty') ∈ known_modifiers.
    """
    if not phrase:
        return ""
    p = normalize_token(phrase)  # garde seulement a-z, espaces, tirets → cohérent avec le reste
    parts = p.split()
    if len(parts) != 2:
        return p

    left, right = parts[0], parts[1]

    # Si la droite n'est pas une teinte connue, on ne touche pas
    if right not in known_tones:
        return p

    # 1) si left est déjà un mod valide → garder tel quel
    if left in known_modifiers:
        if debug:
            print(f"[🧩 PRE-NORM KEEP] modifier already valid: '{left} {right}'")
        return f"{left} {right}"

    # 2) si left se termine par -y/-ish et que sa base est un mod connu → préserver la surface
    if left.endswith("y") or left.endswith("ish"):
        base = recover_base(left, known_modifiers=known_modifiers, known_tones=known_tones, allow_fuzzy=False)
        if base and base in known_modifiers:
            if debug:
                print(f"[🧩 PRE-NORM PRESERVE] '{left} {right}' (base='{base}')")
            return f"{left} {right}"

    # 3) sinon, tenter de récupérer une base mod
    base = recover_base(left, known_modifiers=known_modifiers, known_tones=known_tones, allow_fuzzy=False)
    if base and base in known_modifiers:
        # Essayer une variante -y de la base si elle existe et est connue
        y_form = build_y_variant(base)
        if y_form and y_form in known_modifiers:
            if debug:
                print(f"[🧩 Y-VARIANT] '{base}' -> '{y_form}'")
            return f"{y_form} {right}"
        # Sinon revenir à la base canonique
        if debug:
            print(f"[🧩 PRE-NORM CANON] '{left}' → '{base}' (return '{base} {right}')")
        return f"{base} {right}"

    # 4) fallback: ne rien casser
    return f"{left} {right}"


def get_rgb_from_descriptive_color_llm_first(
    input_color: str,
    llm_client,
    cache=None,
    debug=False
) -> Optional[Tuple[int, int, int]]:
    """
    Does: Resolves RGB from a descriptive color.
          If llm_client is None -> skip LLM steps and do pure fuzzy/css fallbacks.
    """
    if debug:
        print(f"[🌈 START] input_color = '{input_color}'")

    # 🚫 Pas d'LLM → on saute direct aux fallbacks non-LLM
    if not llm_client:
        if debug:
            print("[🛑 NO LLM CLIENT] Skipping LLM steps, using fallbacks only")
        # 1) essai direct (CSS/XKCD) sur le libellé brut
        rgb = _try_simplified_match(input_color, debug=debug)
        if debug:
            print(f"[🔍 _try_simplified_match(raw)] → {rgb}")
        if rgb:
            return rgb
        # 2) fuzzy sur le libellé brut
        rgb = fuzzy_match_rgb_from_known_colors(input_color)
        if debug:
            print(f"[🌀 fuzzy_match_rgb_from_known_colors(raw)] → {rgb}")
        return rgb

    # ✅ LLM dispo → pipeline original
    rgb = query_llm_for_rgb(input_color, llm_client, cache=cache, debug=debug)
    if debug:
        print(f"[🎯 LLM RGB RETURNED] → {rgb}")
    if rgb:
        return rgb

    simplified = simplify_color_description_with_llm(input_color, llm_client, cache=cache, debug=debug)
    if debug:
        print(f"[🧽 SIMPLIFIED] → '{simplified}'")

    rgb = _try_simplified_match(simplified, debug=debug)
    if debug:
        print(f"[🔍 _try_simplified_match] → {rgb}")
    if rgb:
        return rgb

    rgb = fuzzy_match_rgb_from_known_colors(simplified)
    if debug:
        print(f"[🌀 fuzzy_match_rgb_from_known_colors] → {rgb}")
    return rgb



def resolve_rgb_with_llm(
    phrase: str,
    llm_client,
    cache=None,
    debug=False,
    prefer_db_first: bool = False,   # ← ajout
) -> Optional[Tuple[int, int, int]]:
    """
    Does: Entry point for RGB resolution from color phrases using LLM and fallbacks.
    """
    if prefer_db_first or not llm_client:
        # 1) essai direct (CSS/XKCD) sur le libellé brut
        rgb = _try_simplified_match(phrase, debug=debug)
        if debug:
            print(f"[🔍 _try_simplified_match(prefer_db_first)] → {rgb}")
        if rgb:
            return rgb
        # 2) fuzzy
        rgb = fuzzy_match_rgb_from_known_colors(phrase)
        if debug:
            print(f"[🌀 fuzzy_match_rgb_from_known_colors(prefer_db_first)] → {rgb}")
        if rgb or not llm_client:
            return rgb

    # mode LLM-first (comportement d’origine)
    return get_rgb_from_descriptive_color_llm_first(
        input_color=phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug
    )


def _is_known_color_token(tok: str, known_modifiers: Set[str], known_tones: Set[str]) -> bool:
    if not tok:
        return False
    t = tok.strip().lower()
    return (t in known_tones) or (t in known_modifiers)

def process_color_phrase(
    phrase: str,
    known_modifiers: Set[str],
    known_tones: Set[str],
    llm_client=None,
    cache=None,
    debug: bool = True
) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """
    Simplifie une phrase couleur vers (modifier + tone) stable et résout son RGB.
    Ordre:
      1) règles pures (pas d'LLM)
      1.5) pré-normalisation directe sur la phrase brute (ex. 'dust rose' -> 'dusty rose')
      2) fallback LLM si on n'a rien gagné (sauf si pré-norm verrouillée OU token connu)
      2.5) sanitize (+ drop tokens d'1 lettre) puis normalisation mod/tone
      3) fix sémantique
      4) résolution RGB (LLM-first + fallbacks)
    """
    if debug:
        print("\n" + "═" * 60)
        print(f"[🧪 INPUT PHRASE] '{phrase}'")
        print(f"[⚙️ MODIFIERS] Count = {len(known_modifiers)}")
        print(f"[⚙️ TONES]     Count = {len(known_tones)}")

    # --- NEW: si phrase est déjà un token connu (mod ou tone), on fige et on évite l'LLM
    raw_norm = normalize_token(phrase)
    if " " not in raw_norm and _is_known_color_token(raw_norm, known_modifiers, known_tones):
        simplified = raw_norm
        if debug:
            print(f"[🧷 KNOWN TOKEN LOCK] '{phrase}' → '{simplified}' — skipping LLM")
        rgb = resolve_rgb_with_llm(
            simplified,
            llm_client=llm_client,
            cache=cache,
            debug=debug,
            prefer_db_first=True,   # DB/XKCD puis fuzzy avant toute estimation LLM
        )
        if debug:
            print(f"[🎨 FINAL RGB] '{simplified}' → {rgb}")
            print("═" * 60 + "\n")
        return simplified, (None if rgb is False else rgb)

    # 1) Règles sans LLM
    simplified = simplify_phrase_if_needed(
        phrase, known_modifiers, known_tones,
        llm_client=None,
        cache=cache, debug=debug
    )
    simplified = (simplified or "").strip()
    if debug:
        print(f"[🔍 AFTER RULE SIMPLIFICATION] → '{simplified}'")

    # 1.5) Pré-normalisation directe sur la phrase brute
    pre_norm = _normalize_modifier_tone(
        _sanitize_simplified(phrase),
        known_modifiers,
        known_tones,
        debug=debug
    )
    pre_parts = pre_norm.split()
    pre_norm_locked = (len(pre_parts) == 2 and pre_parts[1] in known_tones)

    if pre_norm_locked:
        if debug:
            if pre_norm == phrase:
                print(f"[🧩 PRE-NORM LOCK] '{pre_norm}' (already good) — skipping LLM")
            else:
                print(f"[🧩 PRE-NORM OVERRIDE] '{phrase}' → '{pre_norm}' — skipping LLM")
        simplified = pre_norm
    else:
        if pre_norm and not simplified:
            simplified = pre_norm

    # 2) Fallback LLM UNIQUEMENT si pas de lock et pas déjà mieux qu'une forme mod+tone
    go_llm = (
        not pre_norm_locked
        and llm_client is not None
        and (
            not simplified
            or simplified == phrase
            or not (len(simplified.split()) == 2 and simplified.split()[1] in known_tones)
        )
    )
    if go_llm:
        if debug:
            print(f"[🧠 LLM FALLBACK] Trying LLM for '{phrase}'...")
        llm_simpl = simplify_color_description_with_llm(
            phrase, llm_client=llm_client, cache=cache, debug=debug
        )
        llm_simpl = (llm_simpl or "").strip()
        if llm_simpl:
            simplified = llm_simpl
            if debug:
                print(f"[🤖 LLM RETURNED] → '{simplified}'")

    # 2.5) Sanitize + normalisation finale (idempotent — ne casse pas le lock)
    simplified = _sanitize_simplified(simplified)
    if debug:
        print(f"[🧽 SANITIZED] → '{simplified}'")
    simplified = _normalize_modifier_tone(simplified, known_modifiers, known_tones, debug=debug)

    # 3) Fix sémantique
    if simplified:
        tokens = simplified.split()
        for i, t in enumerate(tokens):
            for conflict in SEMANTIC_CONFLICTS:
                if t in conflict:
                    replacement = sorted(conflict - {t})[0]
                    if debug:
                        print(f"[⚠️ CONFLICT] '{t}' ∈ {set(conflict)} → '{replacement}'")
                    tokens[i] = replacement
                    break
        simplified = " ".join(tokens)

    if debug:
        print(f"[✅ AFTER SEMANTIC FIX] → '{simplified}'")

    # 4) RGB (DB/fuzzy priorisés si on a un lock de pré-norm)
    rgb = resolve_rgb_with_llm(
        simplified or phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug,
        prefer_db_first=pre_norm_locked
    )

    if debug:
        print(f"[🎨 FINAL RGB] '{simplified}' → {rgb}")
        print("═" * 60 + "\n")

    return simplified or "", (None if rgb is False else rgb)
