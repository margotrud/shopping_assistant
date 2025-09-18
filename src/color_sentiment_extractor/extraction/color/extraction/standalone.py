# ──────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────

from color_sentiment_extractor.extraction.color.constants import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.recovery.llm_recovery import _extract_filtered_tokens
from color_sentiment_extractor.extraction.general.expression.expression_helpers import _inject_expression_modifiers
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token


# ──────────────────────────────────────────────────────────────
# 2. Tone-only extraction (strict, no LLM)
# ──────────────────────────────────────────────────────────────

def extract_lone_tones(tokens, known_tones, debug=False):
    """
       Does: Extracts standalone tone tokens found directly in the input token stream.
             Skips cosmetic nouns and matches only tokens present in the known tone set.

       Returns: Set of normalized tone tokens found in input.
       """
    matches = set()
    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)
        if raw in COSMETIC_NOUNS:
            if debug:
                print(f"[⛔ COSMETIC BLOCK] '{raw}' blocked")
            continue
        if raw in known_tones:
            matches.add(raw)
            if debug:
                print(f"[🎯 LONE TONE] Found '{raw}'")
    return matches


# ──────────────────────────────────────────────────────────────
# 3. Injection gating/capping helper
# ──────────────────────────────────────────────────────────────

def _gate_and_cap_injection(tokens, expression_map, known_modifiers, injected, max_injected=5, debug=False):
    """
    Garde uniquement les modificateurs injectés qui sont:
      - réellement déclenchés par un alias présent dans les tokens,
      - connus dans known_modifiers,
      - non déjà présents dans l'entrée,
    puis limite à max_injected (ordre de rencontre des alias).
    """
    present = [normalize_token(t.text, keep_hyphens=True) for t in tokens]
    present_set = set(present)
    allowed = []
    seen = set()

    # on déroule les alias présents dans l'ordre d'apparition
    for alias in present:
        exprs = expression_map.get(alias, [])
        for m in exprs:
            if m in known_modifiers and m not in present_set and m not in seen:
                seen.add(m)
                allowed.append(m)
                if len(allowed) >= max_injected:
                    if debug:
                        print(f"[🔒 INJECTION CAPPED] {len(allowed)} terms kept")
                    return allowed

    # intersect avec la liste “injected” originale pour ne pas surprendre l’ordre
    if injected:
        allowed_set = set(allowed)
        ordered_intersection = [m for m in injected if m in allowed_set]
        if debug:
            print(f"[🧰 INJECTION GATED] kept={ordered_intersection}")
        return ordered_intersection[:max_injected]

    return allowed[:max_injected]


# ──────────────────────────────────────────────────────────────
# 4. Final combination helper
# ──────────────────────────────────────────────────────────────

def _finalize_standalone_phrases(injected, filtered, debug):
    """
    Does: Combines expression-injected modifiers with resolved token matches.
    Returns: Unified set of standalone color terms.
    """
    injected_set = set(injected or [])
    filtered_set = set(filtered or [])
    combined = injected_set | filtered_set  # set union
    if debug:
        print(f"[✅ FINAL STANDALONE SET] {combined}")
    return combined


# ──────────────────────────────────────────────────────────────
# 5. Main entrypoint for standalone phrase extraction
# ──────────────────────────────────────────────────────────────

def extract_standalone_phrases(tokens, known_modifiers, known_tones, expression_map, llm_client, debug=True):
    """
    Does: Extracts standalone tone or modifier tokens from input using three strategies:
          - Expression-based modifier injection (gated + capped)
          - Rule-based + LLM fallback resolution
          - Final union and cleanup

    Returns: Set of valid standalone modifiers and tones found in the input tokens.
    """
    if debug:
        print("\n" + "="*70)
        print("🎯 ENTER extract_standalone_phrases()")
        print("="*70)
        print("[🧪 INPUT TOKENS]")
        for i, t in enumerate(tokens):
            print(f"  {i:02d}: '{t.text}' (POS={t.pos_})")
        print(f"[📚 #TOKENS] {len(tokens)} | #MODIFIERS: {len(known_modifiers)} | #TONES: {len(known_tones)}")

    # ─────────────────────────────────────────────
    # 1. Expression-based modifier injection (gated + capped)
    # ─────────────────────────────────────────────
    injected_raw = _inject_expression_modifiers(tokens, known_modifiers, known_tones, expression_map, debug)
    gated_injected = _gate_and_cap_injection(
        tokens=tokens,
        expression_map=expression_map,
        known_modifiers=known_modifiers,
        injected=injected_raw,
        max_injected=5,
        debug=debug
    )
    if debug:
        print("\n[🧬 EXPRESSION INJECTION] Modifiers from expression map (gated+cap):")
        for term in sorted(gated_injected): print(f"  • {term}")
        print(f"  → Total: {len(gated_injected)} terms")

    # ─────────────────────────────────────────────
    # 2. Rule + LLM fallback filtering
    # ─────────────────────────────────────────────
    filtered_terms = _extract_filtered_tokens(tokens, known_modifiers, known_tones, llm_client, debug)
    if debug:
        print("\n[🧠 RULE + LLM FILTERED TOKENS]")
        for term in sorted(filtered_terms): print(f"  • {term}")
        print(f"  → Total: {len(filtered_terms)} terms")

    # ─────────────────────────────────────────────
    # 3. Final combination + cleanup
    # ─────────────────────────────────────────────
    final = _finalize_standalone_phrases(gated_injected, filtered_terms, debug)
    if debug:
        print("\n[🏁 FINAL STANDALONE PHRASES]")
        for term in sorted(final): print(f"  • {term}")
        print(f"  ✅ Final count: {len(final)}")

    return final
