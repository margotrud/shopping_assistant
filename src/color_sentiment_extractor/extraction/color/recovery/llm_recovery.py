# extraction/color/recovery/llm_recovery.py
import re
from color_sentiment_extractor.extraction.color.constants import COSMETIC_NOUNS
from color_sentiment_extractor.extraction.color.recovery.modifier_resolution import resolve_modifier_token
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base  # ← import top-level

# 🔒 Teintes interdites en autonome (on ne veut pas les promouvoir en "tone" seules)
AUTONOMOUS_TONE_BAN = {"dust", "glow"}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _preserve_surface_mod_when_valid_pair(text: str, known_modifiers: set, known_tones: set, debug: bool=False) -> str:
    """
    Si 'left right' avec right ∈ known_tones et left est une surface suffixée (-y/-ish)
    dont la base ∈ known_modifiers, on conserve la surface telle quelle (ex: 'dusty rose').
    """
    if not text:
        return text
    normalized = text.strip().lower()
    m = re.match(r"^\s*([a-z\-]+)\s+([a-z][a-z\-\s]*)\s*$", normalized)
    if not m:
        return text

    left, right = m.group(1), m.group(2)
    if right not in known_tones:
        return text

    if left.endswith("y") or left.endswith("ish"):
        base = recover_base(
            left,
            known_modifiers=known_modifiers,
            known_tones=known_tones,
            fuzzy_fallback=False,
            debug=False
        )
        if base and base in known_modifiers:
            if debug:
                print(f"[🛡️ PRESERVE SURFACE] '{left} {right}' (base='{base}')")
            return f"{left} {right}"

    return text


# ------------------------------------------------------------
# Core
# ------------------------------------------------------------
def _attempt_simplify_token(
    token: str,
    known_modifiers: set,
    known_tones: set,
    llm_client,
    role: str = "modifier",
    debug: bool = True
) -> str | None:
    """
    Uses LLM to simplify a noisy token into a known tone or modifier.
    Returns a valid simplified form or None.
    """
    if debug:
        print("─────── 🧪 _attempt_simplify_token ───────")
        print(f"[INPUT] token = '{token}'   | role = '{role}'")
        print(f"[STEP] Calling simplify_phrase_if_needed...")

    simplified = simplify_phrase_if_needed(token, known_modifiers, known_tones, llm_client, debug=debug)

    if debug:
        print(f"[SIMPLIFIED] LLM result: '{simplified}'")

    if simplified:
        # récupérer le 1er mot (et non le 1er caractère !)
        index = 0 if role == "modifier" else -1
        words = simplified.strip().split()
        raw_result = words[index] if words else simplified.strip()
        result = normalize_token(raw_result, keep_hyphens=True)

        if debug:
            print(f"[PARSE] Extracted result = '{raw_result}' → normalized = '{result}'")
            print(f"[CHECK] Is '{result}' in known_modifiers? → {result in known_modifiers}")
            print(f"[CHECK] Is '{result}' in known_tones?     → {result in known_tones}")

        # 🚫 banlist pour tones autonomes indésirables
        if role == "tone" and result in AUTONOMOUS_TONE_BAN:
            if debug:
                print(f"[⛔ BANLIST] '{result}' is not allowed as a standalone tone")
            return None

        if (
            (role == "modifier" and result in known_modifiers) or
            (role == "tone" and result in known_tones) or
            result in known_modifiers or result in known_tones  # fallback flex
        ):
            if debug:
                print(f"[✅ RETURN] Final accepted result: '{result}'")
            return result
        else:
            if debug:
                print(f"[⛔ REJECT] Simplified token '{result}' not in known sets")
            # 🔁 Fallback: base recovery
            recovered = recover_base(
                result,
                known_modifiers=known_modifiers,
                known_tones=known_tones,
                fuzzy_fallback=True,
                fuzzy_threshold=78,
                use_cache=False,
                debug=debug,
                depth=0,
            )
            # Re-check banlist si on a demandé un tone
            if role == "tone" and recovered in AUTONOMOUS_TONE_BAN:
                if debug:
                    print(f"[⛔ BANLIST] recovered '{recovered}' disallowed as standalone tone")
                return None

            if recovered:
                if debug:
                    print(f"[✅ FALLBACK RECOVERY] '{result}' → '{recovered}'")
                return recovered

    else:
        if debug:
            print(f"[⛔ REJECT] No simplification result for '{token}'")

    if debug:
        print(f"[FINAL] Returning: None")
        print("────────────────────────────────────────\n")

    return None


def _extract_filtered_tokens(tokens, known_modifiers, known_tones, llm_client, debug):
    """
    Extracts modifier or tone tokens from a token stream using resolution logic,
    with fallback to LLM simplification and several safety filters.
    """
    result = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if debug:
            print(f"\n[🧪 TOKEN] '{tok.text}' → normalized: '{raw}' (POS={tok.pos_})")
            print(f"[🔎 CHECK] In COSMETIC_NOUNS? → {raw in COSMETIC_NOUNS}")

        # Block known cosmetic nouns
        if raw in COSMETIC_NOUNS:
            if debug:
                print(f"[⛔ SKIPPED] Cosmetic noun '{raw}' blocked")
            continue

        # Skip connectors via POS tag
        if tok.pos_ == "CCONJ":
            if debug:
                print(f"[⛔ SKIPPED] Connector '{raw}' ignored (POS=CCONJ)")
            continue

        # Rule-based resolver first
        resolved = resolve_modifier_token(raw, known_modifiers, known_tones)

        # Fallback to LLM simplifier
        if not resolved:
            simplified = simplify_phrase_if_needed(raw, known_modifiers, known_tones, llm_client, debug=debug)
            if simplified:
                # FIX: prendre le 1er mot correctement
                resolved_candidate = simplified.strip().split()[0]
                if resolved_candidate in known_modifiers or resolved_candidate in known_tones:
                    resolved = resolved_candidate
                    if debug:
                        print(f"[🔁 SIMPLIFIED FALLBACK] '{raw}' → '{resolved}'")

        if debug:
            print(f"[🔍 RESOLVED] '{raw}' → '{resolved}'")
            print(f"[📌 raw ∈ tones?] {raw in known_tones}")
            print(f"[📌 resolved ∈ tones?] {resolved in known_tones if resolved else '—'}")
            print(f"[📏 resolved == raw?] {resolved == raw if resolved else '—'}")
            print(f"[📏 resolved starts with raw?] {resolved.startswith(raw) if resolved else '—'}")
            print(f"[📐 contains hyphen?] {'-' in resolved if resolved else '—'}")
            print(f"[🧮 total matches so far] {len(result)}")

        # Safety filters
        if len(raw) <= 3 and resolved != raw and resolved not in known_modifiers and resolved not in known_tones:
            if debug:
                print(f"[⛔ REJECTED] Token '{raw}' too short for safe fuzzy match → '{resolved}'")
            continue

        if resolved and "-" in resolved and not resolved.startswith(raw):
            if debug:
                print(f"[⛔ REJECTED] Fuzzy '{raw}' → '{resolved}' (compound mismatch)")
            continue

        if resolved and " " in resolved and " " not in raw:
            if debug:
                print(f"[⛔ REJECTED] Fuzzy '{raw}' → '{resolved}' (multi-word from single token)")
            continue

        if len(result) >= 3 and resolved and resolved != raw:
            if debug:
                print(f"[⛔ REJECTED] Skipping fuzzy '{raw}' → '{resolved}' (already 3+ matches)")
            continue

        if resolved:
            result.add(resolved)
            if debug:
                print(f"[🎯 STANDALONE MATCH] '{raw}' → '{resolved}'")

    return result


# ------------------------------------------------------------
# LLM wrappers
# ------------------------------------------------------------
def build_prompt(phrase: str) -> str:
    return f"What is the simplified base color or tone implied by: '{phrase}'?"


def simplify_color_description_with_llm(phrase: str, llm_client, cache=None, debug=False) -> str:
    prompt = build_prompt(phrase)
    if debug:
        print(f"[🧠 LLM PROMPT] {prompt}")

    if cache:
        cached = cache.get_simplified(phrase)
        if cached:
            if debug:
                print(f"[🗃️ CACHE HIT] '{phrase}' → '{cached}'")
            return cached

    simplified = llm_client.simplify(prompt)

    if cache:
        cache.store_simplified(phrase, simplified)

    if debug:
        print(f"[🧠 LLM RESPONSE] '{phrase}' → '{simplified}'")
    return simplified


def simplify_phrase_if_needed(phrase, known_modifiers, known_tones, llm_client, cache=None, debug=False):
    """
    Attempts to simplify a descriptive phrase only if it isn't already a known tone.
    Also preserves '-y/-ish' surface when we already have a valid (modifier, tone) pair.
    """
    if llm_client is None:
        return None
    if debug:
        print(f"[🔍 SIMPLIFY] Checking phrase: '{phrase}'")

    # 1) preservation avant toute chose (ne pas aplatir 'dusty rose')
    preserved = _preserve_surface_mod_when_valid_pair(phrase, known_modifiers, known_tones, debug=debug)
    if preserved != phrase:
        # Phrase reconnue comme couple surface valide → on retourne tel quel
        return preserved

    normalized = phrase.lower().strip()
    if normalized in known_tones:
        if debug:
            print(f"[✅ EXACT MATCH] '{phrase}' is a known tone (no fallback)")
        return phrase

    simplified = simplify_color_description_with_llm(
        phrase=phrase,
        llm_client=llm_client,
        cache=cache,
        debug=debug
    )
    if simplified and simplified != phrase:
        if debug:
            print(f"[✨ LLM SIMPLIFIED] '{phrase}' → '{simplified}'")
        # re-apply preservation on the LLM output as well
        simplified = _preserve_surface_mod_when_valid_pair(simplified, known_modifiers, known_tones, debug=debug)
        return simplified

    if debug:
        print(f"[⚠️ UNSIMPLIFIED] No simplification applied, returning raw phrase")
    return phrase
