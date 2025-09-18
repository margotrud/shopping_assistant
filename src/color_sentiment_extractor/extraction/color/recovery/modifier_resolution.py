# Chatbot/extractors/color/utils/modifier_resolution.py

"""
modifier_resolution.py
=======================

Handles the resolution of descriptive modifier tokens in color phrases.
Covers direct match, suffix stripping, compound fallback, and fuzzy logic.
Designed to support modular, multi-step modifier normalization across domains.
"""
from typing import Set
from color_sentiment_extractor.extraction.color.constants import BLOCKED_TOKENS, RECOVER_BASE_OVERRIDES, SEMANTIC_CONFLICTS
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base
from color_sentiment_extractor.extraction.general.token.normalize import singularize, normalize_token
from color_sentiment_extractor.extraction.color.vocab import known_tones as KNOWN_TONES



# =============================================================================
# 1. TONE AND COLOR VALIDATION HELPERS
# =============================================================================

def is_known_tone(word: str, known_tones: set, all_webcolor_names: set) -> bool:
    """
    Does: Checks whether a normalized token is a recognized tone or a standard web color.
    Returns: True if the token is in either the known tone set or web color set.
    """
    norm = normalize_token(word, keep_hyphens=True)
    return norm in known_tones or norm in all_webcolor_names


def is_valid_tone(phrase: str, known_tones, debug=True) -> bool:
    """
    Does: Validates whether a phrase resolves to a known tone using normalization,
    suffix fallback, and heuristic '-y' stripping with fuzzy fallback support.

    Returns: True if a valid tone is recognized through any method, else False.
    """
    norm = normalize_token(phrase, keep_hyphens=True)

    if norm in known_tones:
        return True

    # Délégation unique à recover_base (avec fuzzy autorisée)
    base = recover_base(
        norm,
        allow_fuzzy=True,
        known_modifiers=set(),
        known_tones=known_tones,
        debug=debug,
    )
    return bool(base and base in known_tones)



# =============================================================================
# 2. MODIFIER TOKEN RESOLUTION HELPERS
# =============================================================================

from color_sentiment_extractor.extraction.color.vocab import known_tones as KNOWN_TONES  # keep this import at top

def match_direct_modifier(token: str, known_modifiers: set, known_tones: set | None = None, debug: bool = True) -> str | None:
    """
    Does: Resolves a token to a known modifier using recover_base(), singularization,
    and compound fallback logic.
    Returns: A matching modifier or None if no match is found.
    """
    raw = token
    token = token.strip().lower().replace("-", " ").strip()

    # Step 1: Direct match
    if token in known_modifiers:
        return token

    # Step 2: Use shared suffix recovery logic (pass tones fixture or fallback)
    base = recover_base(
        token,
        allow_fuzzy=True,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or KNOWN_TONES),
        debug=True,
    )
    if base:
        # 2a) exact base is a known modifier
        if base in known_modifiers:
            if debug:
                print(f"[BASE MATCH] '{raw}' → '{base}' (in modifiers)")
            return base

        # 2b) NEW: if base ends with 'y' and its root is a known modifier, prefer the root (glossy → gloss)
        if base.endswith("y"):
            root = base[:-1]
            if root in known_modifiers:
                if debug:
                    print(f"[Y→ROOT MATCH] '{raw}' → '{base}' → '{root}' (in modifiers)")
                return root

        # ⏪ Extra logic for chained overrides like 'rosier' → 'rosy' → 'rose'
        if token.endswith("ier"):
            y_form = token[:-3] + "y"
            if y_form in RECOVER_BASE_OVERRIDES:
                override = RECOVER_BASE_OVERRIDES[y_form]
                if override in known_modifiers:
                    if debug:
                        print(f"[IER → Y → OVERRIDE] '{token}' → '{y_form}' → '{override}'")
                    return override

    # Step 3: Singularize
    singular = singularize(token)
    if singular in known_modifiers:
        if debug:
            print(f"[SINGULAR MATCH] '{raw}' → '{singular}'")
        return singular

    # Step 4: Compound fallback
    if " " in token:
        for part in token.split():
            if part in known_modifiers:
                if debug:
                    print(f"[COMPOUND MATCH] '{raw}' → '{part}'")
                return part

    if debug:
        print(f"[NO MATCH] '{raw}' → no match in known_modifiers")
    return None


def match_suffix_fallback(token: str, known_modifiers: set, known_tones: set, debug: bool = True) -> str | None:
    """
    Does: Attempts to resolve a noisy or suffixed modifier token using recover_base(),
    including variants like 'smoky' → 'smoke'. Accepts space-separated forms too.
    Returns: A valid root (modifier or tone) if found, else None.
    """
    # Use the project-wide normalizer, then collapse spaces for suffix-style checks
    norm = normalize_token(token, keep_hyphens=True)
    raw = norm.lower()
    if debug:
        print(f"\n[🔍 SUFFIX FALLBACK] Token: '{token}' → Normalized: '{raw}'")

    # Handle spaced variant like "soft y"
    collapsed = raw.replace(" ", "")
    if debug:
        print(f"[🔎 COLLAPSED] '{raw}' → '{collapsed}'")
        print(f"[🧾 CHECK MODIFIERS] {collapsed in known_modifiers}")
        print(f"[🧾 CHECK TONES]     {collapsed in known_tones if known_tones is not None else 'N/A'}")
    if collapsed in known_modifiers or (known_tones is not None and collapsed in known_tones):
        if debug:
            print(f"[✅ COLLAPSED MATCH] Returning: '{collapsed}'")
        return collapsed

    # Unified recovery (keyword args — important!)
    base = recover_base(
        raw,
        allow_fuzzy=True,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        debug=debug,
    )
    if debug:
        print(f"[📌 FINAL BASE] '{token}' → '{base}'")
        print(f"[🧾 IN MODIFIERS?] {base in known_modifiers if base else 'N/A'}")
        print(f"[🧾 IN TONES?]     {base in known_tones if (base and known_tones is not None) else 'N/A'}")

    if base and (base in known_modifiers or (known_tones is not None and base in known_tones)):
        if debug:
            print(f"[✅ VALID BASE] Returning: '{base}'")
        return base

    if debug:
        print(f"[❌ NO VALID MATCH] Returning: None")
    return None

def recover_y_with_fallback(token: str, known_modifiers: set, known_tones: set, debug: bool = False) -> str | None:
    """
    Does: Resolve '-y' (and friends) to a canonical base via the unified recover_base(), with fuzzy allowed.
    Returns: base if it’s a known modifier/tone, else None.
    """
    norm = normalize_token(token, keep_hyphens=True)
    base = recover_base(
        norm,
        allow_fuzzy=True,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        debug=debug,
    )
    return base if base and (base in known_modifiers or base in known_tones) else None


# =============================================================================
# 3. MODIFIER RESOLUTION ORCHESTRATOR
# =============================================================================

def resolve_modifier_token(
    raw_token: str,
    known_modifiers: set,
    known_tones: set | None = None,
    allow_fuzzy: bool = True,
    is_tone: bool = False,
    debug: bool = False
) -> str | None:
    """
    Résout un modificateur via la source unique recover_base (avec fuzzy optionnelle).
    Si l'entrée est déjà un ton connu, on la renvoie telle quelle (compat).
    """
    if not raw_token:
        return None

    token = normalize_token(raw_token, keep_hyphens=True)

    # Compat : si c’est déjà un ton connu, garder l’ancien comportement
    if known_tones and token in known_tones:
        if debug:
            print(f"[🎯 KNOWN TONE SHORTCUT] '{raw_token}' est un ton → retour tel quel")
        return token

    base = recover_base(
        token,
        allow_fuzzy=allow_fuzzy,
        known_modifiers=known_modifiers,
        known_tones=(known_tones or KNOWN_TONES),
        debug=debug,
    )

    return base if (base in known_modifiers) else None


# =============================================================================
# 4. MODIFIER CONFLICT & FILTER HELPERS
# =============================================================================

def should_suppress_compound(mod: str, tone: str) -> bool:
    """
    Does: Returns True if mod and tone are semantically redundant based on
    equality or prefix containment (e.g., 'soft soft-pink').
    Returns: True if modifier-tune pair should be suppressed.
    """
    return mod == tone or tone.startswith(mod) or mod.startswith(tone)

def is_blocked_modifier_tone_pair(
    modifier: str,
    tone: str,
    blocked_pairs: Set[tuple[str, str]] = BLOCKED_TOKENS
) -> bool:
    """
    Does: Checks whether a modifier-tone pair is explicitly blocked using a domain-specific blocklist.
    Returns: True if (modifier, tone) or (tone, modifier) is in the blocked list.
    """
    pair = (normalize_token(modifier, keep_hyphens=True), normalize_token(tone, keep_hyphens=True))
    reverse = (normalize_token(tone, keep_hyphens=True), normalize_token(modifier, keep_hyphens=True))
    return pair in blocked_pairs or reverse in blocked_pairs

def is_modifier_compound_conflict(expression: str, modifier_tokens: Set[str]) -> bool:
    """
    Does: Determines whether the expression token semantically overlaps with known modifiers
    by resolving the expression and checking against the modifier token set.
    Returns: True if resolved form is in the known modifier set.
    """
    resolved = resolve_modifier_token(
        expression,
        modifier_tokens,
        known_tones=set(),  # instead of None
        allow_fuzzy=True,
        is_tone=False
    )

    return resolved in modifier_tokens


def resolve_fallback_tokens(
    tokens,
    known_modifiers: Set[str],
    known_tones: Set[str],
    debug: bool = False
) -> Set[str]:
    """
    Does: Recovers missed tone or modifier tokens after extraction using direct match or modifier resolution.
    Returns: Set of resolved modifier or tone tokens.
    """
    resolved = set()

    for tok in tokens:
        raw = normalize_token(tok.text, keep_hyphens=True)

        if raw in known_tones:
            resolved.add(raw)
            continue

        mod = resolve_modifier_token(raw, known_modifiers, known_tones)
        if mod:
            resolved.add(mod)
            if debug:
                print(f"[🧪 FALLBACK TOKEN] '{raw}' → '{mod}'")

    return resolved
