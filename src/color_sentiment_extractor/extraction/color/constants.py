# constants.py
# ============

"""
constants.
=========

Does: Define global, immutable color-domain constants for extraction & recovery
      (conflicts, blocked pairs, suppression rules, suffix guidance, domain nouns).
Used By: Token/base recovery, suffix handling, alias validation, phrase extraction,
         sentiment/classification layers.
Returns: Pure data structures only (no side effects).

Notes importantes:
- Les allowlists/overrides sont volontariamente petites : exceptions réelles, pas de
logique générale.
"""

AUTONOMOUS_TONE_BAN: set[str] = {"dust", "glow"}  # disallow as standalone tones


# ── 1) Conflicts & blocking ──────────────────────────────────────────────────

# Symmetric conflicts between tokens/phrases (order does not matter)
SEMANTIC_CONFLICTS = frozenset(
    {
        frozenset({"white", "offwhite"}),
        frozenset({"cool", "coal"}),
        frozenset({"soft glam", "soft glow"}),
        frozenset({"blurple", "pale"}),
        frozenset({"clay", "classy"}),
        frozenset({"airy", "fairy"}),
        frozenset({"silly", "silk"}),
    }
)

# Ordered (raw, base) pairs that must never be accepted during recovery
BLOCKED_TOKENS = frozenset(
    {
        ("light", "night"),
        ("romantic", "dramatic"),
        ("off blue", "white"),
        ("tint", "mint"),
        ("liner", "linen"),
    }
)

# Suppress tags when others are present (left beats any on right)
EXPRESSION_SUPPRESSION_RULES = {
    "glamorous": {"natural", "daytime"},
    "edgy": {"romantic", "soft glam"},
    "evening": {"daytime"},
    "bold": {"subtle", "neutral"},
    "soft glam": {"glamorous"},
}


# ── 2) Suffix recovery guidance ──────────────────────────────────────────────
# Keep these small: they are for genuine exceptions, not general logic.

Y_SUFFIX_ALLOWLIST = frozenset({"beige", "bronze", "dewy", "rose", "shine"})
ED_SUFFIX_ALLOWLIST = frozenset({"golden"})
NON_SUFFIXABLE_MODIFIERS = frozenset({"metallic"})

# Direct base recovery overrides (input → canonical base)
RECOVER_BASE_OVERRIDES = {
    "icy": "ice",
    "shiny": "shine",
    "rosy": "rose",
}

# Canonical outward forms when generating -y variants (base → outward form)
Y_SUFFIX_OVERRIDE_FORMS = {
    "rose": "rosy",
    "shine": "shiny",
}


# ── 3) Domain nouns to filter from phrase extraction ─────────────────────────
COSMETIC_NOUNS = frozenset(
    {
        "blush",
        "foundation",
        "lipstick",
        "concealer",
        "bronzer",
        "highlighter",
        "mascara",
        "eyeliner",
        "tone",
        "shades",
    }
)
