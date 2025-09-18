# constants.py
# ============

# ─────────────────────────────────────────────
# 🎯 Semantic Conflict Rules
# ─────────────────────────────────────────────

SEMANTIC_CONFLICTS = {
    frozenset({"white", "offwhite"}),
    frozenset({"cool", "coal"}),
    frozenset({"soft glam", "soft glow"}),
    frozenset({"blurple", "pale"}),
    frozenset({"clay", "classy"}),
    frozenset({"airy", "fairy"}),
    frozenset({"silly", "silk"}),
}

BLOCKED_TOKENS = {
    ("light", "night"),
    ("romantic", "dramatic"),
    ("off blue", "white"),
    ("tint", "mint"),
    ("liner", "linen")
}

EXPRESSION_SUPPRESSION_RULES = {
    "glamorous": {"natural", "daytime"},
    "edgy": {"romantic", "soft glam"},
    "evening": {"daytime"},
    "bold": {"subtle", "neutral"},
    "soft glam": {"glamorous"},
    # Extend as needed
}

# ─────────────────────────────────────────────
# 🔤 Suffix Recovery Overrides
# ─────────────────────────────────────────────

Y_SUFFIX_ALLOWLIST = {"beige", "bronze", "dewy", "rose", "shine"}
ED_SUFFIX_ALLOWLIST = {"golden"}
NON_SUFFIXABLE_MODIFIERS = {"metallic"}

RECOVER_BASE_OVERRIDES = {
    "icy": "ice",
    "shiny": "shine",
    "rosy": "rose",
}

Y_SUFFIX_OVERRIDE_FORMS = {
    "rose": "rosy",
    "shine": "shiny",
}


COSMETIC_NOUNS = {
    "blush", "foundation", "lipstick", "concealer",
    "bronzer", "highlighter", "mascara", "eyeliner", "tone", "shades", "foundation"
}
