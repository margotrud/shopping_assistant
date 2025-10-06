# tests/test_general_token_suffix.py
from __future__ import annotations

"""
Tests for extraction/general/token/suffix/{recovery,registry}.py

Does:
  - Patch constants/helpers to keep suffix logic deterministic.
  - Unit-test every recover_* function with parametrized cases.
  - Validate registry dispatch (longest-suffix-first).
  - Validate is_suffix_variant() semantics.
  - Smoke-test build_augmented_suffix_vocab() with a controlled vocab.
"""

import sys
import types
import pytest

# Modules under test
from color_sentiment_extractor.extraction.general.token.suffix import recovery as R
from color_sentiment_extractor.extraction.general.token.suffix import registry as REG


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: Known vocab & deterministic patches
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def known_sets():
    """
    Does: Provide small, realistic sets of known modifiers/tones and webcolor names.
    Returns: dict with 'mods', 'tones', 'web'.
    NOTE: We keep ONLY bases here (no derived forms like creamy/bronzey),
          so is_suffix_variant() can return True on derived tokens.
    """
    mods = {
        # bases only
        "cream", "dust", "gloss", "matte", "soft", "warm", "glow", "gold",
        "tap", "pale", "mute",            "dark", "green", "ivory", "red",
        "shine", "bronze", "beige",

        # comparatives / misc present as standalone tokens
        "trendy", "fancy", "try",
    }
    tones = {"rose", "pink", "navy", "ivory", "green", "beige", "bronze"}
    web = {"navy", "ivory", "beige", "bronze", "rose", "green"}
    return {"mods": frozenset(mods), "tones": frozenset(tones), "web": frozenset(web)}


@pytest.fixture(autouse=True)
def patch_constants_and_helpers(monkeypatch, known_sets):
    """
    Does: Make recovery.py fully deterministic by patching:
          - constants (allowlists/overrides/non-suffixable),
          - helper builders (build_y_variant/build_ey_variant/is_cvc_ending),
          - recover_base import site used inside build_augmented_suffix_vocab().
    """
    # --- Patch constants on recovery module
    monkeypatch.setattr(R, "RECOVER_BASE_OVERRIDES", {
        "shiny": "shine",
        "rosy": "rose",
    }, raising=True)

    monkeypatch.setattr(R, "NON_SUFFIXABLE_MODIFIERS", frozenset({
        # keep example; nothing under test is blocked
        "ultra",
    }), raising=True)

    # Allow -ed generation for bases listed here (plus other standard rules)
    monkeypatch.setattr(R, "ED_SUFFIX_ALLOWLIST", frozenset({"warm"}), raising=True)

    # Also used as conservative allowlist of acceptable roots for -ey
    monkeypatch.setattr(R, "Y_SUFFIX_ALLOWLIST", frozenset({"bronze", "beige"}), raising=True)

    # --- Patch builder helpers imported at module load in recovery.py
    def _build_y_variant(base: str, debug: bool = False) -> str | None:
        # Simple, deterministic: drop trailing 'e' → 'y', else add 'y'
        return (base[:-1] + "y") if base.endswith("e") else (base + "y")

    def _build_ey_variant(base: str, raw: str, debug: bool = False) -> str | None:
        # Only bronze/beige gain an -ey form generatively
        if base in {"bronze", "beige"}:
            return base[:-1] + "ey"
        return None

    def _is_cvc_ending(s: str) -> bool:
        # Minimal CVC heuristic (no w/x/y as final consonant)
        if len(s) < 3:
            return False
        c1, v, c2 = s[-3], s[-2], s[-1]
        vowels = "aeiou"
        return (c1 not in vowels) and (v in vowels) and (c2 not in vowels) and (c2 not in "wxy")

    monkeypatch.setattr(R, "build_y_variant", _build_y_variant, raising=True)
    monkeypatch.setattr(R, "build_ey_variant", _build_ey_variant, raising=True)
    monkeypatch.setattr(R, "is_cvc_ending", _is_cvc_ending, raising=True)

    # --- Patch the imported recover_base symbol used *inside* build_augmented_suffix_vocab()
    dummy_module_name = "color_sentiment_extractor.extraction.general.token.base_recovery"

    def _dummy_recover_base(token: str, *, known_modifiers=None, known_tones=None,
                            allow_fuzzy=False, fuzzy_fallback=False, **_):
        """Route to registry-driven suffix recovery, else identity if known."""
        mods = set(known_modifiers) if known_modifiers else set(known_sets["mods"])
        tones = set(known_tones) if known_tones else set(known_sets["tones"])
        if token in mods or token in tones:
            return token
        base = REG.recover_with_registry(token, mods, tones, debug=False)
        return base or token

    dummy_module = types.SimpleNamespace(recover_base=_dummy_recover_base)
    sys.modules[dummy_module_name] = dummy_module


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for individual recovery helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("creamy", "cream"),
        ("dusty", "dust"),
        ("shiny", "shine"),   # override path
        ("glossy", "gloss"),  # double-consonant collapse
        ("rosy", "rose"),     # override path
    ],
)
def test_recover_y(token, expected, known_sets):
    out = R.recover_y(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("paled", "pale"),     # e-restore
        ("tapped", "tap"),     # CVC collapse
        ("warmed", "warm"),    # allowlist fallback to base
        ("muted", "mute"),     # base candidate last
    ],
)
def test_recover_ed(token, expected, known_sets):
    out = R.recover_ed(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


def test_recover_ing(known_sets):
    assert R.recover_ing("glowing", set(known_sets["mods"]), set(known_sets["tones"])) == "glow"


@pytest.mark.parametrize("token,expected", [("tried", "try")])
def test_recover_ied(token, expected, known_sets):
    out = R.recover_ied(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("trendier", "trendy"),
        ("fancier", "fancy"),
    ],
)
def test_recover_ier(token, expected, known_sets):
    out = R.recover_ier(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("greenish", "green"),
        ("ivorish", "ivory"),  # via +y or recover_base chain
    ],
)
def test_recover_ish(token, expected, known_sets):
    out = R.recover_ish(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("softness", "soft"),
        ("creaminess", "cream"),  # i-drop → cream
    ],
)
def test_recover_ness(token, expected, known_sets):
    out = R.recover_ness(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


def test_recover_ly(known_sets):
    assert R.recover_ly("softly", set(known_sets["mods"]), set(known_sets["tones"])) == "soft"
    assert R.recover_ly("warmly", set(known_sets["mods"]), set(known_sets["tones"])) == "warm"


def test_recover_en(known_sets):
    assert R.recover_en("golden", set(known_sets["mods"]), set(known_sets["tones"])) == "gold"


@pytest.mark.parametrize(
    "token,expected",
    [
        ("bronzey", "bronze"),
        ("beigey", "beige"),
        ("pinkey", None),  # blocked by allowlist
    ],
)
def test_recover_ey(token, expected, known_sets):
    out = R.recover_ey(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert out == expected


def test_recover_ee_to_y(known_sets):
    assert R.recover_ee_to_y("ivoree", set(known_sets["mods"]), set(known_sets["tones"])) == "ivory"


# ──────────────────────────────────────────────────────────────────────────────
# Registry dispatch & variant predicate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("trendier", "trendy"),
        ("tried", "try"),
        ("bronzey", "bronze"),
        ("ivoree", "ivory"),
        ("glowing", "glow"),
        ("paled", "pale"),
        ("greenish", "green"),
        ("golden", "gold"),
        ("softness", "soft"),
        ("warmly", "warm"),
        ("darker", "dark"),
        ("creamy", "cream"),
    ],
)
def test_registry_recover_with_registry(token, expected, known_sets):
    base = REG.recover_with_registry(token, set(known_sets["mods"]), set(known_sets["tones"]))
    assert base == expected


def test_is_suffix_variant_true_and_false(known_sets):
    mods, tones = known_sets["mods"], known_sets["tones"]
    KMODS, KTONES = frozenset(mods), frozenset(tones)

    # True: derived form whose base is known (derived forms not in mods)
    assert R.is_suffix_variant("creamy", KMODS, KTONES) is True
    assert R.is_suffix_variant("bronzey", KMODS, KTONES) is True
    assert R.is_suffix_variant("paled", KMODS, KTONES) is True

    # False: already-known token that is not an override (not a "variant")
    assert R.is_suffix_variant("cream", KMODS, KTONES) is False
    assert R.is_suffix_variant("rose", KMODS, KTONES) is False

    # False: suffix not handled
    assert R.is_suffix_variant("creamiest", KMODS, KTONES) is False


# ──────────────────────────────────────────────────────────────────────────────
# Builder smoke test (deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def test_build_augmented_suffix_vocab_happy_path(monkeypatch, known_sets):
    """
    Does: Validate that the builder emits expected variants with our patched rules,
          while staying deterministic and conservative.
    """
    known_tokens = {"cream", "dust", "warm", "glow", "bronze", "beige", "pale", "tap"}
    # we force-include some derived forms to ensure the builder preserves them when appropriate
    known_modifiers = set(known_sets["mods"]) | {"bronzey", "beigey"}
    web = set(known_sets["web"])

    out = R.build_augmented_suffix_vocab(
        known_tokens=known_tokens,
        known_modifiers=known_modifiers,
        known_tones=set(known_sets["tones"]),
        webcolor_names=web,
        debug=False,
    )

    # Core bases present
    assert {"cream", "dust", "warm", "glow", "bronze", "beige", "pale", "tap"} <= out

    # -y generation (+ CVC doubling for e.g., mat → matty if present)
    assert "creamy" in out
    assert "dusty" in out

    # -ey only for allowlisted roots and present/allowed by our rules
    assert "bronzey" in out
    assert "beigey" in out

    # -ed cases: e-restore, CVC collapse, allowlist
    assert "paled" in out          # pale → paled
    assert "tapped" in out         # tap → tapped (CVC)
    assert "warmed" in out         # warm ∈ ED_SUFFIX_ALLOWLIST (patched)
