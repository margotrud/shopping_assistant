# tests/test_color_suffix_rules.py
from __future__ import annotations

import pytest

# Module sous test
from color_sentiment_extractor.extraction.color.suffix import rules as r


@pytest.fixture(autouse=True)
def patch_constants(monkeypatch):
    """
    Does: Provide deterministic constants to the rules module for all tests.
    Returns: None (patches in-place on import'ed module).
    """
    monkeypatch.setattr(r, "Y_SUFFIX_ALLOWLIST", frozenset({"beige", "gloss"}), raising=True)
    monkeypatch.setattr(r, "Y_SUFFIX_OVERRIDE_FORMS", {"taupe": "taupey"}, raising=True)
    monkeypatch.setattr(
        r,
        "RECOVER_BASE_OVERRIDES",
        {
            "flashy": "flash",
            "rosy": "rose",
            "inked": "ink",
        },
        raising=True,
    )
    # EY allowlist is optional in project; we control it here
    monkeypatch.setattr(r, "EY_SUFFIX_ALLOWLIST", frozenset({"beige"}), raising=True)


# ────────────────────────────────────────────────────────────────────────────
# is_y_suffix_allowed
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "base,expected",
    [
        ("beige", True),     # allowlist wins even if endswith 'e'
        ("soft", True),      # heuristic: endswith 't'
        ("gloss", True),     # allowlist via patch + endswith 'ss'
        ("rose", False),     # endswith 'e' → blocked by rule (not in allowlist)
        ("icy", False),      # endswith 'y'
        ("a", False),        # too short
        ("cocoa", False),    # vowel ending
    ],
)
def test_is_y_suffix_allowed(base, expected):
    assert r.is_y_suffix_allowed(base) is expected


# ────────────────────────────────────────────────────────────────────────────
# is_cvc_ending
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "base,expected",
    [
        ("red", True),       # r-e-d fits CVC
        ("mint", False),     # last3 = i-n-t → first of last3 is vowel ⇒ False
        ("wax", False),      # final 'x' excluded
        ("mew", False),      # final 'w' excluded
        ("day", False),      # final 'y' excluded
        ("aa", False),       # too short
        ("peb", True),       # p-e-b fits CVC
    ],
)
def test_is_cvc_ending(base, expected):
    assert r.is_cvc_ending(base) is expected


# ────────────────────────────────────────────────────────────────────────────
# build_y_variant
# ────────────────────────────────────────────────────────────────────────────

def test_build_y_variant_uses_override_first(caplog):
    assert r.build_y_variant("taupe", debug=True) == "taupey"

@pytest.mark.parametrize(
    "base,expected",
    [
        ("beige", "beigey"),   # allowlist → even though base ends with 'e'
        ("soft", "softy"),     # heuristic allow
        ("rose", None),        # denied: not allowlisted and endswith 'e'
        ("icy", None),         # denied: endswith 'y'
    ],
)
def test_build_y_variant_rules(base, expected):
    assert r.build_y_variant(base) == expected


# ────────────────────────────────────────────────────────────────────────────
# build_ey_variant
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,base,expected",
    [
        ("beige", "beige", "beigey"),    # allowlist for -ey (removes trailing 'e')
        ("bronze", "bronze", "bronzey"), # strict rule: finals 'ze'
        ("fleece", "fleece", "fleecy"),  # should NOT produce '-ey'
    ],
)
def test_build_ey_variant_paths(raw, base, expected):
    out = r.build_ey_variant(base=base, raw=raw, debug=True)
    # For "fleece", -ey path must be refused (handled by -y elsewhere)
    if raw == "fleece":
        assert out is None
    else:
        assert out == expected


# ────────────────────────────────────────────────────────────────────────────
# _apply_reverse_override
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "base,token,expected",
    [
        ("flish", "flashy", "flash"),   # strips 'y' then maps via overrides
        ("rose", "rosy", "rose"),       # strips 'y' then maps to 'rose'
        ("inky", "inked", "ink"),       # strips 'ed' then maps
        ("mint", "minty", "mint"),      # no override hit → fallback to provided base
    ],
)
def test_apply_reverse_override(base, token, expected):
    assert r._apply_reverse_override(base=base, token=token, debug=True) == expected


# ────────────────────────────────────────────────────────────────────────────
# _collapse_repeated_consonant
# ────────────────────────────────────────────────────────────────────────────

def test_collapse_repeated_consonant_hits_known_sets(caplog):
    known_modifiers = {"slim", "dust"}
    known_tones = {"rose", "ink"}
    assert (
            r._collapse_repeated_consonant(
                "slimm",
                known_modifiers,
                known_tones,
                debug=True,
            )
            == "slim"
    )


def test_collapse_repeated_consonant_noop_when_unknown():
    known_modifiers: set[str] = set()
    known_tones: set[str] = {"rose"}
    # "matt" → "mat" not present in known sets, so keep original
    assert r._collapse_repeated_consonant("matt", known_modifiers, known_tones) == "matt"


# ────────────────────────────────────────────────────────────────────────────
# Precedence checks (-y vs -ey)
# ────────────────────────────────────────────────────────────────────────────

def test_y_vs_ey_precedence_beige():
    # both paths yield "beigey"
    assert r.build_y_variant("beige") == "beigey"
    assert r.build_ey_variant(base="beige", raw="beige") == "beigey"

def test_y_vs_ey_precedence_bronze():
    # -y denied (final 'e', not allowlisted); -ey allowed (final 'ze')
    assert r.build_y_variant("bronze") is None
    assert r.build_ey_variant(base="bronze", raw="bronze") == "bronzey"
