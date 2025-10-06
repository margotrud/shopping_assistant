# tests/test_general_token_split_core.py
from __future__ import annotations
import time
import types
import pytest

# Module under test
from color_sentiment_extractor.extraction.general.token.split import split_core as S

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: patch normalizer to be deterministic, lightweight, and similar to prod
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_normalize_token(monkeypatch):
    """
    Does: Patch `normalize_token` with lowercase, hyphen unification,
          removal of fancy quotes/apostrophes, and hyphen→space when requested.
    """
    def _normalize_token(s: str, keep_hyphens: bool = False) -> str:
        s = (s or "").lower().strip()
        # Unify underscores → hyphens
        s = s.replace("_", "-")
        # Unify common Unicode hyphens to ASCII hyphen
        hyphens = "\u2010\u2011\u2012\u2013\u2014\u2212"  # ‐ - ‒ – — −
        for ch in hyphens:
            s = s.replace(ch, "-")
        # Remove fancy quotes and plain apostrophe entirely
        quotes = "\u2018\u2019\u201B\u2032\u02BC"  # ‘ ’ ‛ ′ ʼ
        for ch in quotes + "'":
            s = s.replace(ch, "")
        return s if keep_hyphens else s.replace("-", " ")
    monkeypatch.setattr(S, "normalize_token", _normalize_token, raising=True)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: has_token_overlap
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Dusty-Rose", "soft dusty rose glow", True),      # shared token "dusty" and "rose"
        ("  Navy  Blue ", "navy-blue-satin", True),        # whitespace/hyphen normalization
        ("gloss", "matte", False),                         # no shared tokens
        ("", "anything", False),                           # empty safely handled
        ("‘rose’", "rose-gold", True),                     # punctuation normalized by normalizer
    ],
)
def test_has_token_overlap(a, b, expected):
    assert S.has_token_overlap(a, b) is expected


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for vocab construction in fallback_split tests
# NOTE: we pass raw vocab (non-normalized); split_core handles normalization.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_vocab():
    # Typical color/modifier-ish tokens (short + long for greedy/backtrack)
    return {
        "dusty", "rose", "soft", "glow", "navy", "blue",
        "barely", "there", "barely-there",  # check preference for real splits
        "ultra", "matte", "shimmer", "gloss", "off", "white",
        "acid", "green",
    }


# ─────────────────────────────────────────────────────────────────────────────
# fallback_split_on_longest_substring: core behaviors (A..E) + budget guards
# ─────────────────────────────────────────────────────────────────────────────

def test_split_prefers_two_way_when_exact_two_parts_exist(small_vocab):
    # Path A: 2-way split exact
    out = S.fallback_split_on_longest_substring(
        token="dustyrose",
        vocab=small_vocab,
        prefer_split_over_glued=True,
        time_budget_sec=0.2,
    )
    assert out == ["dusty", "rose"]


def test_split_greedy_longest_prefix_when_three_parts_exist(small_vocab):
    # Path B: greedy segmentation builds multiple parts
    out = S.fallback_split_on_longest_substring(
        token="softdustyrose",
        vocab=small_vocab,
        time_budget_sec=0.2,
    )
    # Greedy should yield valid covering segmentation with ≥2 parts
    assert out == ["soft", "dusty", "rose"]


def test_split_backtracking_when_greedy_cant_cover_cleanly(small_vocab, monkeypatch):
    # Make greedy fail by disallowing parts shorter than 5 to force backtracking
    out = S.fallback_split_on_longest_substring(
        token="ultramattegloss",
        vocab=small_vocab,
        min_part_len=5,          # "ultra"(5), "matte"(5), "gloss"(5) → only BT can assemble them
        time_budget_sec=0.2,
    )
    assert out == ["ultra", "matte", "gloss"]


def test_split_exact_match_when_no_real_split_found(small_vocab):
    # Path D: exact match returned (single token in vocab)
    out = S.fallback_split_on_longest_substring(
        token="shimmer",  # in vocab as single token
        vocab=small_vocab,
        time_budget_sec=0.2,
    )
    assert out == ["shimmer"]


def test_split_longest_substring_fallback_when_partial_found(small_vocab):
    # Path E: longest substring inside the token; sides may be OOV
    # "navyblue" is not in vocab as a single token; "navy" and "blue" are.
    out = S.fallback_split_on_longest_substring(
        token="xxnavyblueyy",
        vocab=small_vocab,
        time_budget_sec=0.2,
    )
    # Should return sides + found substring in order under max_parts
    assert out == ["xx", "navyblue", "yy"] or out == ["xxnavy", "blue", "yy"] or out == ["xx", "navy", "blueyy"]
    # The function’s E-path returns one found substring occurrence; accept any consistent split variant.


def test_trigram_guard_filters_hopeless_inputs(small_vocab):
    # Token shares no trigram with vocab; should early-return []
    out = S.fallback_split_on_longest_substring(
        token="zzqzzqzzq",  # no trigram overlap expected
        vocab=small_vocab,
        time_budget_sec=0.2,
    )
    assert out == []


def test_split_respects_time_budget(monkeypatch, small_vocab):
    # Force le tout premier over_budget() à déclencher: t0=0.0 puis 1.0 ensuite
    seq = [0.0] + [1.0] * 100  # suffisamment d'appels pour couvrir tout le flot

    def fake_monotonic():
        return seq.pop(0) if seq else 1.0

    # ⚠️ Patch sur l’alias du module: S._time.monotonic (et pas time.monotonic)
    monkeypatch.setattr(S._time, "monotonic", fake_monotonic, raising=True)

    out = S.fallback_split_on_longest_substring(
        token="softdustyroseglossultramatte",
        vocab=small_vocab,
        time_budget_sec=0.0,
    )
    assert out == []



def test_prefer_split_over_glued_true_returns_real_split(small_vocab):
    # When both "barely-there" and {"barely","there"} exist, prefer the true split
    out = S.fallback_split_on_longest_substring(
        token="barelythere",
        vocab=small_vocab,
        prefer_split_over_glued=True,
        time_budget_sec=0.2,
    )
    assert out == ["barely", "there"]


# ─────────────────────────────────────────────────────────────────────────────
# recursive_token_split: validator-only splitting
# ─────────────────────────────────────────────────────────────────────────────

def test_recursive_token_split_binary_and_prefix_suffix_paths():
    # Build a validator that recognizes these normalized parts only
    valid_parts = {"soft", "dusty", "rose", "navy", "blue", "ultra", "matte"}

    def _validator(t: str) -> bool:
        return t in valid_parts

    # A) Simple binary split coverage
    out_a = S.recursive_token_split("softdusty", _validator)
    assert out_a == ["soft", "dusty"]

    # B) Three-part via nested splits
    out_b = S.recursive_token_split("softdustyrose", _validator)
    assert out_b == ["soft", "dusty", "rose"]

    # C) Prefix/suffix fallback: "ultramatteblue" → ["ultra","matte","blue"]
    out_c = S.recursive_token_split("ultramatteblue", _validator)
    assert out_c == ["ultra", "matte", "blue"]

    # D) Un-splittable returns None
    assert S.recursive_token_split("unknownchunk", _validator) is None
