# tests/test_color_utils.py


from __future__ import annotations

import importlib

import pytest

"""
rgb_distance tests
==================

Does: Validate sRGB/Lab distances, representative RGB selection, webcolor lookup/
      fuzzy name helpers, and robust RGB parsing with deterministic monkeypatches.
"""

# Importer explicitement l'objet module (évite les collisions de re-export)
rd = importlib.import_module("color_sentiment_extractor.extraction.color.utils.rgb_distance")


# ──────────────────────────────────────────────────────────────────────────────
# Global, deterministic monkeypatches
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def patch_webcolor_names_and_normalizer(monkeypatch):
    """Does: Provide a tiny, stable webcolor vocabulary and a lightweight normalizer."""
    vocab = ["red", "green", "acid green", "navy blue", "off white"]

    def _normalize_token(s: str, keep_hyphens: bool = False) -> str:
        s = s.lower().strip().replace("_", "-")
        return s if keep_hyphens else s.replace("-", " ")

    # Patch sur l'OBJET MODULE (noms réellement utilisés par rgb_distance.py)
    monkeypatch.setattr(rd, "get_all_webcolor_names", lambda: list(vocab), raising=True)
    monkeypatch.setattr(rd, "normalize_token", _normalize_token, raising=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core distance functions
# ──────────────────────────────────────────────────────────────────────────────
def test_rgb_distance_zero_and_max():
    assert rd.rgb_distance((0, 0, 0), (0, 0, 0)) == 0.0
    expected = (3 * (255 ** 2)) ** 0.5
    assert rd.rgb_distance((0, 0, 0), (255, 255, 255)) == pytest.approx(expected, rel=1e-9)


def test_rgb_distance_bounds_validation():
    with pytest.raises(ValueError):
        rd.rgb_distance((256, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        rd.rgb_distance((-1, 0, 0), (0, 0, 0))


def test_lab_distance_basic_properties():
    a = (120, 100, 90)
    b = (121, 99, 88)
    assert rd.lab_distance(a, a) == 0.0
    d1 = rd.lab_distance(a, b)
    d2 = rd.lab_distance(b, a)
    assert d1 > 0.0 and abs(d1 - d2) < 1e-9


def test_is_within_rgb_margin_true_false():
    x = (10, 10, 10)
    y = (20, 20, 20)
    assert rd.is_within_rgb_margin(x, y, margin=20.0) is True
    assert rd.is_within_rgb_margin(x, y, margin=5.0) is False


# ──────────────────────────────────────────────────────────────────────────────
# Representative color / clustering
# ──────────────────────────────────────────────────────────────────────────────
def test_choose_representative_rgb_picks_central_color_lab():
    palette = {"black": (0, 0, 0), "mid": (10, 10, 10), "white": (255, 255, 255)}
    assert rd.choose_representative_rgb(palette) == (10, 10, 10)


def test_choose_representative_rgb_empty_none():
    assert rd.choose_representative_rgb({}) is None


# ──────────────────────────────────────────────────────────────────────────────
# Named color lookups & fuzzy
# ──────────────────────────────────────────────────────────────────────────────
def test_find_similar_color_names_thresholding():
    base = (10, 10, 10)
    known = {"near": (12, 12, 12), "far": (200, 200, 200)}
    near = rd.find_similar_color_names(base, known, threshold=6.0, metric=rd.rgb_distance)
    assert near == ["near"]


def test_nearest_color_name_custom_map():
    base = (0, 10, 200)
    known = {"navy blue": (0, 0, 128), "red": (255, 0, 0)}
    assert rd.nearest_color_name(base, known_rgb_map=known) == "navy blue"


def test_fuzzy_match_rgb_from_known_colors_typo():
    assert rd.fuzzy_match_rgb_from_known_colors("grean", n=1, cutoff=0.6) == "green"


def test_fuzzy_match_rgb_from_known_colors_no_hit_returns_none():
    assert rd.fuzzy_match_rgb_from_known_colors("zzz-not-a-color", n=1, cutoff=0.9) is None


# ──────────────────────────────────────────────────────────────────────────────
# Exact CSS/XKCD lookups (lazy import) + caching sanity
# ──────────────────────────────────────────────────────────────────────────────
def test_try_simplified_match_css_xkcd_and_cache_identity():
    # Sauter proprement si les lib ne sont pas installées
    pytest.importorskip("matplotlib")
    pytest.importorskip("webcolors")

    rgb = rd._try_simplified_match("acid green")
    assert rgb is not None and all(0 <= v <= 255 for v in rgb)

    m1 = rd._get_named_color_map()
    m2 = rd._get_named_color_map()
    assert m1 is m2
    assert isinstance(m1, dict)
    assert "acid green" in m1


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helper
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,expect",
    [
        ("rgb(12, 34, 56)", (12, 34, 56)),
        ("(0,0,0)", (0, 0, 0)),
        ("[255, 128, 64]", (255, 128, 64)),
        ("values: 7, 8, 9 end", (7, 8, 9)),
    ],
)
def test_parse_rgb_tuple_variants_ok(text, expect):
    assert rd._parse_rgb_tuple(text) == expect


@pytest.mark.parametrize("text", ["(999, 0, 0)", "no numbers here", "(10, -1, 10)"])
def test_parse_rgb_tuple_invalid(text):
    assert rd._parse_rgb_tuple(text) is None
