# tests/test_color_logic_pipelines.py
import sys
import types
import importlib
import pytest

# -----------------------------------------------------------------------------
# 0) Patch minimal utils + expression stubs AVANT d'importer les pipelines
# -----------------------------------------------------------------------------
# Stub _try_simplified_match (appelé par rgb_pipeline → utils)
try:
    import color_sentiment_extractor.extraction.color.utils as color_utils
except Exception:
    pytest.skip("Package color_sentiment_extractor introuvable dans le PYTHONPATH.")

if not hasattr(color_utils, "_try_simplified_match"):
    def _try_simplified_match_stub(phrase: str, debug: bool = False):
        return None
    color_utils._try_simplified_match = _try_simplified_match_stub  # type: ignore
importlib.reload(color_utils)

# Stub complet pour general.expression.expression_helpers (évite fuzzy.match_expression_aliases)
expr_helpers_name = "color_sentiment_extractor.extraction.general.expression.expression_helpers"
if expr_helpers_name not in sys.modules:
    expr_helpers_mod = types.ModuleType(expr_helpers_name)

    def map_expressions_to_tones(*args, **kwargs):
        # return empty mapping {expression: set(tones)}
        return {}

    def get_matching_expression_tags_cached(*args, **kwargs):
        # return empty set of tags
        return set()

    def get_all_trigger_tokens(*args, **kwargs):
        # return empty set
        return set()

    def get_glued_token_vocabulary(*args, **kwargs):
        # return empty set
        return set()

    def _inject_expression_modifiers(*args, **kwargs):
        # no-op injection in sets passed by caller
        return None

    # Export minimal API
    expr_helpers_mod.map_expressions_to_tones = map_expressions_to_tones
    expr_helpers_mod.get_matching_expression_tags_cached = get_matching_expression_tags_cached
    expr_helpers_mod.get_all_trigger_tokens = get_all_trigger_tokens
    expr_helpers_mod.get_glued_token_vocabulary = get_glued_token_vocabulary
    expr_helpers_mod._inject_expression_modifiers = _inject_expression_modifiers

    sys.modules[expr_helpers_name] = expr_helpers_mod

# Ensure the package module re-exports those names (matches your __init__.py)
expr_pkg_name = "color_sentiment_extractor.extraction.general.expression"
if expr_pkg_name not in sys.modules:
    expr_pkg = types.ModuleType(expr_pkg_name)
    helpers = sys.modules[expr_helpers_name]
    for n in (
        "map_expressions_to_tones",
        "get_matching_expression_tags_cached",
        "get_all_trigger_tokens",
        "get_glued_token_vocabulary",
        "_inject_expression_modifiers",
    ):
        setattr(expr_pkg, n, getattr(helpers, n))
    sys.modules[expr_pkg_name] = expr_pkg

# -----------------------------------------------------------------------------
# 1) Imports SUT (maintenant safe)
# -----------------------------------------------------------------------------
from color_sentiment_extractor.extraction.color.logic.pipelines import phrase_pipeline as pp
from color_sentiment_extractor.extraction.color.logic.pipelines import rgb_pipeline as rp

# -----------------------------------------------------------------------------
# 2) Helpers pour charger les vrais vocabs (sinon skip)
# -----------------------------------------------------------------------------
def _load_known_sets():
    km = kt = None
    names = set()
    expr = {}
    try:
        from color_sentiment_extractor.extraction.color.constants import (
            KNOWN_MODIFIERS, KNOWN_TONES, WEBCOLOR_NAMES
        )
        km, kt, names = set(KNOWN_MODIFIERS), set(KNOWN_TONES), set(WEBCOLOR_NAMES)
    except Exception:
        pass

    if km is None or kt is None:
        try:
            from color_sentiment_extractor.extraction.color import (
                known_modifiers as _km, known_tones as _kt, all_webcolor_names as _names
            )
            km = set(_km() if callable(_km) else _km)
            kt = set(_kt() if callable(_kt) else _kt)
            names = set(_names() if callable(_names) else _names)
        except Exception:
            pass

    # expressions optionnel – OK si vide
    try:
        from color_sentiment_extractor.extraction.general.expression import (
            get_expression_map as _get_expr_map
        )
        expr = _get_expr_map() if callable(_get_expr_map) else dict(_get_expr_map)
    except Exception:
        expr = {}

    if not km or not kt:
        pytest.skip("Impossible de charger known_modifiers/known_tones depuis le projet.")
    return km, kt, names, expr

# -----------------------------------------------------------------------------
# 3) Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nlp_mock():
    # évite d'exiger en_core_web_sm
    import spacy
    return spacy.blank("en")

@pytest.fixture(scope="module")
def data():
    return _load_known_sets()

# -----------------------------------------------------------------------------
# 4) Tests phrase_pipeline
# -----------------------------------------------------------------------------
def test_extract_all_descriptive_color_phrases_basic(nlp_mock, data, monkeypatch):
    km, kt, names, expr = data
    monkeypatch.setattr(pp, "get_nlp", lambda: nlp_mock)

    mod, tone = sorted(km)[0], sorted(kt)[0]
    txt = f"{mod} {tone} and also {tone}"

    res = pp.extract_all_descriptive_color_phrases(
        text=txt,
        known_tones=kt,
        known_modifiers=km,
        all_webcolor_names=names,
        expression_map=expr,
        nlp=nlp_mock,
        debug=False,
    )

    assert all(r == r.lower() for r in res)
    assert sorted(res) == res
    assert any(tone in r for r in res)

def test_extract_phrases_from_segment_filters_cosmetic(nlp_mock, data, monkeypatch):
    km, kt, names, expr = data
    monkeypatch.setattr(pp, "get_nlp", lambda: nlp_mock)

    from color_sentiment_extractor.extraction.color import COSMETIC_NOUNS
    tone = next(iter(kt))
    mod = next(iter(km))
    cosmetic = next(iter(COSMETIC_NOUNS))
    seg = f"{mod} {tone} {cosmetic}. also {mod} {tone}"

    got = pp.extract_phrases_from_segment(
        segment=seg,
        known_modifiers=km,
        known_tones=kt,
        all_webcolor_names=names,
        expression_map=expr,
        llm_client=None,
        nlp=nlp_mock,
        debug=False,
    )

    assert all(p.split()[-1].lower() not in {c.lower() for c in COSMETIC_NOUNS} for p in got)

def test_extract_phrases_from_segment_removes_redundant_singletons(nlp_mock, data, monkeypatch):
    km, kt, names, expr = data
    monkeypatch.setattr(pp, "get_nlp", lambda: nlp_mock)

    mod, tone = sorted(km)[0], sorted(kt)[0]
    segment = f"{mod} {tone} next to plain {tone}"

    got = pp.extract_phrases_from_segment(
        segment=segment,
        known_modifiers=km,
        known_tones=kt,
        all_webcolor_names=names,
        expression_map=expr,
        llm_client=None,
        nlp=nlp_mock,
        debug=False,
    )

    assert f"{mod} {tone}" in got
    assert tone not in got  # singleton supprimé s'il existe déjà dans un composé

def test_process_segment_colors_alignment(data):
    km, kt, names, expr = data
    mod, tone = sorted(km)[0], sorted(kt)[0]
    phrases = [f"{mod} {tone}", tone]

    simplified, rgbs = pp.process_segment_colors(
        color_phrases=phrases,
        known_modifiers=km,
        known_tones=kt,
        llm_client=None,
        cache=None,
        debug=False,
    )

    assert len(simplified) == len(rgbs) == len(phrases)
    assert all((r is None) or (isinstance(r, tuple) and len(r) == 3) for r in rgbs)

def test_aggregate_color_phrase_results_consistency(nlp_mock, data, monkeypatch):
    km, kt, names, expr = data
    monkeypatch.setattr(pp, "get_nlp", lambda: nlp_mock)

    mod, tone = sorted(km)[0], sorted(kt)[0]
    segs = [
        f"I prefer {mod} {tone} for winter outfits.",
        f"Sometimes just {tone} works.",
    ]

    tone_set, all_phrases, rgb_map = pp.aggregate_color_phrase_results(
        segments=segs,
        known_modifiers=km,
        known_tones=kt,
        all_webcolor_names=names,
        expression_map=expr,
        llm_client=None,
        cache=None,
        nlp=nlp_mock,
        debug=False,
    )

    assert tone_set
    assert set(rgb_map).issubset(tone_set)
    assert set(rgb_map).issubset(set(all_phrases))

# -----------------------------------------------------------------------------
# 5) Tests rgb_pipeline
# -----------------------------------------------------------------------------
def test_process_color_phrase_single_known_tone(data):
    km, kt = data[0], data[1]
    tone = next(iter(kt))

    simplified, rgb = rp.process_color_phrase(
        phrase=tone,
        known_modifiers=km,
        known_tones=kt,
        llm_client=None,
        cache=None,
        debug=False,
    )

    assert simplified == tone
    assert (rgb is None) or (isinstance(rgb, tuple) and len(rgb) == 3)

def test_process_color_phrase_mod_tone_pair(data):
    km, kt = data[0], data[1]
    mod, tone = sorted(km)[0], sorted(kt)[0]

    simplified, rgb = rp.process_color_phrase(
        phrase=f"{mod} {tone}",
        known_modifiers=km,
        known_tones=kt,
        llm_client=None,
        cache=None,
        debug=False,
    )

    parts = simplified.split()
    assert len(parts) in (1, 2)
    if len(parts) == 2:
        assert parts[0] in km and parts[1] in kt
    assert (rgb is None) or (isinstance(rgb, tuple) and len(rgb) == 3)
