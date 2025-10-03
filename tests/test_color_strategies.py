# tests/test_color_strategies.py
from __future__ import annotations
import pytest
import spacy

"""
Tests for color/strategies:
- compound.py : attempt_mod_tone_pair, extract_from_adjacent/split/glued, extract_compound_phrases
- standalone.py: extract_lone_tones, extract_standalone_phrases

Key detail: the strategies skip tokens whose POS tag isn't ADJ/NOUN/PROPN.
Since we use a spaCy blank model (no tagger), we create a tiny DummyToken (_DT)
with pos_="ADJ" for tests that need it (glued/split). This avoids model downloads.
"""

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def nlp():
    return spacy.blank("en")  # tokenization-only

@pytest.fixture
def known_sets():
    known_modifiers = {"soft", "dusty", "shiny", "barely-there", "pastel"}
    known_tones = {"rose", "pink", "lilac", "beige", "taupe", "purple"}
    all_webcolor_names = {"pink", "beige", "lilac", "rose", "taupe", "purple"}
    known_color_tokens = known_modifiers | known_tones
    return dict(
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        all_webcolor_names=all_webcolor_names,
        known_color_tokens=known_color_tokens,
    )

@pytest.fixture
def llm_client():
    class DummyLLM: ...
    return DummyLLM()

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_tokens(nlp, text: str):
    return list(nlp(text))

# Dummy token with a POS tag so extractors don’t skip it
class _DT:
    def __init__(self, text: str, pos_: str = "ADJ"):
        self.text = text
        self.pos_ = pos_

# ── Monkeypatch domain helpers so tests stay deterministic ────────────────────
def patch_recovery_and_token_helpers(monkeypatch, known_sets):
    import color_sentiment_extractor.extraction.color.strategies.compound as compound
    import color_sentiment_extractor.extraction.color.strategies.standalone as standalone

    def resolve_modifier_token(candidate, known_modifiers, known_tones=None, fuzzy=False, debug=False):
        if candidate in known_modifiers:
            return candidate
        if candidate.endswith("y") and candidate[:-1] in known_modifiers:
            return candidate[:-1]
        if candidate in known_modifiers:
            return candidate
        return None

    def is_known_tone(token, known_tones, all_webcolor_names):
        return token in known_tones or token in all_webcolor_names

    def match_suffix_fallback(tok, known_modifiers, known_tones):
        if tok in known_modifiers:
            return tok
        if tok + "y" in known_modifiers:
            return tok + "y"
        if tok.endswith("y") and tok[:-1] in known_modifiers:
            return tok
        return None

    def _attempt_simplify_token(candidate, known_modifiers, known_tones, llm_client, role="modifier", debug=False):
        return {"airy": "soft", "rosy": "rose", "pastelly": "pastel"}.get(candidate)

    def _extract_filtered_tokens(tokens, known_modifiers, known_tones, llm_client, debug=False):
        out = set()
        for t in tokens:
            raw = t.text.lower().replace("_", "-")
            if raw in {"lipstick", "mascara"}:
                continue
            if raw in known_modifiers or raw in known_tones:
                out.add(raw)
        return out

    def build_y_variant(base: str):
        return base if base.endswith("y") else base + "y"

    def split_glued_tokens(raw, known_color_tokens, known_modifiers, debug=False, vocab=None):
        if raw == "dustyrose":
            return ["dusty", "rose"]
        if raw == "pastellilacpink":
            return ["pastel", "lilac", "pink"]
        return []

    def split_tokens_to_parts(text, vocab, debug=False):
        if text == "softpink":
            return ["soft", "pink"]
        if text == "softlilacpink":
            return ["soft", "lilac", "pink"]
        return []

    # --- general.token.recover_base (signature-compatible stub) ---
    def recover_base(token, known_modifiers=None, known_tones=None, fuzzy_fallback=False):
        # trivial base recovery to support tests
        if known_modifiers is None:
            known_modifiers = set()
        if known_tones is None:
            known_tones = set()

        if token in known_modifiers or token in known_tones:
            return token
        if token.endswith("y") and token[:-1] in known_modifiers:
            return token[:-1]
        if token.endswith("ish") and token[:-3] in known_modifiers:
            return token[:-3]
        return None

    monkeypatch.setattr(compound, "recover_base", recover_base, raising=True)

    monkeypatch.setattr(compound, "resolve_modifier_token", resolve_modifier_token, raising=True)
    monkeypatch.setattr(compound, "is_known_tone", is_known_tone, raising=True)
    monkeypatch.setattr(compound, "match_suffix_fallback", match_suffix_fallback, raising=True)
    monkeypatch.setattr(compound, "_attempt_simplify_token", _attempt_simplify_token, raising=True)
    monkeypatch.setattr(compound, "build_y_variant", build_y_variant, raising=True)
    monkeypatch.setattr(compound, "split_glued_tokens", split_glued_tokens, raising=True)
    monkeypatch.setattr(compound, "split_tokens_to_parts", split_tokens_to_parts, raising=True)

    monkeypatch.setattr(standalone, "_extract_filtered_tokens", _extract_filtered_tokens, raising=True)

    def _inject_expression_modifiers(tokens, known_modifiers, known_tones, expression_map, debug=False):
        present = [t.text.lower() for t in tokens]
        out = []
        for alias in present:
            out.extend(expression_map.get(alias, []))
        return out

    monkeypatch.setattr(standalone, "_inject_expression_modifiers", _inject_expression_modifiers, raising=True)

# ── Tests: compound.py ────────────────────────────────────────────────────────
def test_attempt_mod_tone_pair_direct_hit(monkeypatch, known_sets, llm_client):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    compounds, raw = set(), []
    mod.attempt_mod_tone_pair(
        "soft", "pink",
        compounds, raw,
        known_sets["known_modifiers"],
        known_sets["known_tones"],
        known_sets["all_webcolor_names"],
        llm_client,
        debug=False,
    )

    assert "soft pink" in compounds
    assert ("soft", "pink") in raw

def test_attempt_mod_tone_pair_llm_fallback_for_modifier(monkeypatch, known_sets, llm_client):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    compounds, raw = set(), []
    mod.attempt_mod_tone_pair(
        "airy", "lilac",
        compounds, raw,
        known_sets["known_modifiers"],
        known_sets["known_tones"],
        known_sets["all_webcolor_names"],
        llm_client,
        debug=False,
    )
    assert "soft lilac" in compounds

def test_extract_from_adjacent_finds_soft_pink_and_skips_cosmetic(monkeypatch, nlp, known_sets):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    text = "soft pink lipstick"
    tokens = make_tokens(nlp, text)
    compounds, raw = set(), []
    mod.extract_from_adjacent(
        tokens, compounds, raw,
        known_sets["known_modifiers"], known_sets["known_tones"], known_sets["all_webcolor_names"],
        debug=False,
    )
    assert "soft pink" in compounds
    assert all("lipstick" not in c for c in compounds)

def test_extract_from_glued_2part_and_3part(monkeypatch, nlp, known_sets):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    # Use dummy tokens WITH POS=ADJ so the strategy doesn't skip them
    doc = [_DT("dustyrose"), _DT("pastellilacpink")]
    compounds, raw = set(), []
    mod.extract_from_glued(
        doc, compounds, raw,
        known_sets["known_color_tokens"],
        known_sets["known_modifiers"], known_sets["known_tones"],
        known_sets["all_webcolor_names"],
        debug=False,
        aug_vocab=None,
    )
    assert "dusty rose" in compounds
    assert any(c.endswith(" pink") for c in compounds)

def test_extract_from_split_binary_fallback(monkeypatch, nlp, known_sets):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    # Need POS=ADJ for split extractor too
    doc = [_DT("softpink")]
    compounds, raw = set(), []
    mod.extract_from_split(
        doc, compounds, raw,
        known_sets["known_color_tokens"],
        known_sets["known_modifiers"], known_sets["known_tones"],
        known_sets["all_webcolor_names"],
        debug=False,
        aug_vocab=None,
    )
    assert "soft pink" in compounds

def test_extract_compound_phrases_runs_orchestrator(monkeypatch, nlp, known_sets):
    from color_sentiment_extractor.extraction.color.strategies import compound as mod
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    compounds, raw = set(), []

    # On fournit directement des DummyTokens avec pos_="ADJ"
    tokens = [_DT("softpink"), _DT("dustyrose")]

    mod.extract_compound_phrases(
        tokens=tokens,
        compounds=compounds,
        raw_compounds=raw,
        known_color_tokens=known_sets["known_color_tokens"],
        known_modifiers=known_sets["known_modifiers"],
        known_tones=known_sets["known_tones"],
        all_webcolor_names=known_sets["all_webcolor_names"],
        debug=False,
    )
    assert "soft pink" in compounds and "dusty rose" in compounds

# ── Tests: standalone.py ──────────────────────────────────────────────────────
def test_extract_lone_tones_simple(monkeypatch, nlp, known_sets):
    from color_sentiment_extractor.extraction.color.strategies import standalone as st
    toks = make_tokens(nlp, "pink lipstick lilac")
    out = st.extract_lone_tones(toks, known_sets["known_tones"], debug=False)
    assert out == {"pink", "lilac"}

def test_extract_standalone_phrases_injection_and_filter(monkeypatch, nlp, known_sets, llm_client):
    from color_sentiment_extractor.extraction.color.strategies import standalone as st
    patch_recovery_and_token_helpers(monkeypatch, known_sets)

    expression_map = {"glow": ["shiny"], "pastel": ["pastel"]}

    toks = make_tokens(nlp, "pastel glow soft pink")
    out = st.extract_standalone_phrases(
        tokens=toks,
        known_modifiers=known_sets["known_modifiers"],
        known_tones=known_sets["known_tones"],
        expression_map=expression_map,
        llm_client=llm_client,
        debug=False,
        max_injected=3,
    )

    assert {"shiny", "pastel", "soft"}.issubset(out)
