# tests/test_recovery.py
import pytest

from color_sentiment_extractor.extraction.general.utils import load_config
from color_sentiment_extractor.extraction.general.token import normalize_token
from color_sentiment_extractor.extraction.color.recovery import (
    fuzzy_recovery,
    llm_recovery,
    modifier_resolution,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def km():
    return set(load_config("known_modifiers", mode="set"))

@pytest.fixture(scope="module")
def kt():
    return set(load_config("known_tones", mode="set"))

class DummyLLM:
    def simplify(self, phrase: str) -> str:
        # Ici on renvoie la version normalisée, neutre et prédictible
        return normalize_token(phrase, keep_hyphens=True)

@pytest.fixture
def llm():
    return DummyLLM()


# ---------------------------------------------------------------------
# fuzzy_recovery tests
# ---------------------------------------------------------------------

def describe_fuzzy_recovery():

    def it_rejects_empty_strings(km, kt):
        assert not fuzzy_recovery.is_suffix_root_match("", "", known_modifiers=km, known_tones=kt)

    def it_matches_suffix_and_root_variants(km, kt):
        assert fuzzy_recovery.is_suffix_root_match("rosy", "rose", known_modifiers=km, known_tones=kt)

    def it_respects_semantic_conflicts(km, kt):
        # On force une paire en conflit connu si présente
        for a, b in fuzzy_recovery.SEMANTIC_CONFLICTS:
            assert not fuzzy_recovery.is_suffix_root_match(a, b, known_modifiers=km, known_tones=kt)
            break


# ---------------------------------------------------------------------
# llm_recovery tests
# ---------------------------------------------------------------------

def describe_llm_recovery():

    def it_simplifies_known_tone(llm, kt, km):
        tone = next(iter(kt))
        simplified = llm_recovery.simplify_phrase_if_needed(tone, km, kt, llm)
        assert simplified == tone

    def it_preserves_surface_modifiers(llm, km, kt):
        phrase = "dusty rose"  # devrait être conservé
        simplified = llm_recovery.simplify_phrase_if_needed(phrase, km, kt, llm)
        assert simplified == phrase

    def it_attempts_simplify_token_with_role_modifier(llm, km, kt):
        mod = next(iter(km))
        result = llm_recovery._attempt_simplify_token(mod, km, kt, llm, role="modifier")
        assert result in km or result in kt

    def it_extracts_filtered_tokens_removes_cosmetic_nouns(llm, km, kt):
        class Tok:  # stub spaCy-like
            def __init__(self, text, pos_="NOUN"):
                self.text = text
                self.pos_ = pos_
        tokens = [Tok("lipstick"), Tok("dusty")]
        result = llm_recovery._extract_filtered_tokens(tokens, km, kt, llm, debug=False)
        assert "lipstick" not in result


# ---------------------------------------------------------------------
# modifier_resolution tests
# ---------------------------------------------------------------------

def describe_modifier_resolution():

    def it_matches_direct_modifier(km):
        mod = next(iter(km))
        assert modifier_resolution.match_direct_modifier(mod, km) == mod

    def it_resolves_with_base_recovery(km, kt):
        mod = next(iter(km))
        recovered = modifier_resolution.resolve_modifier_token(mod, km, kt)
        assert recovered in km or recovered is None

    def it_detects_suppressed_compounds():
        assert modifier_resolution.should_suppress_compound("red", "red")

    def it_blocks_explicit_pairs():
        if modifier_resolution.BLOCKED_TOKENS:
            m, t = next(iter(modifier_resolution.BLOCKED_TOKENS))
            assert modifier_resolution.is_blocked_modifier_tone_pair(m, t)

    def it_fallback_resolves_tokens(km, kt):
        class Tok:
            def __init__(self, text): self.text = text
        toks = [Tok(next(iter(km)))]
        result = modifier_resolution.resolve_fallback_tokens(toks, km, kt)
        assert isinstance(result, set)
