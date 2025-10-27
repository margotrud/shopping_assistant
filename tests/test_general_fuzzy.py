# tests/test_general_fuzzy.py
from __future__ import annotations

import re
import sys

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Auto-patches (avant import des SUT) — normalizer, base recovery, suffix-root
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_normalize_and_recovery(monkeypatch):
    """
    Does: Provide lightweight normalizer + base/suffix recovery + suffix-root matcher.
    Ensures: AUCUN accès à known_tones.json (on bypass via monkeypatch).
    """

    # --- Normalizer (hyphen-aware) ---
    def _normalize_token(s: str, keep_hyphens: bool = False) -> str:
        if s is None:
            return ""
        s = str(s).lower().strip()
        s = s.replace("_", "-")
        if not keep_hyphens:
            s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    # --- Base recovery minimal ---
    # --- Base recovery minimal ---
    def _recover_base(
            s: str,
            use_cache: bool = True,
            allow_fuzzy: bool = False,
            debug: bool = False,
    ):
        x = _normalize_token(s, keep_hyphens=True)

        if x.endswith("y") and len(x) > 3:
            base = x[:-1]
            # beigey/bronzey → beige/bronze
            base = re.sub(r"e$", "", base) if base.endswith(("ge", "ze")) else base
            if base == "ros":    # rosy → rose
                return "rose"
            if base == "shin":   # shiny → shine
                return "shine"
            return base

        if x.endswith("ed") and len(x) > 3:
            return x[:-2]

        if x.endswith("ing") and len(x) > 4:
            return x[:-3]

        return x

    def _is_suffix_root_match(a: str, b: str, debug: bool = False) -> bool:
        return _recover_base(a) == _recover_base(b)

    # general.token.normalize
    import color_sentiment_extractor.extraction.general.token.normalize as norm_mod
    monkeypatch.setattr(norm_mod, "normalize_token", _normalize_token, raising=True)
    monkeypatch.setattr(
        norm_mod,
        "get_tokens_and_counts",
        lambda s: {t: 1 for t in _normalize_token(s).split()},
        raising=True,
    )

    # general.token.base_recovery
    import color_sentiment_extractor.extraction.general.token.base_recovery as br_mod
    monkeypatch.setattr(br_mod, "recover_base", _recover_base, raising=True)

    # color.recovery.fuzzy_recovery
    import color_sentiment_extractor.extraction.color.recovery.fuzzy_recovery as fr
    monkeypatch.setattr(fr, "is_suffix_root_match", _is_suffix_root_match, raising=True)

    # color.recovery (package re-export)
    import color_sentiment_extractor.extraction.color.recovery as rec_pkg
    monkeypatch.setattr(rec_pkg, "is_suffix_root_match", _is_suffix_root_match, raising=False)

    # ⚠️ Module déjà importé ? Patch direct du symbole local pour éviter l'ancien binding
    mod_name = "color_sentiment_extractor.extraction.general.fuzzy.alias_validation"
    if mod_name in sys.modules:
        av = sys.modules[mod_name]
        try:
            monkeypatch.setattr(av, "is_suffix_root_match", _is_suffix_root_match, raising=True)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def patch_expression_config(monkeypatch):
    """Does: Patch expression_match to use a tiny, stable config dict."""
    from color_sentiment_extractor.extraction.general.fuzzy import expression_match as em

    def _fake_def():
        return {
            "rose gold": {
                "aliases": ["rose gold", "rosegold"],
                "modifiers": ["soft", "dusty"],
            },
            "taupe": {
                "aliases": ["taupe"],
                "modifiers": ["soft"],
            },
        }

    monkeypatch.setattr(em, "_get_expression_def", _fake_def, raising=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests conflict_rules.py
# ─────────────────────────────────────────────────────────────────────────────

def test_is_negation_conflict_basic():
    from color_sentiment_extractor.extraction.general.fuzzy import conflict_rules as CR
    assert CR.is_negation_conflict("no shimmer", "shimmer") is True
    assert CR.is_negation_conflict("without glitter", "glittery") is True
    assert CR.is_negation_conflict("soft glow", "glow") is False


def test_is_embedded_alias_conflict_single_token():
    from color_sentiment_extractor.extraction.general.fuzzy import conflict_rules as CR
    assert CR.is_embedded_alias_conflict("rosewood", "rose") is True
    assert CR.is_embedded_alias_conflict("rosegold", "rose") is True
    assert CR.is_embedded_alias_conflict("rose gold", "rose") is False  # multi-token ignoré


# ─────────────────────────────────────────────────────────────────────────────
# Tests scoring.py
# ─────────────────────────────────────────────────────────────────────────────

def test_rhyming_conflict_short_tokens():
    from color_sentiment_extractor.extraction.general.fuzzy import scoring as S
    assert S.rhyming_conflict("gloss", "moss") is True
    assert S.rhyming_conflict("rose", "hose") is True
    assert S.rhyming_conflict("rose", "rosette") is False


def test_fuzzy_token_overlap_count_consumes_once():
    from color_sentiment_extractor.extraction.general.fuzzy import scoring as S
    a = ["soft", "rose"]
    b = ["sooft", "rose"]
    assert S.fuzzy_token_overlap_count(a, b) == 2


def test_fuzzy_token_score_prefix_bonus_and_len_penalty():
    from color_sentiment_extractor.extraction.general.fuzzy import scoring as S
    assert S.fuzzy_token_score("roseg", "rosegold") > 70
    assert S.fuzzy_token_score("matte", "gloss") < 50


# ─────────────────────────────────────────────────────────────────────────────
# Tests fuzzy_core.py
# ─────────────────────────────────────────────────────────────────────────────

def test_is_exact_match_normalizes_hyphens_spaces():
    from color_sentiment_extractor.extraction.general.fuzzy import fuzzy_core as FC
    assert FC.is_exact_match("Rose-Gold", "rose gold") is True
    assert FC.is_exact_match("rose_gold", "rose gold") is True


def test_is_strong_fuzzy_match_with_conflict_and_negation_guards():
    from color_sentiment_extractor.extraction.general.fuzzy import fuzzy_core as FC

    assert FC.is_strong_fuzzy_match(
        "matte",
        "glossy",
        conflict_groups=[{"matte", "glossy"}],
    ) is False
    assert FC.is_strong_fuzzy_match("no shimmer", "shimmer") is False
    assert FC.is_strong_fuzzy_match("rosegold", "rose gold") is True


def test_fuzzy_match_token_safe_best_and_edits():
    from color_sentiment_extractor.extraction.general.fuzzy import fuzzy_core as FC
    known = {"blue", "green", "rose"}
    assert FC.fuzzy_match_token_safe("gren", known) == "green"     # insertion simple
    assert FC.fuzzy_match_token_safe("rose-gold", {"rose gold"}) == "rose gold"  # normalisation


# ─────────────────────────────────────────────────────────────────────────────
# Tests alias_validation.py
# ─────────────────────────────────────────────────────────────────────────────

def test_is_valid_singleword_alias_with_root_equivalence():
    from color_sentiment_extractor.extraction.general.fuzzy import alias_validation as AV
    tokens = ["dusty", "rose", "glow"]
    tokens = ["dusty", "rose", "glow"]
    ok = AV.is_valid_singleword_alias(
        "rosy",
        "soft rosy glow",
        tokens,
        matched_aliases=set(),
        debug=False,
    )
    assert ok is True  # rosy ~ rose via patch
    assert ok is True  # rosy ~ rose via patch


def test_should_accept_multiword_alias_variants_and_reorder():
    from color_sentiment_extractor.extraction.general.fuzzy import alias_validation as AV
    assert AV.should_accept_multiword_alias("rose gold", "rosegold") is True
    assert AV.should_accept_multiword_alias("gold rose", "rose gold") is True


def test_remove_subsumed_matches_prefers_longest():
    from color_sentiment_extractor.extraction.general.fuzzy import alias_validation as AV
    inp = ["rose", "dusty rose", "rose gold"]
    out = AV.remove_subsumed_matches(inp)
    assert "dusty rose" in out and "rose gold" in out and "rose" not in out


# ─────────────────────────────────────────────────────────────────────────────
# Tests expression_match.py (E2E avec config patchée)
# ─────────────────────────────────────────────────────────────────────────────

def test_match_expression_aliases_alias_first_then_mods():
    from color_sentiment_extractor.extraction.general.fuzzy import expression_match as EM
    res = EM.match_expression_aliases("soft dusty rose gold shimmer", EM._get_expression_def())
    assert "rose gold" in res
    res2 = EM.match_expression_aliases("very soft finish", EM._get_expression_def())
    assert ("taupe" in res2) or (res2 == set())


def test_cached_match_expression_aliases_lru_works(monkeypatch):
    from color_sentiment_extractor.extraction.general.fuzzy import expression_match as EM
    calls = {"n": 0}

    # Capture l'original AVANT patch
    original_get = EM._get_expression_def

    def _spy():
        calls["n"] += 1
        return original_get()  # ← appelle la version originale, pas la patchée

    # Patch avec l'espion
    monkeypatch.setattr(EM, "_get_expression_def", _spy, raising=True)

    # Vide le cache et vérifie l'appel unique
    EM.cached_match_expression_aliases.cache_clear()
    # Vide le cache et vérifie l'appel unique
    EM.cached_match_expression_aliases.cache_clear()
    EM.cached_match_expression_aliases(
        "rose gold glow"
    )  # → 1er appel : charge via _spy → original_get
    EM.cached_match_expression_aliases(
        "rose gold glow"
    )  # → 2e appel : doit être servi par le cache

    assert calls["n"] == 1
