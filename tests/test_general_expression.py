# tests/test_general_expression.py
from __future__ import annotations
"""
Does: End-to-end tests for expression_helpers: trigger vocab, alias extraction,
      expression matching (exact+fuzzy), context/suppression, modifier injection,
      and glued-token vocabulary.
Returns: Deterministic unit tests using a tiny patched config + lightweight stubs.
Used By: CI sanity on expression-driven tone/modifier logic.
"""

import importlib
import re
import sys
import types
import pathlib
import pytest


@pytest.fixture(autouse=True)
def patch_project_dependencies(monkeypatch):
    """
    Does: Ensure correct sys.path, patch deps, purge parent modules from sys.modules,
          then import the module under test fresh.
    Returns: Namespace with `mod` (expression_helpers).
    """
    # --- Put package root on sys.path (supports both layouts: with or without src/)
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidate_src = repo_root / "src"
    pkg_root = candidate_src if candidate_src.exists() else repo_root
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    # ---- Fake config payloads ------------------------------------------------
    FAKE_KNOWN_MODIFIERS = {"soft", "dusty", "matte", "shiny", "glossy"}
    FAKE_KNOWN_TONES     = {"rose", "pink", "taupe", "beige", "lilac"}

    # Validated defs (in this project, "modifiers" == tones post-validation)
    FAKE_EXPR_VALIDATED = {
        "soft glam": {"aliases": ["soft glam", "glamorous look"], "modifiers": ["rose", "pink"]},
        "matte look": {"aliases": ["ultra matte", "fully matte"], "modifiers": ["taupe"]},
        "gloss": {"aliases": ["high shine", "mirror finish"], "modifiers": ["pink"]},
        "romantic": {"aliases": ["date night", "romance vibe"], "modifiers": ["rose", "lilac"]},
        "glam": {"aliases": ["glam", "glam look"], "modifiers": ["pink"]},
    }

    # Raw defs used by get_all_trigger_tokens()
    FAKE_EXPR_RAW = {
        "soft glam": {"aliases": ["soft glam"], "modifiers": ["soft", "glamorous"]},
        "matte look": {"aliases": ["ultra matte"], "modifiers": ["matte"]},
        "gloss": {"aliases": ["high shine"], "modifiers": ["glossy", "shiny"]},
    }

    # Context: promote “romantic” when tokens contain “date” and a clue {night|vibes}
    FAKE_CONTEXT = {
        "romantic": [{"require_tokens": ["date"], "context_clues": ["night", "vibes"]}],
        "soft glam": [], "matte look": [], "gloss": [], "glam": [],
    }

    # Suppression and conflicts
    FAKE_SUPPRESS = {"soft glam": {"glam"}}
    FAKE_SEMANTIC_CONFLICTS = {("matte", "shiny")}

    # ---- Patch real modules (no fake sys.modules inserts) --------------------
    # constants
    const_mod = importlib.import_module("color_sentiment_extractor.extraction.color.constants")
    monkeypatch.setattr(const_mod, "EXPRESSION_SUPPRESSION_RULES", FAKE_SUPPRESS, raising=True)
    monkeypatch.setattr(const_mod, "SEMANTIC_CONFLICTS", FAKE_SEMANTIC_CONFLICTS, raising=True)

    # utils.load_config
    utils_mod = importlib.import_module("color_sentiment_extractor.extraction.general.utils")

    def fake_load_config(name: str, mode: str = "validated_dict"):
        if name == "known_modifiers" and mode == "set": return set(FAKE_KNOWN_MODIFIERS)
        if name == "known_tones"     and mode == "set": return set(FAKE_KNOWN_TONES)
        if name == "expression_definition" and mode == "validated_dict": return FAKE_EXPR_VALIDATED
        if name == "expression_definition" and mode == "raw":            return FAKE_EXPR_RAW
        if name == "expression_context_rules" and mode == "validated_dict": return FAKE_CONTEXT
        raise KeyError(f"Unexpected load_config({name=}, {mode=})")

    monkeypatch.setattr(utils_mod, "load_config", fake_load_config, raising=True)

    # token helpers
    token_pkg = importlib.import_module("color_sentiment_extractor.extraction.general.token")

    def normalize_token(s: str, keep_hyphens: bool = False) -> str:
        s = s.lower().strip().replace("_", "-")
        return s if keep_hyphens else s.replace("-", " ")

    def recover_base(s: str, known_modifiers=None, known_tones=None,
                     debug: bool = False, fuzzy_fallback: bool = False):
        t = normalize_token(s)
        km = FAKE_KNOWN_MODIFIERS if known_modifiers is None else known_modifiers
        kt = FAKE_KNOWN_TONES     if known_tones     is None else known_tones
        if t in km or t in kt:
            return t
        if fuzzy_fallback and t.endswith("y"):
            base = t[:-1]
            return base if base in km or base in kt else None
        return None

    monkeypatch.setattr(token_pkg, "normalize_token", normalize_token, raising=True)
    monkeypatch.setattr(token_pkg, "recover_base",   recover_base,   raising=True)

    # fuzzy expression matcher
    fuzzy_mod = importlib.import_module("color_sentiment_extractor.extraction.general.fuzzy.expression_match")

    def match_expression_aliases(text: str, expression_map):
        text_norm = normalize_token(text, keep_hyphens=True)
        hits, all_aliases = set(), set()
        for data in expression_map.values():
            all_aliases.update(data.get("aliases", []) or [])
            all_aliases.update(data.get("modifiers", []) or [])
        for a in all_aliases:
            n = normalize_token(a, keep_hyphens=True)
            if re.search(rf"(?<![\w/]){re.escape(n)}s?(?![\w/])", text_norm):
                hits.add(n)
        return hits

    monkeypatch.setattr(fuzzy_mod, "match_expression_aliases", match_expression_aliases, raising=True)

    # webcolor vocab
    vocab_mod = importlib.import_module("color_sentiment_extractor.extraction.color.vocab")
    monkeypatch.setattr(vocab_mod, "get_all_webcolor_names", lambda: {"beige", "navy"}, raising=True)

    # ---- Purge target and PARENTS from sys.modules to avoid a non-package parent
    target  = "color_sentiment_extractor.extraction.general.expression.expression_helpers"
    parents = [
        "color_sentiment_extractor.extraction.general.expression",
        "color_sentiment_extractor.extraction.general",
    ]
    for m in [target, *parents]:
        sys.modules.pop(m, None)

    # ---- Fresh import
    eh = importlib.import_module(target)
    return types.SimpleNamespace(mod=eh)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_all_trigger_tokens_builds_tokens(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    trig = eh.get_all_trigger_tokens()
    assert "soft glam" in trig
    assert set(trig["gloss"]) >= {"high shine", "glossy", "shiny"}


def test_get_all_alias_tokens_collects_and_normalizes(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    toks = eh.get_all_alias_tokens({
        "x": {"aliases": ["High Shine"], "modifiers": ["Ultra-Matte"]},
        "y": {"aliases": ["Glamorous Look"], "modifiers": ["Soft"]},
    })
    # keep_hyphens=True in module's _norm → "Ultra-Matte" => "ultra-matte"
    assert {"high shine", "ultra-matte", "glamorous look", "soft"} <= toks


def test_extract_exact_alias_tokens_prefers_longest_forms(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    text = "I want a glam look, like a truly glamorous look tonight."
    hits = eh.extract_exact_alias_tokens(text, {
        "soft glam": {"aliases": ["glamorous look", "glam"], "modifiers": []}
    })
    assert "glamorous look" in hits
    assert "glam" not in hits


def test_map_expressions_with_context_and_suppression(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    text = "Date night soft glam with a glamorous look please."
    mapping = eh.map_expressions_to_tones(text, debug=True)
    assert "romantic" in mapping and set(mapping["romantic"]) == {"rose", "lilac"}
    assert "soft glam" in mapping and "glam" not in mapping
    assert set(mapping["soft glam"]) == {"rose", "pink"}


def test_inject_expression_modifiers_end_to_end(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    inject = getattr(eh, "_inject_expression_modifiers")
    mods = inject(["Ultra", "Matte", "with", "high", "shine", "dusty"], debug=True)
    assert "dusty" in mods
    assert not ({"matte", "shiny"} <= set(mods))


def test_glued_token_vocabulary_includes_knowns_and_webcolors(patch_project_dependencies):
    eh = patch_project_dependencies.mod
    vocab = eh.get_glued_token_vocabulary()
    assert {"rose", "matte", "beige", "navy"} <= set(vocab)
