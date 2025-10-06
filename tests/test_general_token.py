# tests/test_general_token.py
from __future__ import annotations
import pytest

# Modules sous test
from color_sentiment_extractor.extraction.general.token import base_recovery as BR
from color_sentiment_extractor.extraction.general.token import normalize as N


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: patchs déterministes pour constants, suffix-chain, fuzzy, nouns
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_constants_and_helpers(monkeypatch):
    """
    Does: Stabilise l'environnement de test pour base_recovery + normalize.
    """
    # Vocab connu (on passe aussi explicitement dans les appels pour éviter le cache)
    known_mods = {"dusty", "soft", "gloss", "cream"}
    known_tones = {"rose", "beige", "navy", "blue"}

    # constants importées directement dans BR
    monkeypatch.setattr(BR, "RECOVER_BASE_OVERRIDES", {
        "rosy": "rose",    # forme dérivée → base tone
        "flashy": "flash", # (flash n'est pas dans le vocab → sera ignoré s'il est testé)
    }, raising=True)

    # groupes de conflits sémantiques (empêche des mappages douteux)
    monkeypatch.setattr(BR, "SEMANTIC_CONFLICTS", [
        {"airy", "fairy"}, {"matte", "gloss"}, {"blue", "rose"}
    ], raising=True)

    # tokens bloqués dans les deux sens
    monkeypatch.setattr(BR, "BLOCKED_TOKENS", {
        ("mint", "tint"), ("tint", "mint")
    }, raising=True)

    # Suffix recovery minimaliste et déterministe
    def _recover_y(raw: str, *_args, **_kwargs):
        # ex: creamy → cream ; dusty → dust (non connu, ensuite chaining/fuzzy peut aider)
        return raw[:-1] if raw.endswith("y") and len(raw) > 3 else None

    def _recover_ed(raw: str, *_args, **_kwargs):
        # ex: inked → ink ; noté pour compat mais pas utilisé dans ces tests par défaut
        return raw[:-2] if raw.endswith("ed") and len(raw) > 3 else None

    monkeypatch.setattr(BR, "SUFFIX_RECOVERY_FUNCS", [_recover_y, _recover_ed], raising=True)

    # Fuzzy déterministe: retourne un match seulement pour quelques cas
    def _fuzzy_match_token_safe(s: str, candidates: set[str], thr: int, _unused: bool):
        table = {
            "creem": "cream",
            "glos": "gloss",
            "gls": "gloss",   # abréviation courte
            "beije": "beige",
        }
        cand = table.get(s)
        return cand if cand in candidates else None

    monkeypatch.setattr(BR, "fuzzy_match_token_safe", _fuzzy_match_token_safe, raising=True)

    # Patch nouns (pour la singularisation domaine) côté normalize
    monkeypatch.setattr(N, "COSMETIC_NOUNS", {"lipstick", "lip gloss"}, raising=True)

    # Expose aussi via retour si un test en a besoin
    return {"mods": known_mods, "tones": known_tones}


# ──────────────────────────────────────────────────────────────────────────────
# Tests: base_recovery.recover_base
# ──────────────────────────────────────────────────────────────────────────────

def test_recover_base_direct_hit(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    assert BR.recover_base("cream", known_modifiers=km, known_tones=kt) == "cream"
    assert BR.recover_base("ROSE", known_modifiers=km, known_tones=kt) == "rose"

def test_recover_base_override_applies_first(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    # override dict: rosy → rose (dans tones)
    assert BR.recover_base("rosy", known_modifiers=km, known_tones=kt) == "rose"

def test_recover_base_suffix_chain_to_known_base(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    # creamy → (recover_y) → cream ∈ known_modifiers
    assert BR.recover_base("creamy", known_modifiers=km, known_tones=kt) == "cream"

def test_recover_base_suffix_then_mid_fuzzy_salvage(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    # "creem" n'est pas suffixable → fuzzy renvoie "cream"
    assert BR.recover_base("creem", known_modifiers=km, known_tones=kt) == "cream"

def test_recover_base_fuzzy_blocked_and_conflict_guards(patch_constants_and_helpers, monkeypatch):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]

    # 1) first-letter guard: ne doit pas mapper 'blue' → 'rose' même si fuzzy le ferait
    def _bad_fuzzy(s, cands, thr, _):
        return "rose" if s == "blue" else None
    monkeypatch.setattr(BR, "fuzzy_match_token_safe", _bad_fuzzy, raising=True)
    assert BR.recover_base("blue", known_modifiers=km, known_tones=kt) == "blue"  # direct hit prioritaire
    # Si on empêche le direct hit pour tester le guard, on passe un vocab sans 'blue'
    assert BR.recover_base("blue", known_modifiers=km, known_tones={"rose"}) is None

def test_recover_base_blocked_tokens_guard(patch_constants_and_helpers, monkeypatch):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]

    def _fuzzy_tint_mint(s, cands, thr, _):
        return "mint" if s == "tint" else None
    monkeypatch.setattr(BR, "fuzzy_match_token_safe", _fuzzy_tint_mint, raising=True)

    # BLOCKED_TOKENS contient (tint, mint) → doit refuser
    assert BR.recover_base("tint", known_modifiers=km, known_tones=kt) is None

def test_recover_base_abbreviation_fallback_prefers_modifiers(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    # “gls” (3 lettres) : squelette consonantique/voyelles → gloss (modifier)
    out = BR.recover_base("gls", known_modifiers=km, known_tones=kt)
    assert out == "gloss"

def test_recover_base_no_match_returns_none(patch_constants_and_helpers):
    km, kt = patch_constants_and_helpers["mods"], patch_constants_and_helpers["tones"]
    assert BR.recover_base("zzzxxx", known_modifiers=km, known_tones=kt) is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: normalize (singularize, normalize_token, get_tokens_and_counts)
# ──────────────────────────────────────────────────────────────────────────────

def test_singularize_basic_rules():
    assert N.singularize("berries") == "berry"
    assert N.singularize("boxes") == "box"
    assert N.singularize("glosses") == "gloss"
    assert N.singularize("nudes") == "nude"
    # invariants / courts
    assert N.singularize("series") == "series"
    assert N.singularize("ink") == "ink"

def test_normalize_token_hyphens_and_cosmetic_last():
    # keep_hyphens=False: hyphens → spaces ; dernière noun cosmétique au singulier
    assert N.normalize_token("Soft-Blue LIPSTICKS", keep_hyphens=False) == "soft blue lipstick"
    # keep_hyphens=True: on garde les tirets ; pas de singulier car le segment "gloss" seul n'est pas un nom cosmétique
    assert N.normalize_token("navy-Blue   lip-Glosses", keep_hyphens=True) == "navy-blue lip-glosses"

def test_unicode_hygiene_and_space_collapse():
    # guillemets/traits d'union fancy → le tiret long est normalisé puis converti en espace ;
    # les guillemets doubles “ ” NE sont pas mappés actuellement → ils restent.
    s = N.normalize_token(" “Cream—Blue”  _  lipstick  ", keep_hyphens=False)
    assert s == "“cream blue” lipstick"

def test_get_tokens_and_counts_hyphen_aware():
    text = "Soft-blue lip-gloss, soft blue  lip-glosses!"
    counts = N.get_tokens_and_counts(text, keep_hyphens=True)
    # On ne fusionne pas "soft blue" → ils restent séparés.
    assert counts.get("soft-blue", 0) == 1
    assert counts.get("soft", 0) == 1
    assert counts.get("blue", 0) == 1
    assert counts.get("lip-gloss", 0) == 1
    assert counts.get("lip-glosses", 0) == 1

