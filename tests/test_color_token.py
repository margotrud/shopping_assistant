# tests/test_color_token_split.py
from __future__ import annotations

import pytest

# Module à tester
from color_sentiment_extractor.extraction.color.token import split as s


# ───────────────────────────
# Fixtures & Patching
# ───────────────────────────
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Does: Patch les dépendances externes (config, suffix, blocked pairs)
          pour des tests déterministes.
    Returns: None (patches in place sur le module importé).
    """
    # known sets utilisés par plusieurs tests
    KNOWN_TONES = frozenset({"rose", "beige", "lilac", "navy", "blue"})
    KNOWN_MODS = frozenset({"dusty", "soft", "deep"})

    # 1) load_config → renvoie nos mods connus (évite I/O)
    def _fake_load_config(name: str, mode: str = "set"):
        assert name == "known_modifiers"
        assert mode == "set"
        return set(KNOWN_MODS)

    monkeypatch.setattr(s, "load_config", _fake_load_config, raising=True)

    # 2) build_augmented_suffix_vocab → union simple mods+tones
    def _fake_build_vocab(known_tokens, known_modifiers):
        return set(known_tokens) | set(known_modifiers)

    monkeypatch.setattr(s, "build_augmented_suffix_vocab", _fake_build_vocab, raising=True)

    # 3) is_suffix_variant → règles mini pour tests
    def _fake_is_suffix_variant(tok, km, kt, debug=False, allow_fuzzy=False):
        # autoriser "rosy" → "rose", "beigey" → "beige"
        m = {"rosy": "rose", "beigey": "beige"}
        base = m.get(tok)
        return base in set(kt) | set(km) if base else False

    monkeypatch.setattr(s, "is_suffix_variant", _fake_is_suffix_variant, raising=True)

    # 4) BLOCKED_TOKENS par défaut vide (chaque test peut surcharger)
    monkeypatch.setattr(s, "BLOCKED_TOKENS", set(), raising=True)

    # 5) Renvoyer nos sets connus facilement dans les tests
    return {"tones": KNOWN_TONES, "mods": KNOWN_MODS}


@pytest.fixture
def km_kt(patch_dependencies):
    """
    Does: Expose (known_modifiers, known_tones) patchés aux tests.
    Returns: dict avec 'mods' et 'tones'.
    """
    return patch_dependencies


# ───────────────────────────
# Tests split_glued_tokens
# ───────────────────────────
def test_split_glued_tokens_basic(km_kt):
    """Does: Vérifie le split collé simple 'dustyrose' -> ['dusty','rose']."""
    parts = s.split_glued_tokens("dustyrose", km_kt["tones"], km_kt["mods"], debug=True)
    assert parts == ["dusty", "rose"]


def test_split_glued_tokens_hyphenated_passthrough(km_kt):
    """Does: Vérifie que 'deep-beige' est reconnu via découpe directe."""
    parts = s.split_glued_tokens("deep-beige", km_kt["tones"], km_kt["mods"], debug=True)
    assert parts == ["deep", "beige"]


def test_split_glued_tokens_suffix_variant_validation(km_kt, monkeypatch):
    """Does: Vérifie la validation suffix-aware ('beigey' reconnu via is_suffix_variant)."""
    # On force une entrée collée qui doit valider le morceau droit par suffix-variant
    parts = s.split_glued_tokens("deepbeigey", km_kt["tones"], km_kt["mods"], debug=True)
    # 'deep' et 'beigey' (validé contre 'beige') doivent ressortir
    assert parts == ["deep", "beigey"]


def test_split_glued_tokens_uses_fallback_on_timeout(km_kt, monkeypatch):
    """Does: Force un timeout et vérifie le fallback longest-substring."""
    # Patch time budget très court pour forcer le chemin fallback
    def _fast(*args, **kwargs):
        return []

    # monkeypatch fallback pour un résultat déterministe
    def _fake_fallback(token, vocab, debug=False):
        return ["soft", "dusty", "rose"]

    monkeypatch.setattr(s, "fallback_split_on_longest_substring", _fake_fallback, raising=True)
    parts = s.split_glued_tokens(
        "softdustyrose",
        km_kt["tones"],
        km_kt["mods"],
        debug=True,
        time_budget_sec=0.00001,
    )
    assert parts == ["soft", "dusty", "rose"]


# ───────────────────────────
# Tests split_tokens_to_parts (strict 2-parts)
# ───────────────────────────
def test_split_tokens_to_parts_hyphen_shortcut(km_kt):
    """Does: Vérifie le split 2-parties direct via tiret."""
    parts = s.split_tokens_to_parts("dusty-rose", km_kt["tones"], km_kt["mods"], debug=True)
    assert parts == ["dusty", "rose"]


def test_split_tokens_to_parts_exact_membership(km_kt):
    """Does: Vérifie qu’un token exact connu à gauche/droite est accepté."""
    parts = s.split_tokens_to_parts("softbeige", km_kt["tones"], km_kt["mods"], debug=True)
    # split() essaie toutes les coupes; la bonne est ["soft","beige"]
    assert parts == ["soft", "beige"]


def test_split_tokens_to_parts_recovery_left_and_right(km_kt, monkeypatch):
    """Does: Test le recovery strict: 'rosy'→'rose' côté droit; gauche déjà connu."""
    # Patch _recover_base_cached_with_params pour la partie gauche si besoin (ici no-op)
    def _fake_recover_left(**kwargs):
        # on laisse None pour forcer l'usage des membres connus seulement
        return None

    monkeypatch.setattr(s, "_recover_base_cached_with_params", _fake_recover_left, raising=True)

    # Patch recover_base pour la partie droite: 'rosy' → 'rose'
    def _fake_recover_right(
            token,
            known_modifiers,
            known_tones,
            debug=False,
            fuzzy_fallback=False,
            fuzzy_threshold=88,
    ):
        return "rose" if token == "rosy" else None

    monkeypatch.setattr(s, "recover_base", _fake_recover_right, raising=True)

    parts = s.split_tokens_to_parts("dustyrosy", km_kt["tones"], km_kt["mods"], debug=True)
    assert parts == ["dusty", "rose"]


def test_split_tokens_to_parts_blocked_pair(km_kt, monkeypatch):
    """Does: Vérifie qu’une paire bloquée est rejetée (None)."""
    # Bloque ("dusty","rose")
    monkeypatch.setattr(s, "BLOCKED_TOKENS", {("dusty", "rose")}, raising=True)
    parts = s.split_tokens_to_parts("dustyrose", km_kt["tones"], km_kt["mods"], debug=True)
    assert parts is None


def test_split_tokens_to_parts_min_length_guard(km_kt):
    """Does: Assure qu’aucun split n’est proposé si une partie serait trop courte."""
    # 'de' + 'rose' ne doit pas passer le min_part_len=3
    parts = s.split_tokens_to_parts(
        "derose",
        km_kt["tones"],
        km_kt["mods"],
        debug=True,
        min_part_len=3,
    )
    assert parts is None
