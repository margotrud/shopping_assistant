"""
Tests for logic/classification/ (categorizer.py)

Couvre:
- build_tone_modifier_mappings()
- format_tone_modifier_mappings()

Règles vérifiées:
- Bigrammes avec espaces, hyphens ASCII + Unicode, underscores
- Filtrage des cosmetic nouns (ex: 'lipstick') côté tones
- Préférence du hit direct du mod (mod_raw ∈ known_modifiers) sur la base recoverée
- Sortie stable et triée pour format_tone_modifier_mappings()
"""

import importlib

categorizer = importlib.import_module(
    "color_sentiment_extractor.extraction.color.logic.classification.categorizer"
)

def test_module_exports():
    assert hasattr(categorizer, "build_tone_modifier_mappings")
    assert hasattr(categorizer, "format_tone_modifier_mappings")

def test_build_mappings_basic_spaces_and_hyphens():
    phrases = ["dusty rose", "soft-beige"]
    known_tones = {"rose", "beige", "nude"}
    known_modifiers = {"dusty", "soft", "rosy"}
    tones, mods, m2t, t2m = categorizer.build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )
    assert "rose" in tones and "beige" in tones
    assert "dusty" in mods and "soft" in mods
    assert "rose" in m2t["dusty"]
    assert "dusty" in t2m["rose"]

def test_build_mappings_unicode_hyphen_and_underscore():
    phrases = ["super-dusty—rose", "rosy_ nude"]  # '—' = em dash
    known_tones = {"rose", "nude"}
    known_modifiers = {"dusty", "rosy"}
    tones, mods, m2t, t2m = categorizer.build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )
    # super-dusty—rose -> candidates ('super','dusty'), ('dusty','rose') → garde 'dusty rose'
    assert "rose" in tones and "dusty" in mods
    # rosy_ nude -> underscore devient '-' puis bigrammes → 'rosy' + 'nude'
    assert "nude" in tones and "rosy" in mods
    assert "rose" in m2t["dusty"]
    assert "dusty" in t2m["rose"]

def test_cosmetic_nouns_filtered_from_tones():
    # 'lipstick' ne doit jamais être accepté comme tone même s'il apparaît en 2ᵉ position
    phrases = ["dusty lipstick", "rosy lipstick"]
    known_tones = {"lipstick", "rose"}   # même si présent, doit être filtré par COSMETIC_NOUNS interne
    known_modifiers = {"dusty", "rosy"}
    tones, mods, m2t, t2m = categorizer.build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )
    assert "lipstick" not in tones
    assert "dusty" not in m2t or "lipstick" not in m2t.get("dusty", set())

def test_direct_modifier_hit_preferred_over_base_recovery():
    # Si mod_raw ∈ known_modifiers, il est préféré et casse la boucle (break)
    phrases = ["rosy rose"]
    known_tones = {"rose"}
    known_modifiers = {"rosy"}  # direct hit
    tones, mods, m2t, t2m = categorizer.build_tone_modifier_mappings(
        phrases, known_tones, known_modifiers
    )
    assert "rosy" in mods and "rose" in tones
    assert "rose" in m2t["rosy"]
    assert "rosy" in t2m["rose"]

def test_format_mappings_sorted_and_stable_keys():
    phrases = ["dusty rose", "soft beige", "dusty rose"]  # duplicata OK
    known_tones = {"rose", "beige"}
    known_modifiers = {"dusty", "soft"}
    out = categorizer.format_tone_modifier_mappings(phrases, known_tones, known_modifiers)
    assert set(out.keys()) == {"modifiers", "tones"}
    # clés triées + valeurs triées
    mods_keys = list(out["modifiers"].keys())
    tones_keys = list(out["tones"].keys())
    assert mods_keys == sorted(mods_keys)
    assert tones_keys == sorted(tones_keys)
    for v in out["modifiers"].values():
        assert v == sorted(v)
    for v in out["tones"].values():
        assert v == sorted(v)

def test_empty_input_returns_empty_dicts():
    out = categorizer.format_tone_modifier_mappings([], set(), set())
    assert out == {"modifiers": {}, "tones": {}}
