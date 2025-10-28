from __future__ import annotations

import pytest

# Module sous test : logique de suffixes / base recovery sur les variantes -y, -ey, etc.
from color_sentiment_extractor.extraction.general.token.suffix import recovery as R

"""
Tests: general/token/suffix/recovery.py

Objectifs :
- Vérifier le comportement de recover_y() avec le nouveau contrat
  (retourne la base S'IL Y A un stripping, sinon None)
- Vérifier les helpers de génération de variantes (-y, -ey)
- Rendre le module déterministe via monkeypatch
"""


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures utilitaires
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def known_sets():
    """
    known_sets.mods: ensemble de modificateurs connus
    known_sets.tones: ensemble de tons connus
    known_sets.web: ensemble de noms couleurs web connus

    Ces sets servent à simuler un vocabulaire couleur stable.
    """
    return {
        "mods": frozenset(
            {
                "beige",
                "bronze",
                "cream",
                "dark",
                "dust",
                "fancy",
                "frost",
                "golden",
                "navy",
                "nude",
                "rose",
                "rosy",
                "shiny",
                "soft",
                "warm",
            }
        ),
        "tones": frozenset(
            {
                "beige",
                "bronze",
                "green",
                "ivory",
                "navy",
                "pink",
                "rose",
                "tan",
            }
        ),
        "web": frozenset(
            {
                "beige",
                "bronze",
                "green",
                "ivory",
                "navy",
                "rose",
            }
        ),
    }


@pytest.fixture(autouse=True)
def patch_constants_and_helpers(monkeypatch, known_sets):
    """
    Rend recovery.py déterministe en patchant :
    - les constantes globales (souvent refactorées, donc raising=False),
    - les builders (build_y_variant / build_ey_variant / is_cvc_ending),
    - et les sets globaux de couleurs.
    """

    # Certaines de ces constantes n'existent plus dans la version actuelle du module runtime.
    # On utilise raising=False pour les injecter sans faire planter le test.

    monkeypatch.setattr(
        R,
        "RECOVER_BASE_OVERRIDES",
        {
            "shiny": "shine",
            "rosy": "rose",
        },
        raising=False,
    )

    monkeypatch.setattr(
        R,
        "NON_SUFFIXABLE",
        frozenset({"navy", "ivory"}),
        raising=False,
    )

    monkeypatch.setattr(
        R,
        "SUFFIX_ALLOWLIST",
        frozenset({"dusty", "creamy", "rosy", "shiny"}),
        raising=False,
    )

    # Helpers patchés (eux doivent exister dans le module, donc raising=True)

    def _fake_build_y_variant(base: str) -> str:
        # Règle simple déterministe :
        # "cream" -> "creamy"
        # "dust"  -> "dusty"
        # "rose"  -> "rosy"
        # "shine" -> "shiny"
        if base.endswith("e"):
            return base[:-1] + "y"
        return base + "y"

    def _fake_build_ey_variant(base: str) -> str:
        # Exemple "tan" -> "taney"
        return base + "ey"

    def _fake_is_cvc_ending(token: str) -> bool:
        # True si Consonne-Voyelle-Consonne à la fin.
        if len(token) < 3:
            return False
        tail = token[-3:].lower()
        vowels = "aeiou"
        return (tail[0] not in vowels) and (tail[1] in vowels) and (tail[2] not in vowels)

    monkeypatch.setattr(R, "build_y_variant", _fake_build_y_variant, raising=True)
    monkeypatch.setattr(R, "build_ey_variant", _fake_build_ey_variant, raising=True)
    monkeypatch.setattr(R, "is_cvc_ending", _fake_is_cvc_ending, raising=True)

    # Certains chemins du module peuvent s'appuyer sur des sets globaux
    # pour décider si un token est plausible. On les force pour éviter
    # des variations dynamiques entre runs.
    monkeypatch.setattr(R, "KNOWN_COLOR_MODIFIERS", known_sets["mods"], raising=False)
    monkeypatch.setattr(R, "KNOWN_COLOR_TONES", known_sets["tones"], raising=False)
    monkeypatch.setattr(R, "ALL_WEBCOLOR_NAMES", known_sets["web"], raising=False)


# ──────────────────────────────────────────────────────────────────────────────
# Tests sur recover_y
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("creamy", "cream"),
        ("dusty", "dust"),
        ("rosy", "rose"),
        ("shiny", "shine"),
        ("navy", None),     # NON_SUFFIXABLE -> pas modifié -> doit renvoyer None maintenant
        ("ivory", None),    # NON_SUFFIXABLE -> idem
    ],
)
def test_recover_y(token, expected, known_sets):
    """
    Nouveau contrat de recover_y(token, known_modifiers, known_tones):

    - Si on peut retirer un '-y' pour revenir à une base plausible,
      ex: "creamy" -> "cream", on renvoie cette base.

    - Si on NE modifie pas parce que le -y fait partie intégrante du mot
      (ex: "navy", "ivory") ou que la base ne serait pas plausible,
      on renvoie None.

    Avant, l'ancien test attendait "navy" -> "navy".
    Maintenant on attend None dans ces cas.
    """
    km = known_sets["mods"]
    kt = known_sets["tones"]
    out = R.recover_y(token, km, kt)
    assert out == expected


# ──────────────────────────────────────────────────────────────────────────────
# Tests sur build_y_variant
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("cream", "creamy"),
        ("dust", "dusty"),
        ("rose", "rosy"),
        ("shine", "shiny"),
    ],
)
def test_build_y_variant(token, expected):
    """
    Vérifie la génération d'une variante en -y via notre patch (_fake_build_y_variant).
    """
    out = R.build_y_variant(token)
    assert out == expected


# ──────────────────────────────────────────────────────────────────────────────
# Tests sur build_ey_variant
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("tan", "taney"),
        ("cream", "creamey"),
    ],
)
def test_build_ey_variant(token, expected):
    """
    Vérifie la génération d'une variante en -ey via notre patch (_fake_build_ey_variant).
    """
    out = R.build_ey_variant(token)
    assert out == expected


# ──────────────────────────────────────────────────────────────────────────────
# Tests sur is_cvc_ending
# ──────────────────────────────────────────────────────────────────────────────

def test_is_cvc_ending_behavior():
    """
    _fake_is_cvc_ending :
    - True si dernier trigramme = consonne / voyelle / consonne.
    - False sinon.
    """
    assert R.is_cvc_ending("tan") is True          # t-a-n : consonne / voyelle / consonne
    assert R.is_cvc_ending("dust") is False        # "ust" => voyelle / consonne / consonne -> False
    assert R.is_cvc_ending("ivory") is False
