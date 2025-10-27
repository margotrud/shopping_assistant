from __future__ import annotations

import math
import re

import pytest

from color_sentiment_extractor.extraction.general.sentiment import core as S
from color_sentiment_extractor.extraction.general.sentiment import router as R

"""
Tests: general/sentiment (core.py & router.py)
- Sans téléchargement NLTK ni modèles HF (fakes/DI)
- Couvre: VADER/BART hybrid, soft/hard negation, split de clauses,
          classification binaire sans "neutral",
          et build_color_sentiment_summary (router) avec monkeypatch.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & Fakes
# ──────────────────────────────────────────────────────────────────────────────

class FakeVader:
    """Minimal VADER fake: scores with word-boundary matching; negatives win over positives."""

    _NEG = re.compile(r"\b(hate|dislike|awful|bad|terrible)\b", flags=re.I)
    _POS = re.compile(r"\b(love|like|great|amazing|good|fan of)\b", flags=re.I)

    def polarity_scores(self, text: str):
        if self._NEG.search(text):
            return {"compound": -0.7}
        if self._POS.search(text):
            return {"compound": 0.6}
        return {"compound": 0.0}

class FakeBart:
    """Fake zero-shot pipeline: renvoie un ordre de labels déterministe comme core._FakeZeroShot."""

    def __call__(self, text, labels):
        low = str(text).lower()
        if any(k in low for k in ("love", "like", "great", "good", "amazing")):
            order = [labels[0], labels[2], labels[1]]  # positive > neutral > negative
        elif any(k in low for k in ("hate", "dislike", "awful", "bad", "terrible")):
            order = [labels[1], labels[2], labels[0]]  # negative > neutral > positive
        else:
            order = [labels[2], labels[0], labels[1]]  # neutral > positive > negative
        return {"labels": order, "scores": [0.9, 0.08, 0.02]}


@pytest.fixture(autouse=True)
def patch_env_and_loaders(monkeypatch):
    """
    - Empêche tout download NLTK / heavy loads.
    - Force l'usage de fakes pour BART et VADER via DI.
    """
    monkeypatch.setenv("ALLOW_NLTK_DOWNLOAD", "0")
    monkeypatch.setenv("BART_MNLI_FAKE", "1")
    monkeypatch.setattr(S, "get_vader", lambda: FakeVader(), raising=True)
    monkeypatch.setattr(S, "get_sentiment_pipeline", lambda: FakeBart(), raising=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tests core.py (détection & mapping)
# ──────────────────────────────────────────────────────────────────────────────

def test_detect_sentiment_vader_paths():
    assert (
            S.detect_sentiment("I love this product!", vader=FakeVader(), bart=FakeBart())
            == "positive"
    )
    assert S.detect_sentiment("I hate this color", vader=FakeVader(), bart=FakeBart()) == "negative"
    # Neutre → bascule sur BART (FakeBart renverra 'neutral' en top label)
    assert (
            S.detect_sentiment("It's okay, nothing special", vader=FakeVader(), bart=FakeBart())
            == "neutral"
    )


def test_map_sentiment_negation_overrides_to_negative():
    text = "I like pink but not red"
    assert S.map_sentiment("positive", text) == "negative"


def test_is_negated_or_soft_variants():
    hard, soft = S.is_negated_or_soft("not too shiny", debug=False)
    assert (hard, soft) == (False, True)   # soft-negation
    hard, soft = S.is_negated_or_soft("no lipstick please", debug=False)
    assert (hard, soft) == (True, False)   # hard-negation
    hard, soft = S.is_negated_or_soft("I am happy", debug=False)
    assert (hard, soft) == (False, False)


def test_analyze_sentence_sentiment_with_split_and_separators():
    sent = "I love matte, but I dislike glitter."
    res = S.analyze_sentence_sentiment(sent)
    assert isinstance(res, list) and len(res) >= 2
    assert all({"clause", "polarity", "separator"} <= set(r.keys()) for r in res)
    pols = [r["polarity"] for r in res]
    assert "positive" in pols and "negative" in pols


def test_classify_segments_by_sentiment_no_neutral_bucketing():
    segments = ["I love nude rose", "not too shiny", "I dislike glitter", "meh"]
    out = S.classify_segments_by_sentiment_no_neutral(has_splitter=False, segments=segments)
    assert set(out.keys()) == {"positive", "negative"}
    assert any("meh" in s for s in out["positive"])         # neutre → positive par défaut
    assert any("not too shiny" in s for s in out["negative"])  # soft-negation → negative


# ──────────────────────────────────────────────────────────────────────────────
# Tests router.py (résumé par sentiment)
# ──────────────────────────────────────────────────────────────────────────────

def _fake_aggregate_color_phrase_results(
    *, segments, known_modifiers, all_webcolor_names, llm_client, cache, debug
):
    matched_phrases = ["Dusty Rose", "nude rose", "dusty rose"]  # doublons volontairement
    local_rgb = {
        "dusty rose": (210, 150, 160),
        "nude rose": (205, 170, 165),
    }
    matched_tones = {"rose"}
    return matched_tones, matched_phrases, local_rgb


def _fake_choose_representative_rgb(rgb_map):
    if not rgb_map:
        return None
    key = sorted(rgb_map.keys())[0]
    return rgb_map[key]


def _euclid_rgb(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def test_build_color_sentiment_summary_happy_path(monkeypatch):
    monkeypatch.setattr(
        R, "aggregate_color_phrase_results",
        lambda **kwargs: _fake_aggregate_color_phrase_results(**kwargs),
        raising=True,
    )
    monkeypatch.setattr(
        R, "choose_representative_rgb", _fake_choose_representative_rgb, raising=True
    )
    monkeypatch.setattr(R, "rgb_distance", _euclid_rgb, raising=True)

    sentiment = "romantic"
    segments = ["I love dusty rose but not glitter"]
    known_tones = {"rose", "nude"}
    known_modifiers = {"dusty"}
    rgb_map = {"beige": (220, 210, 190)}
    base_rgb_by_sentiment = {}

    summary = R.build_color_sentiment_summary(
        sentiment=sentiment,
        segments=segments,
        known_tones=known_tones,
        known_modifiers=known_modifiers,
        rgb_map=rgb_map,
        base_rgb_by_sentiment=base_rgb_by_sentiment,
        debug=True,
    )

    names = summary["matched_color_names"]
    assert len(names) == 2
    assert (
            [s.casefold() for s in names] == ["dusty rose", "nude rose"]
    )  # ordre + dédup insensibles à la casse
    assert summary["base_rgb"] in {(210, 150, 160), (205, 170, 165)}
    assert isinstance(summary["threshold"], float) and 30.0 <= summary["threshold"] <= 80.0
    assert "dusty rose" in rgb_map and "nude rose" in rgb_map
    assert base_rgb_by_sentiment.get("romantic") == summary["base_rgb"]
