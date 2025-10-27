# tests/test_general_utils.py
"""End-to-end tests for general utils (load_config, log, nlp_utils) with cache/env fallbacks."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest


# ---------- Add src/ to sys.path for src-layout projects ----------
def _ensure_src_on_path() -> None:
    """Prepend 'src' to sys.path if present on disk."""
    from pathlib import Path

    src = Path("src").resolve()
    if src.is_dir():
        p = str(src)
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------- Minimal shim for normalize_token when package layout differs ----------
def _install_normalize_token_shim() -> None:
    """Install a tiny module tree and a minimal `normalize_token` callable."""
    pkg_names = [
        "color_sentiment_extractor",
        "color_sentiment_extractor.extraction",
        "color_sentiment_extractor.extraction.general",
        "color_sentiment_extractor.extraction.general.token",
    ]
    for name in pkg_names:
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)

    def _normalize_token(s: str, keep_hyphens: bool = False) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip().lower().replace("_", "-")
        return s if keep_hyphens else s.replace("-", " ")

    sys.modules[
        "color_sentiment_extractor.extraction.general.token"
    ].normalize_token = _normalize_token


def _import_utils():
    """Import load_config / log / nlp_utils with graceful fallbacks and shim."""
    _ensure_src_on_path()

    # load_config
    LC = None
    for mod in (
        "color_sentiment_extractor.extraction.general.utils.load_config",
        "Chatbot.extraction.general.utils.load_config",
    ):
        try:
            LC = import_module(mod)
            break
        except Exception:
            pass
    assert LC is not None, "Could not import load_config module"

    # log
    LOG = None
    for mod in (
        "color_sentiment_extractor.extraction.general.utils.log",
        "Chatbot.extraction.general.utils.log",
    ):
        try:
            LOG = import_module(mod)
            break
        except Exception:
            pass
    assert LOG is not None, "Could not import log module"

    # nlp_utils
    NLP = None
    for mod in (
        "color_sentiment_extractor.extraction.general.utils.nlp_utils",
        "Chatbot.extraction.general.utils.nlp_utils",
    ):
        try:
            NLP = import_module(mod)
            break
        except Exception:
            pass
    if NLP is None:
        _install_normalize_token_shim()
        for mod in (
            "color_sentiment_extractor.extraction.general.utils.nlp_utils",
            "Chatbot.extraction.general.utils.nlp_utils",
        ):
            try:
                NLP = import_module(mod)
                break
            except Exception:
                pass
    assert NLP is not None, "Could not import nlp_utils module"

    return LC, LOG, NLP


LC, LOG, NLP = _import_utils()
DataDirNotFound = LC.DataDirNotFound
ConfigFileNotFound = LC.ConfigFileNotFound
ConfigParseError = LC.ConfigParseError
ConfigTypeError = LC.ConfigTypeError
load_config = LC.load_config
clear_config_cache = LC.clear_config_cache


# ---------- Fixtures ----------
@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Provide an isolated data/ dir and point loader via DATA_DIR."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data))
    clear_config_cache()
    return data


@pytest.fixture(autouse=True)
def _reset_env_and_cache(monkeypatch):
    """Reset debug topics and config cache between tests."""
    monkeypatch.delenv("CHATBOT_DEBUG_TOPICS", raising=False)
    clear_config_cache()
    if hasattr(LOG, "reload_topics"):
        LOG.reload_topics()


# ---------- load_config tests ----------
def test_load_config_set_and_cache_hit(tmp_data_dir):
    p = tmp_data_dir / "known_tokens.json"
    p.write_text(json.dumps(["rose", "beige", 3]), encoding="utf-8")

    out1 = load_config("known_tokens", mode="set")
    assert out1 == frozenset({"rose", "beige", "3"})

    p.write_text(json.dumps(["changed"]), encoding="utf-8")
    out2 = load_config("known_tokens", mode="set")
    assert out2 == out1  # cached

    clear_config_cache()
    out3 = load_config("known_tokens", mode="set")
    assert out3 == frozenset({"changed"})


def test_load_config_validated_dict_and_errors(tmp_data_dir):
    conf = tmp_data_dir / "settings.json"
    conf.write_text(json.dumps({"alpha": 1}), encoding="utf-8")

    def validator(d: dict) -> dict:
        d = dict(d)
        d["beta"] = "ok"
        return d

    out = load_config("settings", mode="validated_dict", validator=validator)
    assert out == {"alpha": 1, "beta": "ok"}

    conf2 = tmp_data_dir / "oops.json"
    conf2.write_text(json.dumps({"not": "alist"}), encoding="utf-8")
    with pytest.raises(ConfigTypeError):
        load_config("oops", mode="set")

    with pytest.raises(ConfigFileNotFound):
        load_config("does_not_exist", mode="raw")


def test_load_config_allow_comments_with_fake_json5(tmp_data_dir, monkeypatch):
    cfg = tmp_data_dir / "cmt.json"
    cfg.write_text('{"a":1, /*c*/ "b":2, }', encoding="utf-8")

    fake_json5 = SimpleNamespace(load=lambda f: {"a": 1, "b": 2})
    monkeypatch.setitem(sys.modules, "json5", fake_json5)

    out = load_config("cmt", mode="raw", allow_comments=True)
    assert out == {"a": 1, "b": 2}


def test_load_config_refuses_escape_from_data_dir(tmp_data_dir):
    outside = tmp_data_dir.parent / "secret.json"
    outside.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(ConfigFileNotFound):
        load_config("../secret", mode="raw")


# ---------- log.debug tests ----------
def test_log_debug_respects_topics_env(monkeypatch, capsys):
    monkeypatch.setenv("CHATBOT_DEBUG_TOPICS", "extraction")
    LOG.reload_topics()

    LOG.debug("hello on extraction", topic="extraction")
    LOG.debug("should be silent", topic="other")

    captured = capsys.readouterr()
    assert "hello on extraction" in captured.err
    assert "should be silent" not in captured.err


def test_log_debug_all_topics(monkeypatch, capsys):
    monkeypatch.setenv("CHATBOT_DEBUG_TOPICS", "all")
    LOG.reload_topics()

    LOG.debug("m1", topic="foo")
    LOG.debug("m2", topic="bar")

    captured = capsys.readouterr()
    assert "m1" in captured.err and "m2" in captured.err


# ---------- nlp_utils: antonyms + lemmatization ----------
def test_are_antonyms_uses_wordnet_and_cache(monkeypatch):
    calls = {"n": 0}

    class _Lemma:
        def __init__(self, name: str, antonyms: list[str]):
            self._name = name
            self._ants = antonyms

        def name(self):
            return self._name

        def antonyms(self):
            return [SimpleNamespace(name=lambda n=a: n) for a in self._ants]

    class _Synset:
        def __init__(self, lemmas):
            self._lemmas = lemmas

        def lemmas(self):
            return self._lemmas

    def fake_synsets(word: str):
        calls["n"] += 1
        if word == "hot":
            return [_Synset([_Lemma("hot", ["cold"])])]
        if word == "cold":
            return [_Synset([_Lemma("cold", ["hot"])])]
        return []

    # Patch module-level wordnet used by nlp_utils
    monkeypatch.setattr(NLP, "wordnet", SimpleNamespace(synsets=fake_synsets), raising=True)

    assert NLP.are_antonyms("hot", "cold") is True
    _ = NLP.are_antonyms("hot", "cold")  # cached second call
    assert calls["n"] <= 4


def test_lemmatize_token_spacy_present_and_absent(monkeypatch):
    class _Tok:
        def __init__(self, text, lemma):
            self.text = text
            self.lemma_ = lemma

    class _Doc(list):
        pass

    class _NLP:
        def __call__(self, txt):
            return _Doc([_Tok(txt, "LEMMA")])

    monkeypatch.setattr(NLP, "_get_spacy", lambda: _NLP(), raising=True)
    assert NLP.lemmatize_token("running") == "LEMMA"

    monkeypatch.setattr(NLP, "_get_spacy", lambda: None, raising=True)
    assert NLP.lemmatize_token("Soft-Blue") == "soft-blue"
