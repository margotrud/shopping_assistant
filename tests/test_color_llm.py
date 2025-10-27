# tests/test_color_llm.py
import types

# Public facade
import color_sentiment_extractor.extraction.color.llm as llm

# Impl module (pour patcher _session et autres détails internes)
from color_sentiment_extractor.extraction.color.llm import llm_api_client as llm_impl


# ── Dummies ───────────────────────────────────────────────────────────────────
class DummyCache:
    def __init__(self):
        self.data = {}
    def get_rgb(self, key):
        return self.data.get(key)
    def store_rgb(self, key, value):
        self.data[key] = value


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="ok", headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.headers = headers or {}
    def json(self):
        return self._json


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_case_01_has_api_key_true_false(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert llm.has_api_key() is True
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.has_api_key() is False


def test_case_02_get_llm_client_ok(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = llm.get_llm_client()
    assert isinstance(client, llm.OpenRouterClient)


def test_case_03_get_llm_client_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.get_llm_client() is None


def test_case_04_build_color_prompt_contains_phrase_and_rgb():
    phrase = "dusty rose"
    prompt = llm.build_color_prompt(phrase)
    assert phrase in prompt
    assert "RGB" in prompt
    assert "→" in prompt  # format d’exemple


def test_case_05_query_llm_for_rgb_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.query_llm_for_rgb("dusty rose") is None


def test_case_06_query_llm_for_rgb_cache_hit_no_network(monkeypatch):
    # Empêche tout appel réseau : si appelé, on lève
    def raising_post(*args, **kwargs):
        raise AssertionError("Network call should not happen on cache hit")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm_impl, "_session", types.SimpleNamespace(post=raising_post))

    cache = DummyCache()
    cache.store_rgb("dusty rose", (231, 180, 188))
    result = llm.query_llm_for_rgb("dusty rose", cache=cache, retries=0)
    assert result == (231, 180, 188)


def test_case_07_query_llm_for_rgb_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def fake_post(url, headers, json, timeout):
        return DummyResponse(200, {"choices": [{"message": {"content": "(231, 180, 188)"}}]})

    monkeypatch.setattr(llm_impl, "_session", types.SimpleNamespace(post=fake_post))
    result = llm.query_llm_for_rgb("dusty rose", retries=0, debug=True)
    assert result == (231, 180, 188)


def test_case_08_query_llm_for_rgb_rate_limited_then_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    calls = {"n": 0}
    def fake_post(url, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResponse(429, text="rate limited", headers={"Retry-After": "0"})
        return DummyResponse(200, {"choices": [{"message": {"content": "(10, 20, 30)"}}]})

    # Neutralise time.sleep pour accélérer
    monkeypatch.setattr(llm_impl.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_impl, "_session", types.SimpleNamespace(post=fake_post))

    result = llm.query_llm_for_rgb("any phrase", retries=2, debug=True)
    assert result == (10, 20, 30)
    assert calls["n"] >= 2


def test_case_09_query_llm_for_rgb_non_200_gives_none_when_no_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def fake_post(url, headers, json, timeout):
        return DummyResponse(500, text="server error")

    monkeypatch.setattr(llm_impl.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_impl, "_session", types.SimpleNamespace(post=fake_post))

    assert llm.query_llm_for_rgb("deep lavender", retries=0, debug=True) is None


def test_case_10_query_llm_for_rgb_parse_fail_then_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    replies = iter([
        DummyResponse(200, {"choices": [{"message": {"content": "not a tuple"}}]}),
        DummyResponse(200, {"choices": [{"message": {"content": "(1, 2, 3)"}}]}),
    ])

    def fake_post(url, headers, json, timeout):
        return next(replies)

    monkeypatch.setattr(llm_impl.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_impl, "_session", types.SimpleNamespace(post=fake_post))

    # Avec retries=1, on tolère un échec de parsing puis un succès
    assert llm.query_llm_for_rgb("rosy nude", retries=1, debug=True) == (1, 2, 3)


def test_case_11_openrouterclient_simplify_sanitizes_and_limits_to_two_words(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def fake_post(self, messages):
        # Réponse volontairement bruyante : guillemets/ponctuation/plusieurs mots
        return '  "Dusty Rose with extra words!!!"  '

    monkeypatch.setattr(llm_impl.OpenRouterClient, "_post", fake_post)
    client = llm.OpenRouterClient()
    assert client.simplify("very dusty pinkish rose") == "dusty rose"


def test_case_12_openrouterclient_simplify_empty_on_uncertain(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def fake_post(self, messages):
        return ""  # incertain → chaîne vide attendue

    monkeypatch.setattr(llm_impl.OpenRouterClient, "_post", fake_post)
    client = llm.OpenRouterClient()
    assert client.simplify("???") == ""
