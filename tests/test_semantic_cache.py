"""
test_semantic_cache.py — Unit tests for the semantic caching layer.

The real `redis` / `redisvl` packages are never required to run these
tests: we inject lightweight fakes into sys.modules that behave enough
like the real thing (store/check/clear, a distance-based match) to
exercise semantic_cache.py's own logic — connection handling, fail-open
behavior, and the check()/store() contract that router.py relies on.

Covers the four cache scenarios called out in the project requirements:
    1. An exact repeat of a previously asked question -> cache HIT.
    2. A semantically/lexically similar (but not identical) question
       -> cache HIT.
    3. An unrelated question -> cache MISS.
    4. Redis being unavailable -> graceful fallback (no crash, no cache).
"""

import re
import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ---- Fakes standing in for redis / redisvl ---------------------------------

class FakeRedisClient:
    """Stands in for a connected redis.Redis client."""

    def __init__(self, raise_on_ping: Exception | None = None):
        self._raise_on_ping = raise_on_ping

    def ping(self):
        if self._raise_on_ping:
            raise self._raise_on_ping
        return True


def _naive_distance(a: str, b: str) -> float:
    """
    Word-overlap based stand-in for cosine distance: 0.0 = identical word
    sets, 1.0 = no overlap at all. Good enough to distinguish "exact",
    "similar", and "unrelated" without needing real embeddings.
    """
    a_words = _words(a)
    b_words = _words(b)
    if not a_words or not b_words:
        return 1.0
    overlap = len(a_words & b_words) / len(a_words | b_words)
    return 1.0 - overlap


class FakeSemanticCache:
    """Stands in for redisvl.extensions.cache.llm.SemanticCache."""

    def __init__(self, name=None, vectorizer=None, redis_client=None, distance_threshold=0.15):
        self.distance_threshold = distance_threshold
        self._entries = []
        self.cleared = False

    def set_ttl(self, seconds):
        self.ttl = seconds

    def store(self, prompt, response, metadata=None):
        self._entries.append({"prompt": prompt, "response": response, "metadata": metadata or {}})

    def check(self, prompt):
        if not self._entries:
            return []
        best_entry, best_distance = None, 1.0
        for entry in self._entries:
            d = _naive_distance(prompt, entry["prompt"])
            if d < best_distance:
                best_distance, best_entry = d, entry
        if best_entry is None or best_distance > self.distance_threshold:
            return []
        return [{
            "prompt": best_entry["prompt"],
            "response": best_entry["response"],
            "vector_distance": best_distance,
            "metadata": best_entry["metadata"],
        }]

    def clear(self):
        self._entries = []
        self.cleared = True


class FakeVectorizer:
    def __init__(self, model=None, **kwargs):
        self.model = model


def _install_fake_redis_stack(monkeypatch, ping_error: Exception | None = None):
    """
    Injects fake `redis` and `redisvl` packages into sys.modules so
    semantic_cache._try_init()'s imports succeed and use our fakes.
    """
    fake_redis = types.ModuleType("redis")

    class _RedisNamespace:
        @staticmethod
        def from_url(url, **kwargs):
            return FakeRedisClient(raise_on_ping=ping_error)

    fake_redis.Redis = _RedisNamespace

    fake_redisvl = types.ModuleType("redisvl")
    fake_redisvl_extensions = types.ModuleType("redisvl.extensions")
    fake_redisvl_extensions_cache = types.ModuleType("redisvl.extensions.cache")
    fake_redisvl_extensions_cache_llm = types.ModuleType("redisvl.extensions.cache.llm")
    fake_redisvl_extensions_cache_llm.SemanticCache = FakeSemanticCache
    fake_redisvl_utils = types.ModuleType("redisvl.utils")
    fake_redisvl_utils_vectorize = types.ModuleType("redisvl.utils.vectorize")
    fake_redisvl_utils_vectorize.HFTextVectorizer = FakeVectorizer

    modules = {
        "redis": fake_redis,
        "redisvl": fake_redisvl,
        "redisvl.extensions": fake_redisvl_extensions,
        "redisvl.extensions.cache": fake_redisvl_extensions_cache,
        "redisvl.extensions.cache.llm": fake_redisvl_extensions_cache_llm,
        "redisvl.utils": fake_redisvl_utils,
        "redisvl.utils.vectorize": fake_redisvl_utils_vectorize,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def reset_cache_module_state(monkeypatch):
    """
    semantic_cache caches its connection state in module-level globals so
    it only tries to connect once per process. Reset that state before
    every test so each test gets a fresh, independent attempt.
    """
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "_cache", None)
    monkeypatch.setattr(semantic_cache, "_init_attempted", False)
    monkeypatch.setattr(semantic_cache, "_available", False)
    monkeypatch.setattr(semantic_cache, "CACHE_ENABLED", True)
    monkeypatch.setattr(semantic_cache, "DISTANCE_THRESHOLD", 0.5)
    yield


# ---- Tests -------------------------------------------------------------

def test_cache_disabled_via_env_never_connects_and_always_misses(monkeypatch):
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "CACHE_ENABLED", False)
    _install_fake_redis_stack(monkeypatch)  # would succeed if attempted

    assert semantic_cache.is_available() is False
    assert semantic_cache.check("What courses does the IT department offer?") is None

    # store() must be a silent no-op, not an error
    semantic_cache.store("some question", "some answer", ["https://example.com"])


def test_redis_unavailable_falls_back_gracefully(monkeypatch):
    """Requirement: Redis failure -> normal pipeline still works."""
    import semantic_cache

    _install_fake_redis_stack(monkeypatch, ping_error=ConnectionError("Redis is down"))

    assert semantic_cache.is_available() is False
    assert semantic_cache.check("What courses does the IT department offer?") is None

    # store() must not raise even though Redis is unreachable
    semantic_cache.store("What courses does the IT department offer?", "Some answer", [])


def test_redisvl_not_installed_falls_back_gracefully(monkeypatch):
    """If redis/redisvl aren't installed at all, the cache must fail open."""
    import semantic_cache

    # Make the imports inside _try_init() fail like the packages are missing
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redisvl.extensions.cache.llm", None)

    assert semantic_cache.is_available() is False
    assert semantic_cache.check("any question") is None


def test_exact_repeat_question_is_a_cache_hit(monkeypatch):
    import semantic_cache

    _install_fake_redis_stack(monkeypatch)
    question = "What courses does the IT department offer?"

    assert semantic_cache.check(question) is None  # nothing cached yet
    semantic_cache.store(question, "The department offers AI, ML, and Cybersecurity.", ["https://au.edu/it"])

    hit = semantic_cache.check(question)
    assert hit is not None
    assert hit["answer"] == "The department offers AI, ML, and Cybersecurity."
    assert hit["sources"] == ["https://au.edu/it"]
    assert hit["distance"] == 0.0


def test_semantically_similar_question_is_a_cache_hit(monkeypatch):
    import semantic_cache

    _install_fake_redis_stack(monkeypatch)
    original = "What courses does the IT department offer?"
    paraphrase = "What courses are offered by the IT department?"

    semantic_cache.store(original, "AI, ML, and Cybersecurity courses.", ["https://au.edu/it"])

    hit = semantic_cache.check(paraphrase)
    assert hit is not None
    assert hit["matched_question"] == semantic_cache._normalize(original)
    assert hit["answer"] == "AI, ML, and Cybersecurity courses."


def test_unrelated_question_is_a_cache_miss(monkeypatch):
    import semantic_cache

    _install_fake_redis_stack(monkeypatch)
    semantic_cache.store(
        "What courses does the IT department offer?",
        "AI, ML, and Cybersecurity courses.",
        ["https://au.edu/it"],
    )

    miss = semantic_cache.check("What is the capital of France?")
    assert miss is None


def test_store_is_noop_for_empty_answer(monkeypatch):
    import semantic_cache

    _install_fake_redis_stack(monkeypatch)
    semantic_cache.store("some question", "", [])

    assert semantic_cache.check("some question") is None


def test_clear_empties_the_cache(monkeypatch):
    import semantic_cache

    _install_fake_redis_stack(monkeypatch)
    semantic_cache.store("q1", "a1", [])
    semantic_cache.clear()

    assert semantic_cache.check("q1") is None
