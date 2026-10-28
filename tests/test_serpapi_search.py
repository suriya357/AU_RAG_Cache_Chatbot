"""
test_serpapi_search.py — Unit tests for SerpAPI live search fallback.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_web_search_missing_key_raises_configuration_error(monkeypatch):
    """web_search should raise ConfigurationError when SERPAPI_API_KEY is missing or default placeholder."""
    import router
    from exceptions import ConfigurationError

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        router.web_search("what is the weather today?")

    monkeypatch.setenv("SERPAPI_API_KEY", "your_serpapi_key_here")
    with pytest.raises(ConfigurationError):
        router.web_search("what is the weather today?")


def test_web_search_returns_formatted_chunks(monkeypatch):
    """web_search should transform organic_results into standard chunk dicts."""
    import router

    monkeypatch.setenv("SERPAPI_API_KEY", "test_serpapi_key")

    class MockSearch:
        def __init__(self, params):
            self.params = params

        def get_dict(self):
            return {
                "organic_results": [
                    {
                        "title": "Anna University IT Department",
                        "link": "https://departments.auegov.ac.in/it",
                        "snippet": "Department of Information Technology CEG campus details.",
                    },
                    {
                        "title": "Anna University News",
                        "link": "https://annauniv.edu/news",
                        "snippet": "Latest campus updates and announcements.",
                    },
                ]
            }

    monkeypatch.setattr(router, "GoogleSearch", MockSearch)

    chunks = router.web_search("Anna University IT", max_results=2)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "https://departments.auegov.ac.in/it"
    assert chunks[0]["source_url"] == "https://departments.auegov.ac.in/it"
    assert chunks[0]["source_title"] == "Anna University IT Department"
    assert "CEG campus" in chunks[0]["text"]
    assert chunks[1]["source_url"] == "https://annauniv.edu/news"


def test_web_search_api_error_raises_search_fallback_error(monkeypatch):
    """SerpAPI error response dictionary should trigger SearchFallbackError."""
    import router
    from exceptions import SearchFallbackError

    monkeypatch.setenv("SERPAPI_API_KEY", "test_serpapi_key")

    class MockSearchError:
        def __init__(self, params):
            pass

        def get_dict(self):
            return {"error": "Invalid API key or search quota exceeded"}

    monkeypatch.setattr(router, "GoogleSearch", MockSearchError)

    with pytest.raises(SearchFallbackError) as exc_info:
        router.web_search("query triggering error")

    assert "Invalid API key" in str(exc_info.value)
