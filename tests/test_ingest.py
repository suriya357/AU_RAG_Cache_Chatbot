"""
test_ingest.py — Unit tests for Firecrawl-powered ingestion layer.
All tests use mocks to avoid making actual paid API calls.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import chunk_page, make_splitter
from exceptions import ConfigurationError, IngestionError
from firecrawl import (
    FirecrawlApp,
    FirecrawlError,
    RateLimitError,
    RequestTimeoutError,
    UnauthorizedError,
)
from firecrawl.v2.types import Document, DocumentMetadata
from ingest import (
    extract_scrape_data,
    get_firecrawl_app,
    ingest_url,
    run_ingestion,
    save_page,
    scrape_page,
    url_to_slug,
)


# ---- Fixtures -------------------------------------------------------------

@pytest.fixture
def mock_app():
    app = MagicMock()
    return app


@pytest.fixture
def sample_document():
    return Document(
        markdown="# Department of Information Technology\n\nAnna University offers undergraduate and postgraduate courses.",
        metadata=DocumentMetadata(
            title="IT Department - Anna University",
            source_url="https://departments.auegov.ac.in/it",
            status_code=200,
            description="Anna University IT Department official page",
        ),
    )


# ---- Configuration Tests --------------------------------------------------

def test_get_firecrawl_app_missing_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as exc_info:
        get_firecrawl_app()
    assert "FIRECRAWL_API_KEY not found" in str(exc_info.value)


def test_get_firecrawl_app_with_explicit_key():
    app = get_firecrawl_app(api_key="fc-test-key-12345")
    assert app is not None


# ---- Extraction & Parsing Tests -------------------------------------------

def test_extract_scrape_data_v2_document(sample_document):
    title, text, metadata = extract_scrape_data(
        sample_document, "https://departments.auegov.ac.in/it"
    )
    assert title == "IT Department - Anna University"
    assert "Anna University offers" in text
    assert metadata.get("source_url") == "https://departments.auegov.ac.in/it"


def test_extract_scrape_data_v1_dict():
    response = {
        "markdown": "# Courses Offered\n\nB.Tech Information Technology, M.Tech IT.",
        "metadata": {
            "title": "Courses Offered - AU IT",
            "sourceURL": "https://departments.auegov.ac.in/it/coursesoffered",
        },
    }
    title, text, metadata = extract_scrape_data(
        response, "https://departments.auegov.ac.in/it/coursesoffered"
    )
    assert title == "Courses Offered - AU IT"
    assert "B.Tech Information Technology" in text
    assert metadata["title"] == "Courses Offered - AU IT"


def test_extract_scrape_data_missing_title_fallback():
    doc_without_title = Document(
        markdown="# Some content without metadata title",
        metadata=DocumentMetadata(source_url="https://departments.auegov.ac.in/it/teaching"),
    )
    title, text, _ = extract_scrape_data(
        doc_without_title, "https://departments.auegov.ac.in/it/teaching"
    )
    assert title == "It Teaching"
    assert len(text) > 0


# ---- Scraping & Error Handling Tests --------------------------------------

def test_scrape_page_success_document(mock_app, sample_document):
    mock_app.scrape.return_value = sample_document
    url = "https://departments.auegov.ac.in/it"

    title, text, metadata = scrape_page(mock_app, url)

    assert title == "IT Department - Anna University"
    assert "Department of Information Technology" in text
    mock_app.scrape.assert_called_once_with(
        url=url, formats=["markdown"], only_main_content=True
    )


def test_scrape_page_empty_content_raises_ingestion_error(mock_app):
    empty_doc = Document(
        markdown="   ",
        metadata=DocumentMetadata(title="Empty Page"),
    )
    mock_app.scrape.return_value = empty_doc
    url = "https://departments.auegov.ac.in/it/empty"

    with pytest.raises(IngestionError) as exc_info:
        scrape_page(mock_app, url)

    assert "Empty content" in exc_info.value.reason
    assert exc_info.value.url == url


def test_scrape_page_rate_limit_error(mock_app):
    mock_app.scrape.side_effect = RateLimitError("Rate limit of 10 req/min exceeded")
    url = "https://departments.auegov.ac.in/it/coursesoffered"

    with pytest.raises(IngestionError) as exc_info:
        scrape_page(mock_app, url)

    assert "Rate limit" in exc_info.value.reason


def test_scrape_page_timeout_error(mock_app):
    mock_app.scrape.side_effect = RequestTimeoutError("Request timed out after 30s")
    url = "https://departments.auegov.ac.in/it/projects"

    with pytest.raises(IngestionError) as exc_info:
        scrape_page(mock_app, url)

    assert "timed out" in exc_info.value.reason


def test_scrape_page_unauthorized_error(mock_app):
    mock_app.scrape.side_effect = UnauthorizedError("Invalid API key")
    url = "https://departments.auegov.ac.in/it/announcements"

    with pytest.raises(IngestionError) as exc_info:
        scrape_page(mock_app, url)

    assert "Unauthorized" in exc_info.value.reason


# ---- Storage & Downstream Chunker Compatibility Tests ----------------------

def test_save_page_and_chunker_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr("ingest.OUTPUT_DIR", str(tmp_path))

    url = "https://departments.auegov.ac.in/it/coursesoffered"
    title = "Courses Offered - Anna University"
    text = (
        "## Undergraduate Programmes\n\n"
        "B.Tech Information Technology (4 years).\n"
        "B.Tech Artificial Intelligence and Data Science (4 years).\n\n"
        "## Postgraduate Programmes\n\n"
        "M.Tech Information Technology (2 years).\n"
    )
    metadata = {"status_code": 200, "source_url": url}

    filepath = save_page(url, title, text, metadata)
    assert os.path.exists(filepath)

    # Verify JSON file structure matches downstream chunker contract
    with open(filepath, "r", encoding="utf-8") as f:
        raw_record = json.load(f)

    assert raw_record["url"] == url
    assert raw_record["title"] == title
    assert raw_record["text"] == text
    assert raw_record["char_count"] == len(text)
    assert "scraped_at" in raw_record
    assert "firecrawl_metadata" in raw_record

    # Verify chunker can chunk this saved page without any modification
    splitter = make_splitter()
    chunks = chunk_page(raw_record, splitter)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["source_url"] == url
        assert chunk["source_title"] == title
        assert "chunk_id" in chunk
        assert "chunk_index" in chunk


def test_ingest_url_does_not_save_on_failure(mock_app, tmp_path, monkeypatch):
    monkeypatch.setattr("ingest.OUTPUT_DIR", str(tmp_path))
    mock_app.scrape.side_effect = FirecrawlError("Scrape service unavailable")

    url = "https://departments.auegov.ac.in/it/broken"
    result = ingest_url(mock_app, url)

    assert result is None
    # Ensure no file was generated for the failed URL
    expected_slug = url_to_slug(url)
    assert not os.path.exists(os.path.join(tmp_path, f"{expected_slug}.json"))


def test_run_ingestion_manifest_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("ingest.OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("ingest.DELAY_BETWEEN_REQUESTS", 0.0)

    mock_app = MagicMock()
    mock_app.scrape.return_value = Document(
        markdown="# Department Facilities\n\nHigh-performance labs and smart classrooms.",
        metadata=DocumentMetadata(title="Department Facilities"),
    )

    test_urls = [
        "https://departments.auegov.ac.in/it/deptfacilities",
        "https://departments.auegov.ac.in/it/studentsassociation",
    ]

    results = run_ingestion(urls=test_urls, app=mock_app)

    assert len(results) == 2
    manifest_path = os.path.join(tmp_path, "_manifest.json")
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert len(manifest_data) == 2
    assert manifest_data[0]["url"] == test_urls[0]
    assert manifest_data[1]["url"] == test_urls[1]
    assert "filepath" in manifest_data[0]
    assert "char_count" in manifest_data[0]
