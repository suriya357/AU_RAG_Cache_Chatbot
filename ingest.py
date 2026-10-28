"""
ingest.py — Phase 1: Ingestion Layer (Firecrawl-powered)

Scrapes Anna University IT department pages using the Firecrawl SDK,
extracting clean Markdown content and structured metadata, and saves
each page as JSON compatible with chunker.py.

Usage:
    python ingest.py

Output:
    data/raw/<page_slug>.json   (one file per URL)
    data/raw/_manifest.json     (ingestion manifest)

Requires:
    FIRECRAWL_API_KEY in .env file (get a key at https://www.firecrawl.dev)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

try:
    from firecrawl import (
        FirecrawlApp,
        FirecrawlError,
        RateLimitError,
        RequestTimeoutError,
        UnauthorizedError,
    )
except ImportError:
    from firecrawl import FirecrawlApp
    FirecrawlError = Exception
    RateLimitError = Exception
    RequestTimeoutError = Exception
    UnauthorizedError = Exception

from exceptions import ConfigurationError, IngestionError
from logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# ---- Config -----------------------------------------------------------

URLS = [
    "https://departments.auegov.ac.in/it",
    "https://departments.auegov.ac.in/it/coursesoffered",
    "https://departments.auegov.ac.in/it/academicschedule",
    "https://departments.auegov.ac.in/it/teaching",
    "https://departments.auegov.ac.in/it/nonteaching",
    "https://departments.auegov.ac.in/it/deptfacilities",
    "https://departments.auegov.ac.in/it/studentsassociation",
    # --- Sibling pages discovered from site navigation ---
    "https://departments.auegov.ac.in/it/pgstudents",
    "https://departments.auegov.ac.in/it/ugstudents",
    "https://departments.auegov.ac.in/it/phdawarded",
    "https://departments.auegov.ac.in/it/journals",
    "https://departments.auegov.ac.in/it/conferences",
    "https://departments.auegov.ac.in/it/projects",
    "https://departments.auegov.ac.in/it/announcements",
]

OUTPUT_DIR = os.path.join("data", "raw")
DELAY_BETWEEN_REQUESTS = 1.0  # seconds between scrapes
MIN_CONTENT_CHARS = 40        # threshold to warn on unusually short page text


# ---- Core Functions ---------------------------------------------------

def get_firecrawl_app(api_key: str | None = None) -> FirecrawlApp:
    """
    Instantiate and return the FirecrawlApp client.
    Reads FIRECRAWL_API_KEY from environment unless explicitly supplied.
    Raises ConfigurationError if the key is missing.
    """
    key = api_key or os.getenv("FIRECRAWL_API_KEY")
    if not key:
        raise ConfigurationError(
            "FIRECRAWL_API_KEY not found in environment or .env file. "
            "Set FIRECRAWL_API_KEY=your_key_here (get a key at https://www.firecrawl.dev)"
        )
    return FirecrawlApp(api_key=key)


def url_to_slug(url: str) -> str:
    """Turn a URL into a filesystem-safe filename slug."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    return path if path else "index"


def extract_scrape_data(response: Any, url: str) -> tuple[str, str, dict]:
    """
    Safely extract title, markdown text, and metadata from Firecrawl response.
    Supports both Firecrawl v2 Document objects and v1 dictionary payloads.
    """
    if response is None:
        return "", "", {}

    if isinstance(response, dict):
        text = response.get("markdown") or response.get("content") or ""
        metadata = response.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = getattr(metadata, "__dict__", {})
        title = metadata.get("title") or metadata.get("ogTitle") or ""
    else:
        text = getattr(response, "markdown", None) or getattr(response, "content", None) or ""
        metadata_obj = getattr(response, "metadata", None)
        if hasattr(metadata_obj, "model_dump"):
            metadata = metadata_obj.model_dump()
        elif isinstance(metadata_obj, dict):
            metadata = metadata_obj
        elif hasattr(metadata_obj, "__dict__"):
            metadata = metadata_obj.__dict__
        else:
            metadata = {}
        title = (
            metadata.get("title")
            or metadata.get("og_title")
            or getattr(metadata_obj, "title", "")
            or ""
        )

    text = text.strip() if text else ""
    title = title.strip() if title else ""

    # Fallback to readable title from URL slug if title was not extracted
    if not title:
        slug = url_to_slug(url)
        title = slug.replace("_", " ").title()

    return title, text, metadata


def scrape_page(app: FirecrawlApp, url: str) -> tuple[str, str, dict]:
    """
    Scrape a single URL using Firecrawl with Markdown extraction.
    Raises IngestionError on API failure, timeout, rate limit, or empty content.
    """
    try:
        if hasattr(app, "scrape"):
            response = app.scrape(url=url, formats=["markdown"], only_main_content=True)
        elif hasattr(app, "scrape_url"):
            response = app.scrape_url(url, params={"formats": ["markdown"], "onlyMainContent": True})
        else:
            raise ConfigurationError("FirecrawlApp instance does not have a scrape or scrape_url method")
    except RateLimitError as e:
        logger.error(f"Firecrawl rate limit exceeded for {url}: {e}")
        raise IngestionError(url, f"Rate limit exceeded: {e}") from e
    except RequestTimeoutError as e:
        logger.error(f"Firecrawl request timed out for {url}: {e}")
        raise IngestionError(url, f"Request timed out: {e}") from e
    except UnauthorizedError as e:
        logger.error(f"Firecrawl unauthorized (check API key): {e}")
        raise IngestionError(url, f"Unauthorized: {e}") from e
    except FirecrawlError as e:
        logger.error(f"Firecrawl API error scraping {url}: {e}")
        raise IngestionError(url, f"Firecrawl API error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error scraping {url} via Firecrawl: {e}")
        raise IngestionError(url, str(e)) from e

    title, text, metadata = extract_scrape_data(response, url)

    if not text:
        msg = f"Empty content returned by Firecrawl for {url}"
        logger.error(msg)
        raise IngestionError(url, msg)

    if len(text) < MIN_CONTENT_CHARS:
        logger.warning(
            f"Extracted content is unusually short ({len(text)} chars) for {url}"
        )

    return title, text, metadata


def save_page(url: str, title: str, text: str, metadata: dict | None = None) -> str:
    """
    Save a single page's data as JSON matching the contract expected by chunker.py.
    Returns the output filepath.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = url_to_slug(url)
    filepath = os.path.join(OUTPUT_DIR, f"{slug}.json")

    record = {
        "url": url,
        "title": title,
        "text": text,
        "char_count": len(text),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    # Store sanitized Firecrawl metadata if available, without breaking consumers
    if metadata:
        record["firecrawl_metadata"] = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool, list, dict)) and v is not None
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return filepath


def ingest_url(app: FirecrawlApp, url: str) -> dict | None:
    """
    Scrape, process, and save a single URL.
    Returns the summary record on success, or None on failure.
    """
    try:
        logger.info(f"Ingesting: {url}")
        title, text, metadata = scrape_page(app, url)
        filepath = save_page(url, title, text, metadata)
        logger.info(f"Saved -> {filepath} ({len(text)} chars, title={title!r})")
        return {
            "url": url,
            "title": title,
            "char_count": len(text),
            "filepath": filepath,
        }
    except IngestionError as e:
        logger.error(f"Ingestion failed for {url}: {e.reason}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}")
        return None


def run_ingestion(urls: list[str] = URLS, app: FirecrawlApp | None = None) -> list[dict]:
    """
    Execute full ingestion over the provided URL list.
    Saves individual page JSONs and updates _manifest.json.
    """
    if app is None:
        app = get_firecrawl_app()

    logger.info(f"Starting Firecrawl ingestion of {len(urls)} URLs")
    results = []

    for i, url in enumerate(urls):
        result = ingest_url(app, url)
        if result:
            results.append(result)

        # Rate-limiting delay between requests (skip after the last item)
        if i < len(urls) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    logger.info(f"Ingestion complete: {len(results)}/{len(urls)} pages succeeded")

    manifest_path = os.path.join(OUTPUT_DIR, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved -> {manifest_path}")

    return results


def main():
    try:
        app = get_firecrawl_app()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info(
            "Please ensure FIRECRAWL_API_KEY is defined in your environment or .env file."
        )
        return

    run_ingestion(URLS, app)


if __name__ == "__main__":
    main()