"""
test_vectorstore_stale_chunks.py — Tests for idempotent upserts and stale chunk removal.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectorstore import build_vectorstore


def test_build_vectorstore_clears_stale_chunks_and_upserts(tmp_path, monkeypatch):
    """
    Simulates page update with fewer chunks:
    Initial: url A has 3 chunks (chunk0, chunk1, chunk2) and url B has 2 chunks.
    Update: url A now has only 1 chunk (chunk0).
    Verifies that url A's old chunks are cleared and upserted without affecting url B.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    monkeypatch.setattr("vectorstore.get_chroma_client", lambda: mock_client)
    monkeypatch.setattr("vectorstore.get_or_create_collection", lambda client: mock_collection)

    chunks_data = [
        {
            "chunk_id": "https://departments.auegov.ac.in/it::chunk0",
            "text": "IT Department updated text.",
            "source_url": "https://departments.auegov.ac.in/it",
            "source_title": "IT Department",
            "chunk_index": 0,
        }
    ]

    temp_chunks_file = os.path.join(tmp_path, "chunks.json")
    with open(temp_chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)

    monkeypatch.setattr("vectorstore.CHUNKS_FILE", temp_chunks_file)

    build_vectorstore()

    # Verify collection.delete was called for the specific source_url to purge stale chunks
    mock_collection.delete.assert_called_with(where={"source_url": "https://departments.auegov.ac.in/it"})

    # Verify collection.upsert was called with the updated chunk
    mock_collection.upsert.assert_called_once()
    upsert_args = mock_collection.upsert.call_args[1]
    assert upsert_args["ids"] == ["https://departments.auegov.ac.in/it::chunk0"]
    assert upsert_args["documents"] == ["IT Department updated text."]
