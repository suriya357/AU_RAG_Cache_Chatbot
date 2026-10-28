"""
test_hybrid_search.py — Unit tests for BM25, Reciprocal Rank Fusion, and Hybrid Search.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectorstore import (
    tokenize,
    reciprocal_rank_fusion,
    hybrid_query,
    query,
    bm25_query,
)


def test_tokenize():
    """Verify tokenizer lowercases, removes punctuation, and splits words."""
    assert tokenize("B.Tech. in Information Technology!") == ["b", "tech", "in", "information", "technology"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_reciprocal_rank_fusion_boosting():
    """
    A chunk that appears in both vector results and BM25 results
    should receive an additive RRF score and rank above single-source matches.
    """
    vector_results = [
        {"chunk_id": "c1", "text": "Doc 1 text", "source_url": "url1", "distance": 0.2},
        {"chunk_id": "c2", "text": "Doc 2 text", "source_url": "url2", "distance": 0.3},
    ]

    bm25_results = [
        {"chunk_id": "c3", "text": "Doc 3 text", "source_url": "url3", "bm25_score": 9.5},
        {"chunk_id": "c2", "text": "Doc 2 text", "source_url": "url2", "bm25_score": 8.0},
    ]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=5, rrf_k=60)

    # c2 is rank 2 in vector (score 1/62) and rank 2 in bm25 (score 1/62) -> sum = 2/62 = ~0.032258
    # c1 is rank 1 in vector (1/61 = ~0.016393)
    # c3 is rank 1 in bm25 (1/61 = ~0.016393)
    assert fused[0]["chunk_id"] == "c2"
    assert fused[0]["vector_rank"] == 2
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["bm25_score"] == 8.0
    assert fused[0]["distance"] == 0.3
    assert abs(fused[0]["rrf_score"] - (1 / 62 + 1 / 62)) < 1e-6


def test_reciprocal_rank_fusion_top_k():
    """Ensure RRF limits output to top_k."""
    vec = [{"chunk_id": f"v{i}", "text": f"text {i}"} for i in range(10)]
    bm = [{"chunk_id": f"b{i}", "text": f"text {i}"} for i in range(10)]
    fused = reciprocal_rank_fusion(vec, bm, top_k=3, rrf_k=60)
    assert len(fused) == 3


def test_bm25_query_scores_and_ranks(monkeypatch):
    """BM25 query should return relevant chunks matching query terms."""
    mock_chunks = [
        {"chunk_id": "ch1", "text": "Anna University IT department faculty and staff", "source_url": "u1", "source_title": "t1"},
        {"chunk_id": "ch2", "text": "Sports ground and cafeteria campus facilities", "source_url": "u2", "source_title": "t2"},
        {"chunk_id": "ch3", "text": "Undergraduate B.Tech IT department curriculum and courses", "source_url": "u3", "source_title": "t3"},
    ]

    monkeypatch.setattr("vectorstore.load_chunks", lambda path: mock_chunks)
    monkeypatch.setattr("vectorstore._bm25_index", None)
    monkeypatch.setattr("vectorstore._bm25_chunks", None)
    monkeypatch.setattr("vectorstore._bm25_mtime", None)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    results = bm25_query("curriculum B.Tech courses", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "ch3"
    assert results[0]["bm25_rank"] == 1
    assert results[0]["bm25_score"] > 0


def test_hybrid_query_fallback_branches(monkeypatch):
    """If one branch fails or returns empty, hybrid query falls back gracefully to the other."""
    v_results = [{"chunk_id": "v1", "text": "vector doc", "source_url": "u1", "distance": 0.1}]
    b_results = [{"chunk_id": "b1", "text": "bm25 doc", "source_url": "u2", "bm25_score": 5.0}]

    # Case 1: BM25 returns empty -> returns vector results
    monkeypatch.setattr("vectorstore.vector_query", lambda q, top_k: v_results)
    monkeypatch.setattr("vectorstore.bm25_query", lambda q, top_k: [])
    res = hybrid_query("test query", top_k=5)
    assert len(res) == 1
    assert res[0]["chunk_id"] == "v1"

    # Case 2: Vector returns empty -> returns BM25 results
    monkeypatch.setattr("vectorstore.vector_query", lambda q, top_k: [])
    monkeypatch.setattr("vectorstore.bm25_query", lambda q, top_k: b_results)
    res = hybrid_query("test query", top_k=5)
    assert len(res) == 1
    assert res[0]["chunk_id"] == "b1"


def test_query_mode_dispatch(monkeypatch):
    """Test mode selection in unified query()."""
    monkeypatch.setattr("vectorstore.hybrid_query", lambda q, top_k: [{"mode": "hybrid"}])
    monkeypatch.setattr("vectorstore.vector_query", lambda q, top_k: [{"mode": "vector"}])
    monkeypatch.setattr("vectorstore.bm25_query", lambda q, top_k: [{"mode": "bm25"}])

    assert query("test", mode="hybrid") == [{"mode": "hybrid"}]
    assert query("test", mode="vector") == [{"mode": "vector"}]
    assert query("test", mode="bm25") == [{"mode": "bm25"}]

    with pytest.raises(ValueError):
        query("test", mode="invalid_mode")
