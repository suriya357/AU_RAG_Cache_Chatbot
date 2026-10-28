"""
vectorstore.py — Phase 2: Vector Store Layer

Embeds chunks (from chunker.py output) and stores them in a persistent
ChromaDB collection. Also exposes a query() function used by later phases
(re-ranking, routing).

Usage (build the store from scratch):
    python vectorstore.py --build

Usage (quick manual test query):
    python vectorstore.py --query "What courses does the IT department offer?"

Input:
    data/chunks/chunks.json   (produced by chunker.py)
Output:
    data/chroma_db/           (persistent ChromaDB directory)
"""

import argparse
import json
import logging
import os
import re

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from exceptions import VectorStoreError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---- Config -------------------------------------------------------------

CHUNKS_FILE = os.path.join("data", "chunks", "chunks.json")
CHROMA_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "au_it_department"

# Free, local embedding model — no API key, no cost, runs on CPU.
# 'all-MiniLM-L6-v2' is small/fast; swap for 'bge-large-en-v1.5' if you
# want higher quality and don't mind slower embedding.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 20  # how many candidates to pull before re-ranking (Phase 3)


# ---- Core functions -------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(chunks_file: str) -> list[dict]:
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"{chunks_file} not found — run chunker.py first"
        )
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vectorstore():
    """Embed all chunks and add them to ChromaDB. Safe to re-run (upserts)."""
    logger.info("Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_url": c["source_url"],
            "source_title": c["source_title"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # Remove stale chunks for the source URLs being rebuilt so updated pages
    # with fewer chunks do not leave orphaned chunks in the vector store.
    source_urls = list({c["source_url"] for c in chunks if c.get("source_url")})
    for url in source_urls:
        try:
            collection.delete(where={"source_url": url})
        except Exception as e:
            logger.debug(f"No existing chunks to clear for {url}: {e}")

    # Use upsert by id for idempotent insertions without duplicate key errors
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        logger.info(f"Embedded batch {i // batch_size + 1} ({min(i + batch_size, len(ids))}/{len(ids)})")

    logger.info(f"Vector store built. Collection count: {collection.count()}")
    # Warm up or refresh the in-memory BM25 index
    get_bm25_index()


# ---- BM25 Keyword Search Layer ------------------------------------------

_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None
_bm25_mtime: float | None = None


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    return re.findall(r"\w+", text.lower()) if text else []


def get_bm25_index() -> tuple[BM25Okapi | None, list[dict]]:
    """
    Lazily build or refresh the BM25 index over data/chunks/chunks.json.
    Caches the index in memory and auto-refreshes if chunks.json has changed.
    """
    global _bm25_index, _bm25_chunks, _bm25_mtime
    if not os.path.exists(CHUNKS_FILE):
        logger.warning(f"{CHUNKS_FILE} not found — run chunker.py first")
        return None, []

    try:
        current_mtime = os.path.getmtime(CHUNKS_FILE)
    except OSError:
        current_mtime = None

    if _bm25_index is None or _bm25_mtime != current_mtime:
        chunks = load_chunks(CHUNKS_FILE)
        if not chunks:
            _bm25_index = None
            _bm25_chunks = []
        else:
            corpus = [tokenize(c.get("text", "")) for c in chunks]
            _bm25_index = BM25Okapi(corpus)
            _bm25_chunks = chunks
        _bm25_mtime = current_mtime

    return _bm25_index, _bm25_chunks or []


def bm25_query(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Query the corpus using BM25 sparse keyword matching.
    Returns candidate chunks sorted by BM25 score descending.
    """
    try:
        index, chunks = get_bm25_index()
        if not index or not chunks:
            return []

        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scores = index.get_scores(query_tokens)
        scored_indices = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score > 0.0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for rank, (idx, score) in enumerate(scored_indices[:top_k], 1):
            c = chunks[idx]
            candidates.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source_url": c.get("source_url"),
                "source_title": c.get("source_title"),
                "distance": 1.0,  # placeholder distance for downstream compatibility
                "bm25_score": score,
                "bm25_rank": rank,
            })

        return candidates

    except Exception as e:
        raise VectorStoreError("bm25_query", str(e))


def vector_query(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Query the dense vector store (ChromaDB). Returns candidate chunks with
    cosine distances (lower distance = higher semantic similarity).
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        if collection.count() == 0:
            logger.warning("Vector store is empty — did you run vectorstore.py --build?")
            return []

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
        )

        candidates = []
        for i in range(len(results["ids"][0])):
            candidates.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source_url": results["metadatas"][0][i].get("source_url"),
                "source_title": results["metadatas"][0][i].get("source_title"),
                "distance": results["distances"][0][i],
                "vector_rank": i + 1,
            })

        return candidates

    except Exception as e:
        raise VectorStoreError("vector_query", str(e))


# ---- Reciprocal Rank Fusion (RRF) & Hybrid Search -----------------------

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Fuse dense vector candidates and sparse BM25 candidates using
    Reciprocal Rank Fusion (RRF):
        RRF_score(d) = SUM( 1 / (rrf_k + rank(d)) )

    Chunks appearing in both vector and keyword search receive an additive boost.
    """
    fused: dict[str, dict] = {}

    for rank, item in enumerate(vector_results, 1):
        cid = item["chunk_id"]
        score = 1.0 / (rrf_k + rank)
        fused[cid] = {
            "chunk_id": cid,
            "text": item["text"],
            "source_url": item.get("source_url"),
            "source_title": item.get("source_title"),
            "distance": item.get("distance", 1.0),
            "bm25_score": 0.0,
            "vector_rank": rank,
            "bm25_rank": None,
            "rrf_score": score,
        }

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        score = 1.0 / (rrf_k + rank)
        if cid in fused:
            fused[cid]["rrf_score"] += score
            fused[cid]["bm25_score"] = item.get("bm25_score", 0.0)
            fused[cid]["bm25_rank"] = rank
        else:
            fused[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "source_url": item.get("source_url"),
                "source_title": item.get("source_title"),
                "distance": item.get("distance", 1.0),
                "bm25_score": item.get("bm25_score", 0.0),
                "vector_rank": None,
                "bm25_rank": rank,
                "rrf_score": score,
            }

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]


def hybrid_query(question: str, top_k: int = DEFAULT_TOP_K, rrf_k: int = 60) -> list[dict]:
    """
    Perform hybrid retrieval (Dense Vector + Sparse BM25 + RRF):
    Pulls up to top_k * 2 candidates from each branch and fuses them.
    Fails open to the surviving branch if one encounters an error or empty result.
    """
    pull_k = min(top_k * 2, 50)

    try:
        v_results = vector_query(question, top_k=pull_k)
    except Exception as e:
        logger.warning(f"Vector search branch failed in hybrid query: {e}")
        v_results = []

    try:
        b_results = bm25_query(question, top_k=pull_k)
    except Exception as e:
        logger.warning(f"BM25 search branch failed in hybrid query: {e}")
        b_results = []

    if not v_results and not b_results:
        logger.warning("Both vector and BM25 searches returned no results.")
        return []

    if not v_results:
        return b_results[:top_k]
    if not b_results:
        return v_results[:top_k]

    return reciprocal_rank_fusion(v_results, b_results, top_k=top_k, rrf_k=rrf_k)


def query(question: str, top_k: int = DEFAULT_TOP_K, mode: str = "hybrid") -> list[dict]:
    """
    Unified retrieval interface:
        mode="hybrid" -> Vector + BM25 with Reciprocal Rank Fusion (Default)
        mode="vector" -> Dense ChromaDB vector search only
        mode="bm25"   -> Sparse BM25 keyword search only
    """
    if mode == "vector":
        return vector_query(question, top_k=top_k)
    elif mode == "bm25":
        return bm25_query(question, top_k=top_k)
    elif mode == "hybrid":
        return hybrid_query(question, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}. Must be 'hybrid', 'vector', or 'bm25'.")


# ---- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build or query the RAG vector & hybrid store")
    parser.add_argument("--build", action="store_true", help="Embed chunks and build the vector store")
    parser.add_argument("--query", type=str, help="Run a test query against the store")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to show for --query")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "vector", "bm25"],
        help="Retrieval mode: hybrid (default), vector, or bm25",
    )
    args = parser.parse_args()

    if args.build:
        build_vectorstore()
    elif args.query:
        results = query(args.query, top_k=args.top_k, mode=args.mode)
        print(f"\nTop {len(results)} results (mode={args.mode}) for: {args.query!r}\n")
        for i, r in enumerate(results, 1):
            scores_info = []
            if "rrf_score" in r:
                scores_info.append(f"rrf_score={r['rrf_score']:.5f}")
            if r.get("vector_rank") is not None:
                scores_info.append(f"vec_rank={r['vector_rank']} (dist={r.get('distance', 1.0):.4f})")
            if r.get("bm25_rank") is not None:
                scores_info.append(f"bm25_rank={r['bm25_rank']} (score={r.get('bm25_score', 0.0):.2f})")
            scores_str = " | ".join(scores_info) or f"distance={r.get('distance', 1.0):.4f}"

            print(f"--- Result {i} [{scores_str}] ---")
            print(f"Source: {r['source_title']} ({r['source_url']})")
            print(f"Text: {r['text'][:200]}...")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()