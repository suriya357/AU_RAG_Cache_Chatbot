"""
reranker.py — Phase 3: Re-ranking Layer

Takes the top-K candidates from vector search (bi-encoder — fast but coarse)
and re-scores them with a cross-encoder (slow but precise) to surface the
most genuinely relevant chunks for the final answer.

Why two stages instead of one:
    - Bi-encoders (used in vectorstore.py) embed the query and each chunk
      SEPARATELY, then compare vectors. Fast, scales to millions of chunks,
      but loses some nuance since query and chunk never "see" each other
      directly.
    - Cross-encoders embed the query and chunk TOGETHER in one pass, so the
      model can directly compare them. Much more accurate, but too slow to
      run over an entire vector store — hence: use it only on the vector
      search's top 20 candidates, not all chunks.

Usage:
    python reranker.py --query "What courses does the IT department offer?"

Input:
    Candidates from vectorstore.query() (Phase 2)
Output:
    Top-N re-ranked chunks, ready for generation (Phase 4)
"""

import argparse
import logging

from sentence_transformers import CrossEncoder

import vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -------------------------------------------------------------

# Small, fast, well-regarded cross-encoder for re-ranking. Runs on CPU fine.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVE_TOP_K = 20   # candidates pulled from vector search before re-ranking
RERANK_TOP_N = 5      # final number of chunks kept after re-ranking

# A cross-encoder score below this suggests the KB probably doesn't have a
# good answer — used later by router.py to decide whether to fall back to
# live web search. Tune this after looking at real scores for your data.
CONFIDENCE_THRESHOLD = 0.0


# ---- Core functions -------------------------------------------------------

_model_cache = None


def get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder once and reuse it (loading is the slow part)."""
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        _model_cache = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model_cache


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Re-score vector search candidates with a cross-encoder.

    Args:
        query: the user's question
        candidates: list of dicts from vectorstore.query(), each with a "text" field
        top_n: how many top results to return after re-ranking

    Returns:
        candidates list, re-ordered and trimmed to top_n, each with an added
        "rerank_score" field (higher = more relevant)
    """
    if not candidates:
        return []

    model = get_cross_encoder()

    # Cross-encoder expects (query, document) pairs
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]


def retrieve_and_rerank(
    query: str,
    retrieve_k: int = RETRIEVE_TOP_K,
    rerank_n: int = RERANK_TOP_N,
    mode: str = "hybrid",
) -> list[dict]:
    """
    Full pipeline: hybrid search (vector + BM25) -> cross-encoder re-rank.
    This is the function later phases (generator.py, router.py) will call.
    """
    logger.info(f"Retrieving top {retrieve_k} candidates (mode={mode}) for: {query!r}")
    candidates = vectorstore.query(query, top_k=retrieve_k, mode=mode)

    if not candidates:
        logger.warning("No candidates returned from store")
        return []

    logger.info(f"Re-ranking {len(candidates)} candidates -> top {rerank_n}")
    reranked = rerank(query, candidates, top_n=rerank_n)

    return reranked


def is_confident(reranked_results: list[dict], threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """
    Quick heuristic: does the top result score high enough to trust the KB
    answer, or should router.py (Phase 5) fall back to live search?
    """
    if not reranked_results:
        return False
    return reranked_results[0]["rerank_score"] >= threshold


# ---- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test retrieval + re-ranking")
    parser.add_argument("--query", type=str, required=True, help="Question to test")
    parser.add_argument("--retrieve_k", type=int, default=RETRIEVE_TOP_K)
    parser.add_argument("--rerank_n", type=int, default=RERANK_TOP_N)
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "vector", "bm25"],
        help="Retrieval mode: hybrid (default), vector, or bm25",
    )
    args = parser.parse_args()

    results = retrieve_and_rerank(args.query, args.retrieve_k, args.rerank_n, mode=args.mode)

    print(f"\nTop {len(results)} re-ranked results (mode={args.mode}) for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        dist_str = f"{r.get('distance', 1.0):.4f}" if r.get("distance") is not None else "N/A"
        rrf_str = f" | rrf_score={r['rrf_score']:.5f}" if "rrf_score" in r else ""
        print(f"--- Rank {i} | rerank_score={r['rerank_score']:.4f} | distance={dist_str}{rrf_str} ---")
        print(f"Source: {r['source_title']} ({r['source_url']})")
        print(f"Text: {r['text'][:200]}...")
        print()

    print(f"Confident enough to trust KB answer: {is_confident(results)}")


if __name__ == "__main__":
    main()