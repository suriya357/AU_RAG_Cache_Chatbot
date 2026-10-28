"""
router.py — Phase 5: Agentic Routing Layer (now cache-aware)

This is the "agentic" part of the system: after trying to answer from the
knowledge base, the agent decides whether that answer is trustworthy, and
if not, autonomously falls back to a live web search instead of just
returning a bad or empty answer.

Phase 9 — Semantic Caching:
    Before ANY of that happens, route_and_answer() checks the semantic
    cache (semantic_cache.py, backed by Redis + RedisVL) for a previously
    answered, semantically similar question:

        User Query -> semantic_cache.check()
            HIT  -> return the cached answer immediately. The vector
                    store, cross-encoder, and Groq are never called —
                    this is a genuine full-pipeline bypass, not just a
                    skipped LLM call.
            MISS -> run the existing KB -> (search fallback) pipeline
                    exactly as before, then store the final answer in
                    the cache for next time.

    The cache fails open: if Redis is unreachable or disabled, check()
    always returns None and store() is a no-op, so the KB/search/LLM
    pipeline below is completely unaffected — this file's core logic is
    otherwise unchanged from the pre-caching version.

Decision signal used for KB vs. search:
    generator.py's system prompt instructs the LLM to output exactly
    INSUFFICIENT_CONTEXT when the retrieved chunks don't answer the
    question. That's the trigger checked here — no extra LLM call needed
    for the routing decision itself, which keeps this fast and cheap.

Usage:
    python router.py --query "What courses does the IT department offer?"
    python router.py --query "What is the weather in Chennai today?"   (forces search fallback)

Requires:
    GROQ_API_KEY in .env (see generator.py)
    Optional: REDIS_URL / SEMANTIC_CACHE_* in .env (see semantic_cache.py)
"""

import argparse
import logging
import os

from serpapi import GoogleSearch

import generator
import semantic_cache
from exceptions import ConfigurationError, GenerationError, SearchFallbackError, VectorStoreError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---- Config -------------------------------------------------------------

SEARCH_RESULT_COUNT = 5
SEARCH_TIMEOUT = 10  # seconds


# ---- Core functions -------------------------------------------------------

def web_search(query: str, max_results: int = SEARCH_RESULT_COUNT) -> list[dict]:
    """
    Run a live web search via SerpAPI (Google Search engine) and return results
    shaped like the chunk dicts generator.py expects, so they can be fed into
    the same build_context_block() / generate_answer() functions unchanged.
    """
    logger.info(f"Falling back to live SerpAPI web search for: {query!r}")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key or api_key == "your_serpapi_key_here":
        err_msg = (
            "SERPAPI_API_KEY not configured. Please add your SerpAPI key to .env "
            "(get a free key at https://serpapi.com)"
        )
        logger.error(err_msg)
        raise ConfigurationError(err_msg)

    try:
        search = GoogleSearch({
            "q": query,
            "api_key": api_key,
            "num": max_results,
            "engine": "google",
        })
        raw_results = search.get_dict()
        if "error" in raw_results:
            raise SearchFallbackError(query, raw_results["error"])
        organic_results = raw_results.get("organic_results", [])
    except (ConfigurationError, SearchFallbackError):
        raise
    except Exception as e:
        logger.error(f"SerpAPI web search failed: {e}")
        raise SearchFallbackError(query, str(e)) from e

    search_chunks = []
    for r in organic_results[:max_results]:
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        title = r.get("title", "Web result")
        if snippet or link:
            search_chunks.append({
                "chunk_id": link,
                "text": snippet or title,
                "source_url": link,
                "source_title": title,
            })

    logger.info(f"SerpAPI web search returned {len(search_chunks)} results")
    return search_chunks


def route_and_answer(question: str, retrieve_k: int = 20, rerank_n: int = 5) -> dict:
    """
    The core agentic decision, now with a semantic-cache gate in front:
        0. Check the semantic cache. On a hit, return immediately — no
           retrieval, no re-ranking, no LLM call.
        1. Cache miss -> try answering from the knowledge base.
        2. If the KB has enough context -> return that answer, used_search=False.
        3. If not -> fall back to live web search, regenerate, used_search=True.
        4. Whenever step 2 or 3 produces a real answer, store it in the
           cache so a future semantically similar question is a hit.

    Returns:
        {
            "answer": str,
            "sources": list[str],
            "used_search": bool,
            "kb_attempted": bool,   # False only when served straight from cache
            "cache_hit": bool,      # True if this answer came from the semantic cache
        }
    """
    logger.info(f"Routing query: {question!r}")

    # ---- Step 0: semantic cache gate ----------------------------------
    # This check happens BEFORE any retrieval/reranking/LLM work, so a hit
    # is a genuine full-pipeline bypass, not just a skipped LLM call.
    cached = semantic_cache.check(question)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "used_search": False,
            "kb_attempted": False,
            "cache_hit": True,
        }

    try:
        # Step 1: try the knowledge base first
        kb_result = generator.answer_question(question, retrieve_k, rerank_n)

        if not kb_result["insufficient"]:
            logger.info("KB context was sufficient — answering from knowledge base")
            semantic_cache.store(question, kb_result["answer"], kb_result["sources"])
            return {
                "answer": kb_result["answer"],
                "sources": kb_result["sources"],
                "used_search": False,
                "kb_attempted": True,
                "cache_hit": False,
            }

        # Step 2: KB wasn't enough -> fall back to live search
        logger.info("KB context was insufficient — routing to live web search")

        try:
            search_chunks = web_search(question)
        except (SearchFallbackError, ConfigurationError) as e:
            logger.error(f"Search fallback unavailable: {e}")
            search_chunks = []

        if not search_chunks:
            # Both KB and search came up empty — fail gracefully, never crash.
            # Not cached: there's no useful answer here to serve up next time.
            return {
                "answer": (
                    "I couldn't find this in the knowledge base, and the live "
                    "search fallback didn't return results either. Please try "
                    "rephrasing your question."
                ),
                "sources": [],
                "used_search": True,
                "kb_attempted": True,
                "cache_hit": False,
            }

        search_result = generator.generate_answer(question, search_chunks)
        semantic_cache.store(question, search_result["answer"], search_result["sources"])

        return {
            "answer": search_result["answer"],
            "sources": search_result["sources"],
            "used_search": True,
            "kb_attempted": True,
            "cache_hit": False,
        }

    except ConfigurationError:
        # Missing API key etc. — this needs the person to fix setup, not a
        # generic apology. Re-raise so it surfaces clearly at startup.
        raise

    except (VectorStoreError, GenerationError) as e:
        # Something failed in the core pipeline that retries couldn't fix.
        # Log the real error for debugging, but never show a stack trace
        # or crash to whoever is asking the question. Not cached — an
        # error message shouldn't get served up as a cached "answer".
        logger.error(f"Pipeline error while answering {question!r}: {e}")
        return {
            "answer": "Sorry, something went wrong while processing your question. Please try again.",
            "sources": [],
            "used_search": False,
            "kb_attempted": True,
            "cache_hit": False,
            "error": str(e),
        }


# ---- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test the full agentic routing pipeline")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    result = route_and_answer(args.query)

    print(f"\nQuestion: {args.query}")
    print(f"Served from semantic cache: {result.get('cache_hit', False)}")
    print(f"Used live search fallback: {result['used_search']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()