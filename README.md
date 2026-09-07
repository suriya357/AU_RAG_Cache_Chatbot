# Anna University IT Department — Agentic RAG Chatbot with Redis Semantic Caching and Live web Search

An agentic Retrieval-Augmented Generation (RAG) system that answers questions
about the Anna University IT department using its own scraped web content,
autonomously falls back to live web search when the knowledge base doesn't
have enough context, and now sits behind a **Redis + RedisVL semantic
cache** that answers repeated or semantically similar questions instantly —
with **zero retrieval and zero LLM calls** — while degrading gracefully to
the original pipeline if Redis is ever unavailable.

**Stack:** Python, FastAPI (backend), Streamlit (frontend), ChromaDB,
Groq, Redis + RedisVL — entirely free to run, no paid services required.

---

## What's new in this version

This project merges two previously separate codebases:

1. **Anna University RAG Chatbot** (`ingest.py` → `chunker.py` →
   `vectorstore.py` → `reranker.py` → `generator.py` → `router.py` →
   `main.py`/`app.py`) — unchanged in behavior.
2. **Redis Semantic Caching** (originally a standalone Streamlit demo) —
   its core idea (RedisVL `SemanticCache` + a local embedding model) is now
   a proper module, **`semantic_cache.py`**, wired directly into the RAG
   pipeline's actual request flow — not bolted on as a separate app.

The only pipeline file that changed behavior is **`router.py`**: it now
checks the semantic cache *before* doing any retrieval, and writes fresh
answers back to the cache after a cache miss. Every other file
(`ingest.py`, `chunker.py`, `vectorstore.py`, `reranker.py`,
`generator.py`, `exceptions.py`, `logging_config.py`) is untouched.
`main.py` and `app.py` gained one extra field (`cache_hit`) to surface
whether an answer came from the cache.

---

## Architecture

```
                         User Query
                             │
                             ▼
              ┌───────────────────────────┐
              │   semantic_cache.check()    │   Redis + RedisVL
              │   (router.py, Step 0)        │   vector similarity search
              └──────────────┬────────────┘
                     │                │
              CACHE HIT           CACHE MISS
                     │                │
                     ▼                ▼
        Return cached answer   ┌─────────────────────────────┐
        immediately.           │  Knowledge Base path          │
        No retrieval, no       │  ChromaDB vector search        │
        re-ranking, no LLM     │  → Cross-encoder re-rank          │
        call at all.           │  → Generate answer (Groq)            │
                                └─────────────┬───────────────────┘
                                              │ INSUFFICIENT_CONTEXT?
                                              ▼
                                ┌─────────────────────────────┐
                                │  Live Search fallback          │
                                │  DuckDuckGo → Generate (Groq)    │
                                └─────────────┬───────────────────┘
                                              │
                                              ▼
                                 semantic_cache.store()
                                 (cache the fresh answer for
                                  the next similar question)
                                              │
                                              ▼
                                     Return answer
```

If Redis is unreachable, disabled, or `redis`/`redisvl` aren't installed,
`semantic_cache.check()` always returns `None` (a "miss") and
`semantic_cache.store()` is a silent no-op — the knowledge-base and
search-fallback path runs exactly as it did before caching existed.

The frontend and backend remain fully decoupled — Streamlit (`app.py`)
talks to FastAPI (`main.py`) purely over HTTP.

---

## Why this reduces LLM/API cost

Every cache **hit** means:
- No ChromaDB vector search
- No cross-encoder re-ranking pass
- No Groq API call (the actual billed resource)
- Answer returned in low-single-digit milliseconds instead of the
  seconds a full retrieve → rerank → generate round trip takes

In a real deployment, a meaningful share of questions are exact repeats or
close paraphrases of ones already asked ("What courses does the IT
department offer?" vs. "Which courses are offered by the IT dept?"). The
semantic cache — unlike a plain exact-string cache — recognizes these as
the same question via embedding similarity, so paraphrases hit the cache
too. Every hit is one fewer Groq API call and one fewer set of embedding +
cross-encoder computations, which is where this system's actual per-query
cost lives.

---

## Project Structure

```
.
├── ingest.py              # Phase 1 — scrapes source pages (14 URLs)
├── chunker.py             # Phase 2 — splits text into chunks
├── vectorstore.py         # Phase 2 — embeddings + ChromaDB
├── reranker.py            # Phase 3 — cross-encoder re-ranking
├── generator.py           # Phase 4 — Groq generation + retry logic
├── router.py              # Phase 5 — semantic cache gate + agentic KB/search routing
├── semantic_cache.py      # Phase 9 — Redis + RedisVL semantic cache (NEW)
├── exceptions.py          # Phase 6 — custom exception types
├── logging_config.py      # Phase 6 — centralized logging
├── main.py                # Phase 7 — FastAPI backend
├── app.py                 # Frontend — Streamlit chat UI
├── eval_qa_set.py         # Phase 8 — labeled eval questions
├── evaluate.py            # Phase 8 — accuracy evaluation
├── tests/                 # pytest unit + integration tests
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_generator.py
│   ├── test_main.py
│   ├── test_reranker.py
│   ├── test_router.py         # includes cache-integration tests
│   └── test_semantic_cache.py # NEW — cache-specific unit tests
├── requirements.txt        # one merged, conflict-resolved dependency list
├── .env.example             # documented template — copy to .env
└── .gitignore
```

---

## Setup — Step by Step

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

Go to **https://console.groq.com/keys**, sign up (free), and create a key.

### 4. Configure your `.env` file

```bash
cp .env.example .env
```

Then edit `.env` and set your real `GROQ_API_KEY`. The Redis/semantic-cache
variables already have sensible defaults — see [Redis & Semantic Cache
Configuration](#redis--semantic-cache-configuration) below if you want to
tune them. **Never commit `.env` to version control** (it's already
gitignored).

### 5. (Optional but recommended) Start Redis

The app runs perfectly well **without** Redis — it just runs without
caching. To get the caching benefits, start Redis locally, e.g. via Docker:

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This exposes Redis on port `6379` (used by the app) and RedisInsight, a
GUI for inspecting cached entries, on port `8001`
(http://localhost:8001).

---

## Redis & Semantic Cache Configuration

All caching behavior is controlled via environment variables (`.env`) —
nothing is hard-coded. See `.env.example` for the full documented list;
the key ones:

| Variable | Default | What it does |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Where to find Redis |
| `SEMANTIC_CACHE_ENABLED` | `true` | Set to `false` to bypass the cache entirely |
| `SEMANTIC_CACHE_THRESHOLD` | `0.15` | Cosine **distance** cutoff for a hit — lower = stricter matching |
| `SEMANTIC_CACHE_TTL_SECONDS` | `86400` (24h) | How long a cached answer stays valid |
| `SEMANTIC_CACHE_NAME` | `au_it_rag_cache` | Redis index name for the cache |

**Tuning the threshold:** `SEMANTIC_CACHE_THRESHOLD` is a cosine distance
(lower = more similar), not a percentage. `0.0` only matches
near-identical questions; `0.10–0.20` is a good default for catching
paraphrases; going much above `0.3` risks serving a cached answer for a
question that isn't actually the same, which is worse than a cache miss.
If in doubt, start low and loosen it after watching real HIT/MISS logs.

---

## Step-by-Step: Build the Knowledge Base

Run these **in order**, once, to scrape the site and build the vector
store (unchanged from the original RAG project — semantic caching doesn't
touch this part):

```bash
# Step 1 — Scrape all 14 department pages into data/raw/
python ingest.py

# Step 2 — Split scraped text into overlapping chunks
python chunker.py

# Step 3 — Embed chunks and build the ChromaDB vector store
python vectorstore.py --build
```

**What to check after this:**
- `data/raw/` should contain ~14 JSON files (one per scraped page) plus `_manifest.json`
- `data/chunks/chunks.json` should contain the full chunk list
- `data/chroma_db/` should exist (this is your vector database)

If you want to add more pages later, edit the `URLS` list at the top of
`ingest.py`, then re-run all three steps above (`vectorstore.py --build` is
safe to re-run — it upserts, no duplicates).

---

## Step-by-Step: Run the Application

You need **two terminals** running at the same time — one for the backend,
one for the frontend. (Redis, if you're using it, runs separately as its
own process/container — see Setup step 5.)

### Terminal 1 — Start the backend (FastAPI)

```bash
source venv/bin/activate
uvicorn main:app --reload
```

You should see something like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```
and, if Redis is reachable, a log line like:
```
Semantic cache connected -> redis://localhost:6379 (index=au_it_rag_cache, threshold=0.15, ttl=86400s)
```
If Redis isn't running, you'll instead see a warning and the app will
continue normally:
```
Semantic cache unavailable (...) — continuing without it; every query will go through the normal RAG + LLM pipeline.
```

**Verify it's working:** open http://127.0.0.1:8000/docs in a browser —
you should see interactive API documentation, including a `cache_hit`
field on the `/query` response.

### Terminal 2 — Start the frontend (Streamlit)

```bash
source venv/bin/activate
streamlit run app.py
```

This will automatically open a browser tab at **http://localhost:8501**.

### Using the app

- Type a question in the chat box, or click one of the example questions
  in the sidebar
- Each answer shows how it was produced:
  - ⚡ **Answered from semantic cache (no LLM call)**
  - 📚 **Answered from knowledge base**
  - 🔎 **Answered via live web search fallback**
- Ask the same (or a reworded) question again to see the ⚡ cache badge
  and a much faster response
- Click "Clear conversation" in the sidebar to start fresh

---

## Cache HIT/MISS Logging

`semantic_cache.py` logs every lookup and write through the project's
existing centralized logger (`logging_config.py`), so entries show up both
in the console and in `logs/rag_pipeline.log`:

```
2024-01-01 12:00:00 [INFO] semantic_cache: CACHE MISS: 'What courses does the IT department offer?'
2024-01-01 12:00:03 [INFO] semantic_cache: CACHE STORE: 'What courses does the IT department offer?'
2024-01-01 12:00:15 [INFO] semantic_cache: CACHE HIT: 'Which courses does IT offer?' matched cached question 'What courses does the IT department offer?' (distance=0.081)
```

`router.py` also logs the routing decision itself (whether a query was
served from cache, the KB, or search), so the full lifecycle of any
question is traceable from the logs alone.

---

## Testing

```bash
pytest tests/ -v                                  # run all tests
pytest tests/ --cov=. --cov-report=term-missing    # with coverage
```

Tests stub out **all** external services — Groq, ChromaDB, DuckDuckGo,
**and Redis/RedisVL** — so the full suite runs fast, offline, and without
needing real API keys, a real Groq account, or a real Redis instance.

`tests/test_semantic_cache.py` covers the semantic-cache module directly:
- **Exact repeat query** → cache hit, same answer returned
- **Semantically similar (paraphrased) query** → cache hit
- **Unrelated query** → cache miss
- **Redis unreachable / not installed** → graceful fallback, no crash
- Cache disabled via `SEMANTIC_CACHE_ENABLED=false` → always a miss, `store()` is a no-op
- `clear()` empties the cache

`tests/test_router.py` covers the *integration* — the part of the
requirements that matters most:
- A cache hit **never calls `generator.answer_question`** (asserted via a
  monkeypatch that raises if it's called) — proving the LLM is genuinely
  skipped, not just short-circuited after being called
- A cache miss that gets a good KB answer **writes it back to the cache**
- A cache miss that falls through to web search **also writes that answer
  back to the cache**
- The generic "couldn't find this anywhere" fallback message is **never**
  cached
- Simulated Redis failure (`check`/`store` behaving as they would with
  Redis down) still produces a correct answer via the normal pipeline

---

## Evaluation

```bash
python evaluate.py
```

Runs the labeled question set in `eval_qa_set.py` against your **live**
pipeline and reports real keyword accuracy and routing accuracy. This is
unaffected by caching (it measures answer quality, not cache behavior) —
run it with `SEMANTIC_CACHE_ENABLED=false` if you want to evaluate the raw
RAG pipeline in isolation.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Streamlit sidebar shows "Backend unreachable" | Make sure Terminal 1 (`uvicorn main:app --reload`) is still running |
| `GROQ_API_KEY not found` error | Check your `.env` file exists (copied from `.env.example`) and has the correct variable name |
| Log says "Semantic cache unavailable" | Expected if Redis isn't running — the app still works, just without caching. Start Redis (see Setup step 5) if you want caching |
| Cache never hits on questions that seem similar | Lower `SEMANTIC_CACHE_THRESHOLD` isn't the fix — try *raising* it slightly (e.g. `0.15` → `0.25`) since a lower number is stricter, not looser |
| Cache hits on questions that shouldn't match | *Lower* `SEMANTIC_CACHE_THRESHOLD` (e.g. `0.15` → `0.08`) to require closer matches |
| Vector store query returns nothing | Re-run `python vectorstore.py --build` — did the knowledge-base build steps complete without errors? |
| `ModuleNotFoundError` for `langchain_text_splitters` or `redisvl` | Run `pip install -r requirements.txt` again |
| Port 8000 or 8501 already in use | Another process is using that port — stop it, or run `uvicorn main:app --reload --port 8001` (and update `API_BASE` in `app.py` to match) |
| Want to force a clean cache | Call `python -c "import semantic_cache; semantic_cache.clear()"`, or use RedisInsight (http://localhost:8001) if running the Docker setup above |

---

## Key Design Decisions

**Semantic cache as a true pipeline gate, not a wrapper around the LLM
call.** `router.route_and_answer()` checks the cache *before* calling
`generator.answer_question()` at all — on a hit, ChromaDB, the
cross-encoder, and Groq are never touched. This is what makes the cost
savings real: a hit isn't "skip the API call after already doing the
retrieval work", it's "skip everything".

**Fail-open caching.** `semantic_cache.py` never raises out of `check()`
or `store()`. Any failure — Redis down, package not installed, a
malformed response — is caught, logged, and treated as "cache
unavailable", so a Redis outage degrades the app to its pre-caching
behavior instead of taking it down.

**Reuses the project's existing patterns.** The cache module follows the
same conventions as the rest of the codebase: config via `.env`
(`generator.py` does the same for `GROQ_API_KEY`), the same centralized
logger (`logging_config.py`), and the same "return a clean result dict,
never let a raw exception reach the API layer" philosophy already used in
`router.py`'s KB/search error handling.

**Frontend/backend decoupling preserved.** `app.py` still never imports
the pipeline directly — it only talks to `main.py` over HTTP, now
including the extra `cache_hit` field.

**Two-stage retrieval, unchanged.** Vector search (fast, coarse) narrows
candidates to 20, then a cross-encoder (slow, precise) re-ranks down to
the top 5 — exactly as before caching was added.

**Routing without an extra LLM call, unchanged.** The generation step's
own prompt instructs the model to emit `INSUFFICIENT_CONTEXT` when it
can't answer — `router.py` just checks for that marker, avoiding a
separate "should I search?" API call.
