# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A self-hosted RAG (Retrieval-Augmented Generation) assistant that indexes code repositories and provides an AI chat interface with context-aware answers. It sits between Open WebUI and an LLM (Gemini or OpenAI), injecting retrieved context into queries.

## Running the System

```bash
# First-time setup
cp .env.example .env
# Edit .env: set GEMINI_API_KEY (or OpenAI key) and REPOS_HOST_PATH

# Start Qdrant, then index, then full stack
docker compose up -d qdrant
docker compose run --rm indexer
docker compose up -d
```

```bash
# Re-index repositories
docker compose run --rm indexer

# Incremental nightly reindex (git pull + reindex)
bash scripts/reindex.sh
```

## Key Endpoints (RAG Proxy, port 8001)

- `POST /v1/chat/completions` — OpenAI-compatible endpoint (main entry point from Open WebUI)
- `GET /health` — health check
- `GET /metrics` — latency percentiles, cost tracking
- `GET /stats` — Qdrant collection stats
- `POST /reindex` — trigger re-indexing via API

## Architecture

```
Open WebUI (port 3000)
    ↓ /v1/chat/completions
RAG Proxy — app/rag_proxy.py (FastAPI, port 8001)
    ↓ hybrid search              ↓ generate response
Qdrant (port 6333)          Gemini / OpenAI API
```

**Indexing pipeline:** `scripts/indexer.py` reads repo files from `REPOS_HOST_PATH`, chunks them with code-aware logic, embeds via `sentence-transformers`, and upserts into Qdrant with both vector and text indexes (enabling hybrid search).

**Query pipeline:** For each chat request, `rag_proxy.py` extracts the user query, runs hybrid search (vector + BM25-style text), optionally reranks with a cross-encoder, builds a context-augmented system prompt, then streams the LLM response back.

## Key Design Decisions

- **Hybrid search weight** (`HYBRID_ALPHA`): 0.0 = pure text, 1.0 = pure vector, default 0.7
- **Reranking**: disabled by default; enable with `RERANKING_ENABLED=true` (uses `ms-marco-MiniLM-L-6-v2`)
- **Smart chunking**: code-aware, splits on function/class boundaries with configurable overlap
- **Repo filtering**: query is scanned for repo names (fuzzy match) to scope search
- **Resilience**: circuit breaker (5 failures → 30s cooldown) + exponential backoff retry + rate limiting (30 req/min/IP)
- **Streaming**: uses `asyncio.Queue` to forward LLM chunks in real time to the client
- **Incremental indexing**: content-hashed, only changed files are re-embedded

## LLM Provider Selection

Set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` in `.env`. The OpenAI provider also works with GitHub Copilot API (set `OPENAI_BASE_URL` accordingly).

## Files to Know

- `app/rag_proxy.py` — the entire proxy service (~900 lines): search, reranking, prompt building, streaming, circuit breaker, metrics
- `scripts/indexer.py` — chunking, embedding, Qdrant upsert; handles Python/JS/TS/Java/Go/Rust and many others, plus Excel files
- `docker-compose.yml` — defines all four services: `qdrant`, `indexer`, `rag-proxy`, `open-webui`
- `.env.example` — all 40+ configuration variables with descriptions
