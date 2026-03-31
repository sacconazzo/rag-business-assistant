# 🤖 RAG Business Assistant

AI assistant for querying the business logic of your repositories.
**Qdrant** + **Google Gemini** + **Open WebUI**.

---

## 📋 Architecture

```
         CDN / Load Balancer
                │
        ┌───────┴───────┐
        │  Open WebUI   │  ← your interface
        │  (port 3000)  │
        └───────┬───────┘
                │
        ┌───────┴────────────────┐     ┌──────────────┐
        │  RAG Proxy (FastAPI)   │────▶│  Gemini API  │
        │  (port 8001)           │     └──────────────┘
        │                        │
        │  • Hybrid search       │
        │  • Reranking           │
        │  • Circuit breaker     │
        │  • Rate limiting       │
        │  • Retry + backoff     │
        │  • Metrics + logging   │
        └───────┬────────────────┘
                │
        ┌───────┴───────┐
        │    Qdrant     │
        │  (port 6333)  │
        └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Gemini API key → https://aistudio.google.com/apikey
- Folder containing your repositories

### 1. Configure

```bash
cd rag-business-assistant
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=AIzaSy-your-key
REPOS_HOST_PATH=/workspace/projects/my-app   # your folder, as-is
```

The folder must contain subfolders (each subfolder = one repo):
```
/workspace/projects/my-app/
├── repo-api/
├── repo-frontend/
├── repo-services/
└── ...
```

### 2. Index

```bash
docker compose up -d qdrant
sleep 10
docker compose run --rm indexer
```

### 3. Start

```bash
docker compose up -d
```

### 4. Configure Open WebUI

1. Open http://localhost:3000, create admin account
2. **Admin Panel → Settings → Connections**
3. Add OpenAI API: URL `http://rag-proxy:8001/v1`, Key `not-needed`
4. Select model **"business-assistant"** in the chat

---

## 🔍 RAG Quality

The system uses three techniques to maximize relevance:

### Hybrid Search
Combines **vector** search (semantic, captures meaning) with **full-text** search (keyword, captures exact terms). The weight is configurable via `HYBRID_ALPHA`:
- `1.0` = vector only
- `0.0` = full-text only
- `0.65` = default, balanced for code (more weight to keyword matching for function/variable names)

Results appearing in both searches receive a boost.

### Reranking
After search, a **cross-encoder** (ms-marco-MiniLM-L-6-v2) re-evaluates each result by directly comparing it with the question. Slower than vector search alone, but much more accurate because the cross-encoder "reads" question and context together.

Configurable: `ENABLE_RERANKING=true/false`, `RERANK_CANDIDATES=30` (how many candidates to evaluate).

### Smart Chunking
Code is split into logical blocks (functions, classes, methods), not fixed lines. Each chunk includes the file path as a header to preserve context. Consecutive chunks overlap by `CHUNK_OVERLAP_CHARS` characters (default 200) to avoid losing context at chunk boundaries.

---

## 🧠 Embedding Models

### Default model: `BAAI/bge-base-en-v1.5`
General-purpose model with strong MTEB code search benchmark scores. 110M parameters, 768 dimensions, ~440MB. Runs well on CPU (MacBook Pro M3/M4).

### `EMBEDDING_QUERY_PREFIX`
BGE is an **instruction-following** model: it requires a text prefix on queries for optimal results. Documents are indexed **without** the prefix, while search queries use it. The default value is `Represent this sentence for searching relevant passages: `. If you switch to a model that doesn't need a prefix (e.g. `all-mpnet-base-v2`), just set this variable to empty.

### Recommended alternative: `jinaai/jina-embeddings-v2-base-code`
Specifically designed for source code, with an 8K token context window (vs 512 for BGE). Ideal when code comprehension is the priority and indexing time is not critical. To use it:
```env
EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code
EMBEDDING_QUERY_PREFIX=
```
> ⚠️ Changing the model requires full re-indexing: `docker compose run --rm indexer`

### Other alternatives
- `all-mpnet-base-v2` — good general-purpose model (768 dim, no prefix needed)

---

## 🛡️ Resilience

### Retry with exponential backoff
If Gemini returns a temporary error (500, 429, timeout), the proxy automatically retries with increasing delay: 1s → 2s → 4s. Does not retry on client errors (invalid API key, malformed request).

### Circuit breaker
If Gemini fails 5 consecutive times, the circuit **opens**: the proxy stops calling Gemini for 30 seconds and returns 503. After the cooldown, it tries a test request. If successful, resumes normally.

This avoids accumulating timeouts during a service outage.

### Rate limiting
Configurable per-IP limit (default: 30 requests/minute). Returns 429 with `Retry-After` header.

---

## 📊 Observability

### Metrics endpoints

```bash
# Health check (for CDN / load balancer)
curl http://localhost:8001/health

# Detailed metrics
curl http://localhost:8001/metrics

# Qdrant stats + metrics
curl http://localhost:8001/stats
```


### Query log

Every question is logged to `/app/logs/queries.jsonl` with:
- Question text (truncated to 200 chars)
- Number of sources found
- Total latency and RAG latency
- Tokens consumed
- Timestamp

The `logs-data` volume persists across restarts. You can analyze logs with `jq`:

```bash
# Questions without context (indicates indexing issues)
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq 'select(.fonti == 0)'

# Slowest questions
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq -s 'sort_by(-.latency_ms) | .[0:10]'

# Approximate daily cost
curl -s http://localhost:8001/metrics | jq '.cost_usd'
```

### Qdrant Dashboard
http://localhost:6333/dashboard — indexed vectors, performance, collection status.

---

## 🔄 Update the index

```bash
docker compose run --rm indexer
```

Or nightly cron:
```cron
0 3 * * * cd /path/to/rag-business-assistant && bash scripts/reindex.sh >> logs/reindex.log 2>&1
```

Or via API:
```bash
curl -X POST http://localhost:8001/reindex
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | **(required)** | Gemini API key |
| `REPOS_HOST_PATH` | **(required)** | Absolute path to repositories folder |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `SYSTEM_PROMPT` | `` | Custom system prompt (empty = default) |
| `MAX_RISULTATI` | `8` | Chunks in context |
| `HYBRID_ALPHA` | `0.65` | Vector vs full-text weight (0=text, 1=vector) |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking |
| `RERANK_CANDIDATES` | `30` | Candidates evaluated before reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RERANK_TRUNCATE` | `1024` | Max chars passed to reranker |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_QUERY_PREFIX` | `Represent this sentence...` | Query prefix for instruction-following models |
| `CHUNK_MAX_CHARS` | `1500` | Maximum chunk size (chars) |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap between consecutive chunks (chars) |
| `BATCH_SIZE` | `64` | Indexing batch size |
| `HNSW_M` | `16` | HNSW connections per node |
| `HNSW_EF` | `128` | HNSW construction candidates |
| `ENABLE_QUANTIZATION` | `true` | Scalar INT8 quantization |
| `FORCE_REINDEX` | `false` | Drop and recreate collection on next index run |
| `COLLECTION_NAME` | `codebase` | Qdrant collection name |
| `GEMINI_MAX_RETRIES` | `3` | Retries on error |
| `GEMINI_RETRY_DELAY` | `1.0` | Base retry delay (sec) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests/min per IP |
| `LOG_LEVEL` | `INFO` | Log level |
| `ENABLE_QUERY_LOG` | `true` | Log queries to file |
| `QUERY_LOG_MAX_SIZE_MB` | `50` | Max query log file size before rotation |
| `QDRANT_PORT` | `6333` | Qdrant exposed port |
| `WEBUI_PORT` | `3000` | Open WebUI exposed port |
| `RAG_PROXY_PORT` | `8001` | RAG Proxy exposed port |
