# 🤖 RAG Business Assistant

AI assistant for querying the business logic of your repositories.
**Qdrant** + **Google Gemini** + **Open WebUI**.

Production-ready: hybrid search, reranking, circuit breaker, retry, metrics, logging.

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
REPOS_HOST_PATH=/home/mario/projects   # your folder, as-is
```

The folder must contain subfolders (each subfolder = one repo):
```
/home/mario/projects/
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
- `0.7` = default, good balance

Results appearing in both searches receive a boost.

### Reranking
After search, a **cross-encoder** (ms-marco-MiniLM-L-6-v2) re-evaluates each result by directly comparing it with the question. Slower than vector search alone, but much more accurate because the cross-encoder "reads" question and context together.

Configurable: `ENABLE_RERANKING=true/false`, `RERANK_CANDIDATES=30` (how many candidates to evaluate).

### Smart Chunking
Code is split into logical blocks (functions, classes, methods), not fixed lines. Each chunk includes the filename as a header to preserve context.

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

### What you find in /metrics

```json
{
  "total_queries": 1542,
  "total_errors": 3,
  "error_rate_pct": 0.2,
  "queries_no_context": 12,
  "latency": {
    "avg_ms": 1850,
    "p50_ms": 1620,
    "p95_ms": 3200,
    "p99_ms": 5100
  },
  "rag_latency": {
    "avg_ms": 145,
    "p95_ms": 320
  },
  "tokens": {
    "total_input": 2450000,
    "total_output": 890000
  },
  "cost_usd": {
    "input": 0.1838,
    "output": 0.267,
    "total": 0.4508
  },
  "errors_by_type": {
    "rate_limit": 2,
    "retry_APIError": 1
  }
}
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
| `GEMINI_API_KEY` | (required) | Gemini API key |
| `REPOS_HOST_PATH` | (required) | Path to repositories folder |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `MAX_RISULTATI` | `8` | Chunks in context |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking |
| `RERANK_CANDIDATES` | `30` | Pre-reranking candidates |
| `HYBRID_ALPHA` | `0.7` | Vector vs full-text weight |
| `GEMINI_MAX_RETRIES` | `3` | Retries on error |
| `GEMINI_RETRY_DELAY` | `1.0` | Base retry delay (sec) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests/min per IP |
| `LOG_LEVEL` | `INFO` | Log level |
| `ENABLE_QUERY_LOG` | `true` | Log queries to file |

---

## 📁 Structure

```
rag-business-assistant/
├── docker-compose.yml
├── Dockerfile.proxy
├── Dockerfile.indexer
├── .env.example
├── .gitignore
├── README.md
├── app/
│   └── rag_proxy.py        # RAG + Gemini + resilience + metrics
└── scripts/
    ├── indexer.py            # Indexing → Qdrant
    └── reindex.sh            # Automatic update
```
