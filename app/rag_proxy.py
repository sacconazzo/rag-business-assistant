"""
RAG Proxy — Production
Resilience (retry, rate limit, circuit breaker)
Observability (structured logging, metrics, cost tracking)
RAG quality (hybrid search, reranking with cross-encoder)
"""

import os
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
from functools import partial

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codebase")
MAX_RISULTATI = int(os.getenv("MAX_RISULTATI", "8"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_QUERY_LOG = os.getenv("ENABLE_QUERY_LOG", "true").lower() == "true"

# Resilience
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_DELAY = float(os.getenv("GEMINI_RETRY_DELAY", "1.0"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# Advanced RAG
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.65"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TRUNCATE = int(os.getenv("RERANK_TRUNCATE", "1024"))

# Query log rotation
QUERY_LOG_MAX_SIZE_MB = int(os.getenv("QUERY_LOG_MAX_SIZE_MB", "50"))

# System prompt (configurable)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are an expert assistant for our business codebase.
Answer questions about business logic based EXCLUSIVELY on the code
snippets and documentation provided as context.

Rules:
- If the context does not contain enough information, say so clearly.
- Cite the file and repository from which you take the information.
- Explain the logic clearly, as if talking to a colleague.
- If you find inconsistencies in the code, point them out.
- Reply in the same language as the question.
""")


# ============================================================
# STRUCTURED LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag-proxy")


class QueryLogger:
    """Writes each query to a JSONL file with automatic rotation."""

    def __init__(self, log_dir="/app/logs", max_size_mb: int = QUERY_LOG_MAX_SIZE_MB):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "queries.jsonl")
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def _rotate_if_needed(self):
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_size_bytes:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                rotated = os.path.join(self.log_dir, f"queries_{timestamp}.jsonl")
                os.rename(self.log_file, rotated)
                # Keep only last 5 rotated files
                rotated_files = sorted(
                    [f for f in os.listdir(self.log_dir) if f.startswith("queries_") and f.endswith(".jsonl")]
                )
                for old_file in rotated_files[:-5]:
                    try:
                        os.remove(os.path.join(self.log_dir, old_file))
                    except OSError:
                        pass
                logger.info(f"Query log rotated: {rotated}")
        except Exception as e:
            logger.error(f"Log rotation error: {e}")

    def log(self, entry: dict):
        if not ENABLE_QUERY_LOG:
            return
        entry["timestamp"] = datetime.utcnow().isoformat()
        try:
            self._rotate_if_needed()
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Log write error: {e}")


query_logger = QueryLogger()


# ============================================================
# METRICS
# ============================================================

MAX_ERROR_TYPES = 100  # Cap on distinct error types tracked


class Metrics:
    def __init__(self):
        self.total_queries = 0
        self.total_errors = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.latencies = []
        self.rag_latencies = []
        self.errors_by_type = defaultdict(int)
        self.queries_no_context = 0
        self.started_at = datetime.utcnow().isoformat()

    def record_query(self, latency_ms: float, rag_ms: float, tokens_in: int, tokens_out: int, had_context: bool):
        self.total_queries += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.latencies.append(latency_ms)
        self.rag_latencies.append(rag_ms)
        if not had_context:
            self.queries_no_context += 1
        # Keep only last 1000
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
            self.rag_latencies = self.rag_latencies[-1000:]

    def record_error(self, error_type: str):
        self.total_errors += 1
        # Prevent unbounded growth of error types
        if error_type in self.errors_by_type or len(self.errors_by_type) < MAX_ERROR_TYPES:
            self.errors_by_type[error_type] += 1
        else:
            self.errors_by_type["_other"] += 1

    def _percentile(self, data: list, p: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def summary(self) -> dict:
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_rag = sum(self.rag_latencies) / len(self.rag_latencies) if self.rag_latencies else 0

        # Estimated Gemini Flash costs ($/M token): input=0.075, output=0.30
        cost_in = (self.total_tokens_in / 1_000_000) * 0.075
        cost_out = (self.total_tokens_out / 1_000_000) * 0.30

        return {
            "started_at": self.started_at,
            "total_queries": self.total_queries,
            "total_errors": self.total_errors,
            "error_rate_pct": round((self.total_errors / max(self.total_queries, 1)) * 100, 1),
            "queries_no_context": self.queries_no_context,
            "latency": {
                "avg_ms": round(avg_latency),
                "p50_ms": round(self._percentile(self.latencies, 0.5)),
                "p95_ms": round(self._percentile(self.latencies, 0.95)),
                "p99_ms": round(self._percentile(self.latencies, 0.99)),
            },
            "rag_latency": {
                "avg_ms": round(avg_rag),
                "p95_ms": round(self._percentile(self.rag_latencies, 0.95)),
            },
            "tokens": {
                "total_input": self.total_tokens_in,
                "total_output": self.total_tokens_out,
            },
            "cost_usd": {
                "input": round(cost_in, 4),
                "output": round(cost_out, 4),
                "total": round(cost_in + cost_out, 4),
            },
            "errors_by_type": dict(self.errors_by_type),
        }


metrics = Metrics()


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.requests = defaultdict(list)

    def check(self, client_ip: str) -> bool:
        now = time.time()
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < 60]
        if len(self.requests[client_ip]) >= self.max_per_minute:
            return False
        self.requests[client_ip].append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        now = time.time()
        recent = [t for t in self.requests.get(client_ip, []) if now - t < 60]
        return max(0, self.max_per_minute - len(recent))


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitBreaker:
    """
    Se Gemini fallisce N volte di fila, apre il circuito per un cooldown.
    Evita di bombardare l'API durante un'interruzione.
    """
    CLOSED = "closed"       # tutto ok
    OPEN = "open"           # bloccato, non chiama
    HALF_OPEN = "half_open" # prova una richiesta

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = self.CLOSED
        self.failures = 0
        self.last_failure_time = 0

    def can_execute(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.cooldown_seconds:
                self.state = self.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN (tentativo)")
                return True
            return False
        # HALF_OPEN: lascia passare una richiesta
        return True

    def record_success(self):
        if self.state == self.HALF_OPEN:
            logger.info("Circuit breaker → CLOSED (recuperato)")
        self.failures = 0
        self.state = self.CLOSED

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(
                f"Circuit breaker → OPEN ({self.failures} fallimenti consecutivi, "
                f"cooldown {self.cooldown_seconds}s)"
            )


circuit_breaker = CircuitBreaker()


# ============================================================
# RETRY CON BACKOFF
# ============================================================

async def call_gemini_with_retry(chat, content: str, stream: bool = False):
    if not circuit_breaker.can_execute():
        raise HTTPException(
            status_code=503,
            detail="Servizio temporaneamente non disponibile. Riprova tra qualche secondo."
        )

    loop = asyncio.get_event_loop()
    last_error = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            # Run synchronous Gemini SDK in thread to avoid blocking the event loop
            result = await loop.run_in_executor(
                None, partial(chat.send_message, content, stream=stream)
            )
            circuit_breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Non ritentare su errori client
            if any(kw in error_str for kw in ["invalid", "permission", "api_key", "not found"]):
                circuit_breaker.record_failure()
                raise

            delay = GEMINI_RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Gemini tentativo {attempt + 1}/{GEMINI_MAX_RETRIES}: {e}. Retry in {delay:.1f}s")
            metrics.record_error(f"retry_{type(e).__name__}")
            await asyncio.sleep(delay)

    circuit_breaker.record_failure()
    raise last_error


# ============================================================
# REINDEX LOCK
# ============================================================

_reindex_lock = asyncio.Lock()


# ============================================================
# INIT
# ============================================================

qdrant: QdrantClient = None
embedder: SentenceTransformer = None
reranker: CrossEncoder = None
gemini_model = None
known_repos: set[str] = set()


def _load_known_repos():
    """Load distinct repo names from Qdrant for natural language repo detection."""
    global known_repos
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return
        repos = set()
        offset = None
        while True:
            results, offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=["repo"],
                with_vectors=False,
            )
            for point in results:
                repo = point.payload.get("repo", "")
                if repo:
                    repos.add(repo)
            if offset is None:
                break
        known_repos = repos
        logger.info(f"Known repos: {sorted(known_repos)}")
    except Exception as e:
        logger.warning(f"Could not load repo list: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, embedder, reranker, gemini_model

    qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
    collections = [c.name for c in qdrant.get_collections().collections]
    count = qdrant.count(COLLECTION_NAME).count if COLLECTION_NAME in collections else 0
    logger.info(f"Qdrant: {QDRANT_URL} — {count} vectors")

    embedder = SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"attn_implementation": "eager"})
    logger.info(f"Embedding: {EMBEDDING_MODEL} (dim={embedder.get_sentence_embedding_dimension()})")

    if ENABLE_RERANKING:
        reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Reranker: {RERANKER_MODEL}")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    logger.info(f"Gemini: {GEMINI_MODEL}")

    _load_known_repos()

    yield
    logger.info("Shutdown")


app = FastAPI(title="RAG Proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# HYBRID SEARCH + RERANKING
# ============================================================

def cerca_contesto(domanda: str, n_risultati: int = MAX_RISULTATI, repo_filter: str = None) -> list[dict]:
    """
    1. Vector search (semantic)
    2. Full-text search (keyword)
    3. Fusion with HYBRID_ALPHA weight
    4. Reranking with cross-encoder
    """
    query_vector = embedder.encode(EMBEDDING_QUERY_PREFIX + domanda).tolist()
    candidates = RERANK_CANDIDATES if ENABLE_RERANKING else n_risultati

    # Optional repo filter
    qdrant_filter = None
    if repo_filter:
        qdrant_filter = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo_filter))])

    # --- Vector search ---
    vector_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=candidates,
        with_payload=True,
    ).points

    # --- Full-text search ---
    text_results = []
    try:
        text_filter_conditions = [FieldCondition(key="content", match=MatchText(text=domanda))]
        if repo_filter:
            text_filter_conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo_filter)))
        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=text_filter_conditions),
            limit=candidates,
            with_payload=True,
            with_vectors=False,
        )
        text_results = scroll_result[0] if scroll_result else []
    except Exception as e:
        logger.debug(f"Full-text fallback: {e}")

    # --- Fusion ---
    seen_ids = set()
    merged = []

    for hit in vector_results:
        seen_ids.add(hit.id)
        merged.append({
            "id": hit.id,
            "content": hit.payload.get("content", ""),
            "file": hit.payload.get("file", "?"),
            "repo": hit.payload.get("repo", "?"),
            "extension": hit.payload.get("extension", ""),
            "score": hit.score * HYBRID_ALPHA,
            "source": "vector",
        })

    for point in text_results:
        if point.id not in seen_ids:
            seen_ids.add(point.id)
            merged.append({
                "id": point.id,
                "content": point.payload.get("content", ""),
                "file": point.payload.get("file", "?"),
                "repo": point.payload.get("repo", "?"),
                "extension": point.payload.get("extension", ""),
                "score": (1 - HYBRID_ALPHA) * 0.8,
                "source": "text",
            })
        else:
            # Boost for results appearing in both searches
            for m in merged:
                if m["id"] == point.id:
                    m["score"] += (1 - HYBRID_ALPHA) * 0.5
                    m["source"] = "hybrid"
                    break

    # --- Reranking ---
    if ENABLE_RERANKING and reranker and merged:
        pairs = [(domanda, m["content"][:RERANK_TRUNCATE]) for m in merged]
        scores = reranker.predict(pairs)
        for m, s in zip(merged, scores):
            m["rerank_score"] = float(s)
        merged.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        merged.sort(key=lambda x: x["score"], reverse=True)

    risultati = []
    for m in merged[:n_risultati]:
        risultati.append({
            "codice": m["content"],
            "file": m["file"],
            "repo": m["repo"],
            "extension": m["extension"],
            "score": round(m.get("rerank_score", m["score"]), 3),
            "source": m["source"],
        })

    return risultati


# ============================================================
# HELPERS
# ============================================================

def openai_to_gemini_history(messages: list) -> list:
    history = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        history.append({"role": "model" if role == "assistant" else "user", "parts": [msg.get("content", "")]})
    return history


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


def _detect_repo_filter(domanda: str) -> tuple[str, str | None]:
    """Detect repo name mentioned in the question using known repo names from Qdrant.

    Supports:
      - Exact match: "in payment-service how does checkout work?"
      - Partial match: "in payment come funziona il checkout?"
        (matches "my-company-payment-service-v2" if "payment" is a segment)

    Segments are split on '-', '_', '.', '/' so "payment" matches a repo
    named "acme-payment-service" but not "acme-repayment-tool".
    """
    if not known_repos:
        return domanda, None

    import re
    domanda_lower = domanda.lower()
    # Extract candidate words (2+ chars, alphanumeric/hyphens) from question
    domanda_words = set(re.findall(r"[a-z0-9](?:[a-z0-9\-_.]*[a-z0-9])?", domanda_lower))

    best_match = None
    best_score = 0  # number of matching segments

    for repo in known_repos:
        repo_lower = repo.lower()

        # 1) Exact full-name match (highest priority)
        if repo_lower in domanda_lower:
            seg_count = len(re.split(r"[-_./]", repo_lower))
            if best_match is None or seg_count > best_score or (seg_count == best_score and len(repo) > len(best_match)):
                best_match = repo
                best_score = seg_count
            continue

        # 2) Partial: check how many repo segments appear as words in the question
        repo_segments = set(re.split(r"[-_./]", repo_lower))
        repo_segments.discard("")
        # Only consider segments with 3+ chars to avoid false positives
        meaningful_segments = {s for s in repo_segments if len(s) >= 3}
        if not meaningful_segments:
            continue

        matched = meaningful_segments & domanda_words
        if len(matched) >= 1 and len(matched) / len(meaningful_segments) >= 0.4:
            score = len(matched)
            if score > best_score or (score == best_score and best_match and len(repo) > len(best_match)):
                best_match = repo
                best_score = score

    if best_match:
        logger.debug(f"Repo filter detected: '{best_match}' (score={best_score})")

    return domanda, best_match


# ============================================================
# ENDPOINTS OPENAI-COMPATIBILI
# ============================================================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "business-assistant",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rag-proxy",
            "name": "Business Logic Assistant",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start_time = time.time()
    client_ip = get_client_ip(request)

    # Rate limit
    if not rate_limiter.check(client_ip):
        metrics.record_error("rate_limit")
        raise HTTPException(status_code=429, detail="Too many requests. Please retry in a minute.", headers={"Retry-After": "60"})

    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages")

    domanda = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            domanda = msg.get("content", "")
            break
    if not domanda:
        raise HTTPException(status_code=400, detail="No question found")

    # Detect repo name mentioned naturally in the question
    domanda_clean, repo_filter = _detect_repo_filter(domanda)

    # --- RAG ---
    contesto_completo = ""
    n_fonti = 0
    rag_ms = 0

    try:
        rag_start = time.time()
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in collections and qdrant.count(COLLECTION_NAME).count > 0:
            contesti = cerca_contesto(domanda_clean, repo_filter=repo_filter)
            blocchi = []
            for ctx in contesti:
                tag = f" [{ctx['source']}]" if ctx["source"] != "vector" else ""
                blocchi.append(f"--- File: {ctx['file']} (repo: {ctx['repo']}, score: {ctx['score']}{tag}) ---\n{ctx['codice']}")
            contesto_completo = "\n\n".join(blocchi)
            n_fonti = len(blocchi)
        rag_ms = (time.time() - rag_start) * 1000
    except Exception as e:
        logger.error(f"RAG error: {e}")
        metrics.record_error("rag_error")

    # --- Prompt ---
    history = openai_to_gemini_history(messages[:-1])
    user_content = (
        f"Context from codebase ({n_fonti} snippets):\n\n{contesto_completo}\n\n---\n\nQuestion: {domanda_clean}"
        if contesto_completo else domanda_clean
    )

    # --- Gemini ---
    try:
        chat = gemini_model.start_chat(history=history)

        if stream:
            return StreamingResponse(
                _stream_gemini(chat, user_content, domanda_clean, start_time, rag_ms, n_fonti),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await call_gemini_with_retry(chat, user_content)
        response_text = response.text

        latency_ms = (time.time() - start_time) * 1000
        tokens_in = sum(len(m.get("content", "")) // 4 for m in messages) + len(contesto_completo) // 4
        tokens_out = len(response_text) // 4

        metrics.record_query(latency_ms, rag_ms, tokens_in, tokens_out, had_context=n_fonti > 0)
        query_logger.log({
            "domanda": domanda[:200], "fonti": n_fonti, "latency_ms": round(latency_ms),
            "rag_ms": round(rag_ms), "tokens_in": tokens_in, "tokens_out": tokens_out,
        })

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "business-assistant",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
        }

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error(type(e).__name__)
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")


async def _stream_gemini(chat, user_content: str, domanda: str, start_time: float, rag_ms: float, n_fonti: int):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    full_response = []
    loop = asyncio.get_event_loop()

    try:
        # Get the streaming response in a thread (send_message is sync)
        response = await call_gemini_with_retry(chat, user_content, stream=True)

        # Iterate over chunks in a thread to avoid blocking the event loop
        def _iter_chunks():
            chunks = []
            for chunk in response:
                if chunk.text:
                    chunks.append(chunk.text)
            return chunks

        chunks = await loop.run_in_executor(None, _iter_chunks)

        for text in chunks:
            full_response.append(text)
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "business-assistant",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Send finish chunk
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "business-assistant",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

        # Record metrics
        full_text = "".join(full_response)
        latency_ms = (time.time() - start_time) * 1000
        tokens_in = len(user_content) // 4
        tokens_out = len(full_text) // 4
        metrics.record_query(latency_ms, rag_ms, tokens_in, tokens_out, had_context=n_fonti > 0)
        query_logger.log({
            "domanda": domanda[:200], "fonti": n_fonti, "latency_ms": round(latency_ms),
            "rag_ms": round(rag_ms), "tokens_in": tokens_in, "tokens_out": tokens_out, "stream": True,
        })

    except Exception as e:
        metrics.record_error(type(e).__name__)
        logger.error(f"Streaming error: {e}")
        # Send error as a proper SSE chunk so the client sees the message
        error_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "business-assistant",
            "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {str(e)}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


# ============================================================
# OPERATIONAL ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    qdrant_ok = False
    count = 0
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in collections:
            count = qdrant.count(COLLECTION_NAME).count
        qdrant_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": {"status": "ok" if qdrant_ok else "error", "vectors": count},
        "gemini": {"model": GEMINI_MODEL, "circuit_breaker": circuit_breaker.state},
        "rag": {"reranking": ENABLE_RERANKING, "hybrid_alpha": HYBRID_ALPHA, "reranker_model": RERANKER_MODEL if ENABLE_RERANKING else None},
    }


@app.get("/metrics")
async def get_metrics():
    return metrics.summary()


@app.get("/stats")
async def stats():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return {
            "collection": {
                "name": COLLECTION_NAME,
                "vectors": info.points_count,
                "indexed": info.indexed_vectors_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
            },
            "metrics": metrics.summary(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/reindex")
async def reindex():
    if _reindex_lock.locked():
        raise HTTPException(status_code=409, detail="Reindex already in progress")

    async with _reindex_lock:
        import subprocess
        try:
            result = subprocess.run(
                ["python", "/app/scripts/indexer.py"],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "QDRANT_URL": QDRANT_URL, "REPOS_PATH": "/data/repos"},
            )
            if result.returncode == 0:
                _load_known_repos()
            return {"status": "ok" if result.returncode == 0 else "error", "output": result.stdout[-1000:], "errors": result.stderr[-500:]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MIDDLEWARE: request logging
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    if request.url.path.startswith("/v1/chat"):
        ip = get_client_ip(request)
        logger.info(f"POST /v1/chat/completions → {response.status_code} ({elapsed:.0f}ms) IP={ip} remaining={rate_limiter.remaining(ip)}")
    return response
