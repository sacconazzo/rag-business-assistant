"""
RAG Proxy — Production
Resilienza (retry, rate limit, circuit breaker)
Osservabilità (logging strutturato, metriche, costi)
Qualità RAG (hybrid search, reranking con cross-encoder)
"""

import os
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime
from collections import defaultdict

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, NamedVector, Query
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_QUERY_LOG = os.getenv("ENABLE_QUERY_LOG", "true").lower() == "true"

# Resilienza
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_DELAY = float(os.getenv("GEMINI_RETRY_DELAY", "1.0"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# RAG avanzato
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TRUNCATE = int(os.getenv("RERANK_TRUNCATE", "512"))

SYSTEM_PROMPT = """Sei un assistente esperto della nostra codebase aziendale.
Rispondi alle domande sulla logica di business basandoti ESCLUSIVAMENTE
sui frammenti di codice e documentazione forniti come contesto.

Regole:
- Se il contesto non contiene informazioni sufficienti, dillo chiaramente.
- Cita il file e il repository da cui prendi le informazioni.
- Spiega la logica in modo chiaro, come se parlassi con un collega.
- Se trovi incongruenze nel codice, segnalale.
- Rispondi nella stessa lingua della domanda.
"""


# ============================================================
# LOGGING STRUTTURATO
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag-proxy")


class QueryLogger:
    """Scrive ogni query in un file JSONL per analisi successive."""

    def __init__(self, log_dir="/app/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "queries.jsonl")

    def log(self, entry: dict):
        if not ENABLE_QUERY_LOG:
            return
        entry["timestamp"] = datetime.utcnow().isoformat()
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Errore scrittura log: {e}")


query_logger = QueryLogger()


# ============================================================
# METRICHE
# ============================================================

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
        # Tieni solo le ultime 1000
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
            self.rag_latencies = self.rag_latencies[-1000:]

    def record_error(self, error_type: str):
        self.total_errors += 1
        self.errors_by_type[error_type] += 1

    def _percentile(self, data: list, p: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def summary(self) -> dict:
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_rag = sum(self.rag_latencies) / len(self.rag_latencies) if self.rag_latencies else 0

        # Stima costi Gemini Flash ($/M token): input=0.075, output=0.30
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

    last_error = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            if stream:
                result = chat.send_message(content, stream=True)
            else:
                result = chat.send_message(content)
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
# INIT
# ============================================================

qdrant: QdrantClient = None
embedder: SentenceTransformer = None
reranker: CrossEncoder = None
gemini_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, embedder, reranker, gemini_model

    qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
    collections = [c.name for c in qdrant.get_collections().collections]
    count = qdrant.count(COLLECTION_NAME).count if COLLECTION_NAME in collections else 0
    logger.info(f"Qdrant: {QDRANT_URL} — {count} vettori")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Embedding: {EMBEDDING_MODEL} (dim={embedder.get_sentence_embedding_dimension()})")

    if ENABLE_RERANKING:
        reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Reranker: {RERANKER_MODEL}")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY obbligatoria")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    logger.info(f"Gemini: {GEMINI_MODEL}")

    yield
    logger.info("Shutdown")


app = FastAPI(title="RAG Proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# HYBRID SEARCH + RERANKING
# ============================================================

def cerca_contesto(domanda: str, n_risultati: int = MAX_RISULTATI) -> list[dict]:
    """
    1. Ricerca vettoriale (semantica)
    2. Ricerca full-text (keyword)
    3. Fusione con peso HYBRID_ALPHA
    4. Reranking con cross-encoder
    """
    query_vector = embedder.encode(domanda).tolist()
    candidates = RERANK_CANDIDATES if ENABLE_RERANKING else n_risultati

    # --- Ricerca vettoriale ---
    vector_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=candidates,
        with_payload=True,
    ).points

    # --- Ricerca full-text ---
    text_results = []
    try:
        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="content", match=MatchText(text=domanda))]),
            limit=candidates,
            with_payload=True,
            with_vectors=False,
        )
        text_results = scroll_result[0] if scroll_result else []
    except Exception as e:
        logger.debug(f"Full-text fallback: {e}")

    # --- Fusione ---
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
            # Boost per risultati presenti in entrambe le ricerche
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
        raise HTTPException(status_code=429, detail="Troppe richieste. Riprova tra un minuto.", headers={"Retry-After": "60"})

    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="Nessun messaggio")

    domanda = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            domanda = msg.get("content", "")
            break
    if not domanda:
        raise HTTPException(status_code=400, detail="Nessuna domanda")

    # --- RAG ---
    contesto_completo = ""
    n_fonti = 0
    rag_ms = 0

    try:
        rag_start = time.time()
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in collections and qdrant.count(COLLECTION_NAME).count > 0:
            contesti = cerca_contesto(domanda)
            blocchi = []
            for ctx in contesti:
                tag = f" [{ctx['source']}]" if ctx["source"] != "vector" else ""
                blocchi.append(f"--- File: {ctx['file']} (repo: {ctx['repo']}, score: {ctx['score']}{tag}) ---\n{ctx['codice']}")
            contesto_completo = "\n\n".join(blocchi)
            n_fonti = len(blocchi)
        rag_ms = (time.time() - rag_start) * 1000
    except Exception as e:
        logger.error(f"Errore RAG: {e}")
        metrics.record_error("rag_error")

    # --- Prompt ---
    history = openai_to_gemini_history(messages[:-1])
    user_content = (
        f"Contesto dalla codebase ({n_fonti} frammenti):\n\n{contesto_completo}\n\n---\n\nDomanda: {domanda}"
        if contesto_completo else domanda
    )

    # --- Gemini ---
    try:
        chat = gemini_model.start_chat(history=history)

        if stream:
            return StreamingResponse(
                _stream_gemini(chat, user_content, domanda, start_time, rag_ms, n_fonti),
                media_type="text/event-stream",
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
        logger.error(f"Errore Gemini: {e}")
        raise HTTPException(status_code=502, detail=f"Errore Gemini: {str(e)}")


async def _stream_gemini(chat, user_content: str, domanda: str, start_time: float, rag_ms: float, n_fonti: int):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    full_response = []

    try:
        response = await call_gemini_with_retry(chat, user_content, stream=True)

        for chunk in response:
            if chunk.text:
                full_response.append(chunk.text)
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'business-assistant', 'choices': [{'index': 0, 'delta': {'content': chunk.text}, 'finish_reason': None}]})}\n\n"

        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'business-assistant', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        full_text = "".join(full_response)
        latency_ms = (time.time() - start_time) * 1000
        tokens_out = len(full_text) // 4
        metrics.record_query(latency_ms, rag_ms, 0, tokens_out, had_context=n_fonti > 0)
        query_logger.log({"domanda": domanda[:200], "fonti": n_fonti, "latency_ms": round(latency_ms), "rag_ms": round(rag_ms), "tokens_out": tokens_out, "stream": True})

    except Exception as e:
        metrics.record_error(type(e).__name__)
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'business-assistant', 'choices': [{'index': 0, 'delta': {'content': f'\\n\\n⚠️ Errore: {str(e)}'}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"


# ============================================================
# ENDPOINTS OPERATIVI
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
        "qdrant": {"status": "ok" if qdrant_ok else "error", "vettori": count},
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
                "vettori": info.points_count,
                "indicizzati": info.indexed_vectors_count,
                "dimensione_vettore": info.config.params.vectors.size,
                "distanza": info.config.params.vectors.distance.value,
            },
            "metriche": metrics.summary(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/reindex")
async def reindex():
    import subprocess
    try:
        result = subprocess.run(
            ["python", "/app/scripts/indexer.py"],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "QDRANT_URL": QDRANT_URL, "REPOS_PATH": "/data/repos"},
        )
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
