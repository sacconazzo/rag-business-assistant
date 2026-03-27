# 🤖 RAG Business Assistant

Assistente AI per domande sulla logica di business dei tuoi repository.
**Qdrant** + **Google Gemini** + **Open WebUI**.

Production-ready: hybrid search, reranking, circuit breaker, retry, metriche, logging.

---

## 📋 Architettura

```
         CDN / Load Balancer
                │
        ┌───────┴───────┐
        │  Open WebUI   │  ← la tua interface
        │  (porta 3000) │
        └───────┬───────┘
                │
        ┌───────┴────────────────┐     ┌──────────────┐
        │  RAG Proxy (FastAPI)   │────▶│  Gemini API  │
        │  (porta 8001)          │     └──────────────┘
        │                        │
        │  • Hybrid search       │
        │  • Reranking           │
        │  • Circuit breaker     │
        │  • Rate limiting       │
        │  • Retry + backoff     │
        │  • Metriche + logging  │
        └───────┬────────────────┘
                │
        ┌───────┴───────┐
        │    Qdrant     │
        │  (porta 6333) │
        └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisiti

- Docker e Docker Compose
- Chiave API Gemini → https://aistudio.google.com/apikey
- Cartella con i tuoi repository

### 1. Configura

```bash
cd rag-business-assistant
cp .env.example .env
```

Modifica nel `.env`:
```env
GEMINI_API_KEY=AIzaSy-la-tua-chiave
REPOS_HOST_PATH=/home/mario/progetti   # la tua cartella, così com'è
```

La cartella deve contenere sottocartelle (ogni sottocartella = un repo):
```
/home/mario/progetti/
├── repo-api/
├── repo-frontend/
├── repo-servizi/
└── ...
```

### 2. Indicizza

```bash
docker compose up -d qdrant
sleep 10
docker compose run --rm indexer
```

### 3. Avvia

```bash
docker compose up -d
```

### 4. Configura Open WebUI

1. Apri http://localhost:3000, crea account admin
2. **Admin Panel → Settings → Connections**
3. Aggiungi OpenAI API: URL `http://rag-proxy:8001/v1`, Key `not-needed`
4. Seleziona modello **"business-assistant"** nella chat

---

## 🔍 Qualità RAG

Il sistema usa tre tecniche per massimizzare la pertinenza:

### Hybrid Search
Combina ricerca **vettoriale** (semantica, cattura il significato) con ricerca **full-text** (keyword, cattura termini esatti). Il peso è configurabile via `HYBRID_ALPHA`:
- `1.0` = solo vettoriale
- `0.0` = solo full-text
- `0.7` = default, buon bilanciamento

I risultati che appaiono in entrambe le ricerche ricevono un boost.

### Reranking
Dopo la ricerca, un **cross-encoder** (ms-marco-MiniLM-L-6-v2) rivaluta ogni risultato confrontandolo direttamente con la domanda. Più lento della sola ricerca vettoriale, ma molto più preciso perché il cross-encoder "legge" domanda e contesto insieme.

Configurabile: `ENABLE_RERANKING=true/false`, `RERANK_CANDIDATES=30` (quanti candidati valutare).

### Chunking intelligente
Il codice viene spezzato per blocchi logici (funzioni, classi, metodi), non a righe fisse. Ogni chunk include il nome del file come header per mantenere il contesto.

---

## 🛡️ Resilienza

### Retry con backoff esponenziale
Se Gemini restituisce un errore temporaneo (500, 429, timeout), il proxy ritenta automaticamente con attesa crescente: 1s → 2s → 4s. Non ritenta su errori client (API key invalida, richiesta malformata).

### Circuit breaker
Se Gemini fallisce 5 volte consecutive, il circuito si **apre**: il proxy smette di chiamare Gemini per 30 secondi e restituisce 503. Dopo il cooldown, prova una richiesta di test. Se va a buon fine, riprende normalmente.

Questo evita di accumulare timeout durante un'interruzione di servizio.

### Rate limiting
Limite configurabile per IP (default: 30 richieste/minuto). Restituisce 429 con header `Retry-After`.

---

## 📊 Osservabilità

### Endpoint metriche

```bash
# Health check (per il CDN / load balancer)
curl http://localhost:8001/health

# Metriche dettagliate
curl http://localhost:8001/metrics

# Statistiche Qdrant + metriche
curl http://localhost:8001/stats
```

### Cosa trovi in /metrics

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

Ogni domanda viene registrata in `/app/logs/queries.jsonl` con:
- Testo della domanda (troncato a 200 char)
- Numero di fonti trovate
- Latenza totale e latenza RAG
- Token consumati
- Timestamp

Il volume `logs-data` persiste tra i restart. Puoi analizzare i log con `jq`:

```bash
# Domande senza contesto (indica problemi di indicizzazione)
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq 'select(.fonti == 0)'

# Domande più lente
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq -s 'sort_by(-.latency_ms) | .[0:10]'

# Costo giornaliero approssimativo
curl -s http://localhost:8001/metrics | jq '.cost_usd'
```

### Dashboard Qdrant
http://localhost:6333/dashboard — vettori indicizzati, performance, stato collection.

---

## 🔄 Aggiornare l'indice

```bash
docker compose run --rm indexer
```

Oppure cron notturno:
```cron
0 3 * * * cd /path/to/rag-business-assistant && bash scripts/reindex.sh >> logs/reindex.log 2>&1
```

Oppure via API:
```bash
curl -X POST http://localhost:8001/reindex
```

---

## ⚙️ Configurazione

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `GEMINI_API_KEY` | (obblig.) | Chiave Gemini |
| `REPOS_HOST_PATH` | (obblig.) | Percorso cartella repository |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modello Gemini |
| `MAX_RISULTATI` | `8` | Chunk nel contesto |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking |
| `RERANK_CANDIDATES` | `30` | Candidati pre-reranking |
| `HYBRID_ALPHA` | `0.7` | Peso vettoriale vs full-text |
| `GEMINI_MAX_RETRIES` | `3` | Tentativi su errore |
| `GEMINI_RETRY_DELAY` | `1.0` | Delay base retry (sec) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max richieste/min per IP |
| `LOG_LEVEL` | `INFO` | Livello log |
| `ENABLE_QUERY_LOG` | `true` | Log query su file |

---

## 📁 Struttura

```
rag-business-assistant/
├── docker-compose.yml
├── Dockerfile.proxy
├── Dockerfile.indexer
├── .env.example
├── .gitignore
├── README.md
├── app/
│   └── rag_proxy.py        # RAG + Gemini + resilienza + metriche
└── scripts/
    ├── indexer.py            # Indicizzazione → Qdrant
    └── reindex.sh            # Aggiornamento automatico
```
