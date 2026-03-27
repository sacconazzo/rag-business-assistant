# 🤖 RAG Business Assistant

Assistente AI per interrogare la logica di business dei tuoi repository.
**Qdrant** + **Google Gemini** + **Open WebUI**.

---

## 📋 Architettura

```
         CDN / Load Balancer
                │
        ┌───────┴───────┐
        │  Open WebUI   │  ← interfaccia utente
        │  (porta 3000) │
        └───────┬───────┘
                │
        ┌───────┴────────────────┐     ┌──────────────┐
        │  RAG Proxy (FastAPI)   │────▶│  Gemini API  │
        │  (porta 8001)          │     └──────────────┘
        │                        │
        │  • Ricerca ibrida      │
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

## 🚀 Avvio rapido

### Prerequisiti

- Docker e Docker Compose
- Chiave API Gemini → https://aistudio.google.com/apikey
- Cartella contenente i tuoi repository

### 1. Configura

```bash
cd rag-business-assistant
cp .env.example .env
```

Modifica `.env`:
```env
GEMINI_API_KEY=AIzaSy-la-tua-chiave
REPOS_HOST_PATH=/home/mario/projects   # la tua cartella, così com'è
```

La cartella deve contenere sottocartelle (ogni sottocartella = un repository):
```
/home/mario/projects/
├── repo-api/
├── repo-frontend/
├── repo-services/
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

1. Apri http://localhost:3000, crea l'account admin
2. **Pannello Admin → Impostazioni → Connessioni**
3. Aggiungi OpenAI API: URL `http://rag-proxy:8001/v1`, Chiave `not-needed`
4. Seleziona il modello **"business-assistant"** nella chat

---

## 🔍 Qualità RAG

Il sistema usa tre tecniche per massimizzare la rilevanza dei risultati:

### Ricerca ibrida
Combina ricerca **vettoriale** (semantica, cattura il significato) con ricerca **full-text** (keyword, cattura termini esatti). Il peso è configurabile tramite `HYBRID_ALPHA`:
- `1.0` = solo vettoriale
- `0.0` = solo full-text
- `0.65` = default, bilanciato per codice (più peso al keyword matching per nomi funzioni/variabili)

I risultati presenti in entrambe le ricerche ricevono un bonus di punteggio.

### Reranking
Dopo la ricerca, un **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rivaluta ogni risultato confrontandolo direttamente con la domanda. Più lento della sola ricerca vettoriale, ma molto più preciso perché il cross-encoder "legge" domanda e contesto insieme.

Configurabile: `ENABLE_RERANKING=true/false`, `RERANK_CANDIDATES=30` (quanti candidati valutare).

### Chunking intelligente
Il codice viene suddiviso in blocchi logici (funzioni, classi, route handler Express), non per righe fisse. Ogni chunk include il percorso completo del file come header per preservare il contesto. I chunk consecutivi si sovrappongono di `CHUNK_OVERLAP_CHARS` caratteri (default 200) per evitare perdita di contesto ai confini.

---

## 🧠 Modelli embedding

### Modello predefinito: `BAAI/bge-base-en-v1.5`
Modello general-purpose con ottimi punteggi nei benchmark MTEB per code search. 110M parametri, 768 dimensioni, ~440MB. Gira bene su CPU (MacBook Pro M3/M4).

### `EMBEDDING_QUERY_PREFIX`
BGE è un modello **instruction-following**: richiede un prefisso testuale sulle query per ottenere risultati ottimali. I documenti vengono indicizzati **senza** prefisso, mentre le query di ricerca lo usano. Il valore predefinito è `Represent this sentence for searching relevant passages: `. Se si passa a un modello senza prefisso (es. `all-mpnet-base-v2`), basta svuotare la variabile.

### Modello alternativo consigliato: `jinaai/jina-embeddings-v2-base-code`
Specifico per codice sorgente, con finestra di contesto da 8K token (vs 512 di BGE). Ideale se la priorità è la comprensione del codice JavaScript e il tempo di indicizzazione non è critico. Per usarlo:
```env
EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code
EMBEDDING_QUERY_PREFIX=
```
> ⚠️ Cambiare modello richiede la reindicizzazione completa: `docker compose run --rm indexer`

### Altre alternative
- `all-mpnet-base-v2` — buon modello general-purpose (768 dim, nessun prefisso necessario)

---

## 🛡️ Resilienza

### Retry con backoff esponenziale
Se Gemini restituisce un errore temporaneo (500, 429, timeout), il proxy ritenta automaticamente con ritardo crescente: 1s → 2s → 4s. Non ritenta su errori client (chiave API invalida, richiesta malformata).

### Circuit breaker
Se Gemini fallisce 5 volte consecutive, il circuito si **apre**: il proxy smette di chiamare Gemini per 30 secondi e restituisce 503. Dopo il cooldown, prova una richiesta di test. Se ha successo, riprende normalmente.

Questo evita di accumulare timeout durante un'interruzione del servizio.

### Rate limiting
Limite configurabile per IP (default: 30 richieste/minuto). Restituisce 429 con header `Retry-After`.

---

## 📊 Osservabilità

### Endpoint metriche

```bash
# Health check (per CDN / load balancer)
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

### Log delle query

Ogni domanda viene loggata in `/app/logs/queries.jsonl` con:
- Testo della domanda (troncato a 200 caratteri)
- Numero di fonti trovate
- Latenza totale e latenza RAG
- Token consumati
- Timestamp

Il volume `logs-data` persiste tra i riavvii. Puoi analizzare i log con `jq`:

```bash
# Domande senza contesto (indica problemi di indicizzazione)
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq 'select(.fonti == 0)'

# Domande più lente
docker compose exec rag-proxy cat /app/logs/queries.jsonl | jq -s 'sort_by(-.latency_ms) | .[0:10]'

# Costo giornaliero approssimativo
curl -s http://localhost:8001/metrics | jq '.cost_usd'
```

### Dashboard Qdrant
http://localhost:6333/dashboard — vettori indicizzati, performance, stato della collection.

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
| `GEMINI_API_KEY` | (obbligatorio) | Chiave API Gemini |
| `REPOS_HOST_PATH` | (obbligatorio) | Percorso alla cartella dei repository |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modello Gemini |
| `MAX_RISULTATI` | `8` | Chunk nel contesto |
| `CHUNK_MAX_CHARS` | `1500` | Dimensione massima chunk (caratteri) |
| `CHUNK_OVERLAP_CHARS` | `200` | Sovrapposizione tra chunk consecutivi (caratteri) |
| `ENABLE_RERANKING` | `true` | Reranking con cross-encoder |
| `RERANK_CANDIDATES` | `30` | Candidati pre-reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modello cross-encoder |
| `RERANK_TRUNCATE` | `512` | Caratteri massimi passati al reranker |
| `HYBRID_ALPHA` | `0.65` | Peso vettoriale vs full-text |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Modello sentence-transformer |
| `EMBEDDING_QUERY_PREFIX` | `Represent this sentence...` | Prefisso query per modelli instruction-following (vedi sezione Modelli embedding) |
| `BATCH_SIZE` | `64` | Dimensione batch indicizzazione |
| `HNSW_M` | `16` | Connessioni per nodo HNSW |
| `HNSW_EF` | `128` | Candidati costruzione HNSW |
| `ENABLE_QUANTIZATION` | `true` | Quantizzazione scalare INT8 |
| `GEMINI_MAX_RETRIES` | `3` | Tentativi in caso di errore |
| `GEMINI_RETRY_DELAY` | `1.0` | Ritardo base retry (sec) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Richieste max/min per IP |
| `LOG_LEVEL` | `INFO` | Livello di log |
| `ENABLE_QUERY_LOG` | `true` | Salva log delle query su file |

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
