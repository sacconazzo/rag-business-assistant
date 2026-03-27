"""
Indexer — Scansiona i repository e indicizza in Qdrant.
Esegui con: docker compose run --rm indexer
"""

import os
import sys
import glob
import time
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
)
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codebase")
REPOS_PATH = os.getenv("REPOS_PATH", "./repos")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

FILE_EXTENSIONS = [
    "*.py", "*.js", "*.ts", "*.tsx", "*.jsx",
    "*.java", "*.kt", "*.scala", "*.cs", "*.vb",
    "*.go", "*.rs", "*.rb", "*.php",
    "*.c", "*.cpp", "*.h", "*.hpp", "*.swift", "*.m",
    "*.sql",
    "*.md", "*.txt", "*.rst",
    "*.yaml", "*.yml", "*.toml", "*.json",
]

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    "env", ".env", "dist", "build", ".next", ".nuxt",
    "target", "bin", "obj", ".idea", ".vscode",
    "vendor", "packages", ".tox", "coverage",
    "test_data", "fixtures", "migrations",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "composer.lock",
    ".DS_Store", "Thumbs.db",
}

MAX_FILE_SIZE = 100_000


def chunk_codice(testo: str, filepath: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    filename = os.path.basename(filepath)
    header = f"# File: {filename}\n"
    righe = testo.split("\n")
    chunks, chunk_corrente, lunghezza = [], [], 0

    block_starters = (
        "def ", "class ", "function ", "async function ",
        "public ", "private ", "protected ", "internal ",
        "export ", "const ", "let ", "var ",
        "## ", "# ", "### ",
        "func ", "fn ", "impl ",
        "interface ", "struct ", "enum ",
        "module ", "namespace ",
        "describe(", "it(", "test(",
    )

    for riga in righe:
        if any(riga.strip().startswith(s) for s in block_starters) and lunghezza > max_chars // 3:
            chunks.append(header + "\n".join(chunk_corrente))
            chunk_corrente, lunghezza = [], 0

        chunk_corrente.append(riga)
        lunghezza += len(riga) + 1

        if lunghezza >= max_chars:
            chunks.append(header + "\n".join(chunk_corrente))
            chunk_corrente, lunghezza = [], 0

    if chunk_corrente:
        chunks.append(header + "\n".join(chunk_corrente))

    return [c for c in chunks if len(c.strip()) > 50]


def should_skip(filepath: str) -> bool:
    parts = filepath.split(os.sep)
    if any(d in SKIP_DIRS for d in parts):
        return True
    if os.path.basename(filepath) in SKIP_FILES:
        return True
    try:
        return os.path.getsize(filepath) > MAX_FILE_SIZE
    except OSError:
        return True


def scan_repos(repos_path: str) -> list[dict]:
    if not os.path.exists(repos_path):
        print(f"❌ Cartella non trovata: {repos_path}")
        sys.exit(1)

    repo_dirs = sorted([d for d in os.listdir(repos_path) if os.path.isdir(os.path.join(repos_path, d)) and not d.startswith(".")])
    if not repo_dirs:
        print(f"❌ Nessun repository in: {repos_path}")
        print(f"   Controlla REPOS_HOST_PATH nel .env")
        sys.exit(1)

    documenti = []
    for repo_name in repo_dirs:
        repo_path = os.path.join(repos_path, repo_name)
        file_count, chunk_count = 0, 0

        for ext in FILE_EXTENSIONS:
            for filepath in glob.glob(os.path.join(repo_path, "**", ext), recursive=True):
                if should_skip(filepath):
                    continue
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        contenuto = f.read()
                except Exception:
                    continue
                if not contenuto.strip() or len(contenuto) < 20:
                    continue

                file_count += 1
                rel_path = os.path.relpath(filepath, repos_path)
                for i, chunk in enumerate(chunk_codice(contenuto, filepath)):
                    documenti.append({
                        "id": hashlib.md5(f"{rel_path}::chunk_{i}".encode()).hexdigest(),
                        "content": chunk,
                        "file": rel_path,
                        "repo": repo_name,
                        "chunk_index": i,
                        "extension": os.path.splitext(filepath)[1],
                    })
                    chunk_count += 1

        print(f"📂 {repo_name}: {file_count} file → {chunk_count} chunk")
    return documenti


def indicizza(documenti: list[dict]):
    print(f"\n🔗 Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    print(f"🧠 Embedding: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = embedder.get_sentence_embedding_dimension()

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"🗑️  Drop collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"📦 Creazione collection: {COLLECTION_NAME} (dim={vector_size})")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Indici per filtri e full-text search
    client.create_payload_index(collection_name=COLLECTION_NAME, field_name="repo", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name=COLLECTION_NAME, field_name="extension", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(
        collection_name=COLLECTION_NAME, field_name="content",
        field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD, min_token_len=2, max_token_len=20),
    )

    BATCH_SIZE = 128
    total = len(documenti)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = documenti[start:end]
        embeddings = embedder.encode([d["content"] for d in batch], show_progress_bar=False)
        points = [
            PointStruct(id=d["id"], vector=emb.tolist(), payload={
                "content": d["content"], "file": d["file"], "repo": d["repo"],
                "chunk_index": d["chunk_index"], "extension": d["extension"],
            })
            for d, emb in zip(batch, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  ⏳ {min(100, int(end / total * 100))}% ({end}/{total})")

    return client.count(COLLECTION_NAME).count


def main():
    print("=" * 60)
    print("🔍 RAG Indexer")
    print("=" * 60)
    print(f"📁 {REPOS_PATH}")
    print(f"🔗 {QDRANT_URL}")
    print(f"✂️  chunk max: {CHUNK_MAX_CHARS} chars")
    print()

    t = time.time()
    docs = scan_repos(REPOS_PATH)
    if not docs:
        print("❌ Nessun documento!")
        sys.exit(1)

    print(f"\n📊 {len(docs)} chunk da indicizzare")
    n = indicizza(docs)
    print(f"\n✅ Completato in {time.time() - t:.1f}s — {n} vettori")
    print("=" * 60)


if __name__ == "__main__":
    main()
