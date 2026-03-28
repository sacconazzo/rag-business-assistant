"""
Indexer — Scans repositories and indexes them into Qdrant.
Run with: docker compose run --rm indexer
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
    HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codebase")
REPOS_PATH = os.getenv("REPOS_PATH", "./repos")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF = int(os.getenv("HNSW_EF", "128"))
ENABLE_QUANTIZATION = os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"

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


def _get_overlap_lines(lines: list[str], max_overlap: int) -> tuple[list[str], int]:
    """Returns (overlap_lines, overlap_length) from the end of the given lines."""
    if max_overlap <= 0 or not lines:
        return [], 0
    overlap_lines = []
    overlap_len = 0
    for line in reversed(lines):
        overlap_len += len(line) + 1
        overlap_lines.append(line)
        if overlap_len >= max_overlap:
            break
    overlap_lines.reverse()
    return overlap_lines, overlap_len


def chunk_codice(testo: str, filepath: str, max_chars: int = CHUNK_MAX_CHARS,
                 overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    header = f"# File: {filepath}\n"
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
        "router.get(", "router.post(", "router.put(", "router.delete(", "router.patch(",
        "app.get(", "app.post(", "app.put(", "app.delete(", "app.use(",
        "module.exports",
    )

    for riga in righe:
        if any(riga.strip().startswith(s) for s in block_starters) and lunghezza > max_chars // 3:
            chunks.append(header + "\n".join(chunk_corrente))
            overlap_lines, overlap_len = _get_overlap_lines(chunk_corrente, overlap)
            chunk_corrente, lunghezza = overlap_lines, overlap_len

        chunk_corrente.append(riga)
        lunghezza += len(riga) + 1

        if lunghezza >= max_chars:
            chunks.append(header + "\n".join(chunk_corrente))
            overlap_lines, overlap_len = _get_overlap_lines(chunk_corrente, overlap)
            chunk_corrente, lunghezza = overlap_lines, overlap_len

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
        print(f"❌ Folder not found: {repos_path}")
        sys.exit(1)

    repo_dirs = sorted([d for d in os.listdir(repos_path) if os.path.isdir(os.path.join(repos_path, d)) and not d.startswith(".")])
    if not repo_dirs:
        print(f"❌ No repositories in: {repos_path}")
        print(f"   Check REPOS_HOST_PATH in .env")
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
                for i, chunk in enumerate(chunk_codice(contenuto, rel_path)):
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
    embedder = SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"attn_implementation": "eager"})
    vector_size = embedder.get_sentence_embedding_dimension()

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"🗑️  Drop collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"📦 Creating collection: {COLLECTION_NAME} (dim={vector_size})")
    quantization = ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
    ) if ENABLE_QUANTIZATION else None
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF),
        quantization_config=quantization,
    )

    # Indexes for filters and full-text search
    client.create_payload_index(collection_name=COLLECTION_NAME, field_name="repo", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name=COLLECTION_NAME, field_name="extension", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name=COLLECTION_NAME, field_name="file", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(
        collection_name=COLLECTION_NAME, field_name="content",
        field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True),
    )

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
    print(f"✂️  chunk max: {CHUNK_MAX_CHARS} chars, overlap: {CHUNK_OVERLAP_CHARS} chars")
    print(f"🔧 HNSW m={HNSW_M} ef={HNSW_EF}, quantization={'on' if ENABLE_QUANTIZATION else 'off'}")
    print()

    t = time.time()
    docs = scan_repos(REPOS_PATH)
    if not docs:
        print("❌ No documents found!")
        sys.exit(1)

    print(f"\n📊 {len(docs)} chunks to index")
    n = indicizza(docs)
    print(f"\n✅ Completed in {time.time() - t:.1f}s — {n} vectors")
    print("=" * 60)


if __name__ == "__main__":
    main()
