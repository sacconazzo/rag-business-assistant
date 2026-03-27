#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') — Inizio re-indicizzazione"

# Se i repo sono git, aggiornali
if [ -d "repos" ]; then
    echo "📥 Git pull..."
    for repo_dir in repos/*/; do
        if [ -d "$repo_dir/.git" ]; then
            repo_name=$(basename "$repo_dir")
            echo "  → $repo_name"
            (cd "$repo_dir" && git pull --quiet 2>/dev/null) || echo "  ⚠️  Errore pull $repo_name"
        fi
    done
fi

echo "🔍 Indicizzazione..."
docker compose run --rm indexer

echo "$(date '+%Y-%m-%d %H:%M:%S') — Completato ✅"
