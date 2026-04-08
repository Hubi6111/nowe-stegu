#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "═══ Stegu Visualizer — Setup ═══"
echo ""

mkdir -p data/uploads data/outputs data/masks data/model-cache

echo "→ Frontend (npm install)..."
(cd frontend && npm install)

echo "→ API service (Python venv)..."
python3 -m venv api/.venv
api/.venv/bin/pip install --upgrade pip -q
api/.venv/bin/pip install -r api/requirements.txt

echo "→ Inference service (Python venv)..."
python3 -m venv inference/.venv
inference/.venv/bin/pip install --upgrade pip -q
inference/.venv/bin/pip install -r inference/requirements.txt

echo ""
if [ ! -f .env ]; then
  cp .env.example .env
  echo "✓ Created .env from .env.example — fill in GEMINI_API_KEY"
else
  echo "✓ .env already exists"
fi

echo "✓ Setup complete. Run: make dev"
