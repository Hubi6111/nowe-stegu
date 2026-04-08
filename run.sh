#!/usr/bin/env bash
# ═══ Stegu Visualizer — single command start ═══
# Usage:  bash run.sh
set -euo pipefail
cd "$(dirname "$0")"

cleanup() {
  echo ""
  echo "Stopping…"
  [ -n "${PID_API:-}" ]      && kill "$PID_API"      2>/dev/null || true
  [ -n "${PID_FRONTEND:-}" ] && kill "$PID_FRONTEND"  2>/dev/null || true
  wait 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Load .env ─────────────────────────────────────────────
if [ -f .env ]; then
  set -a; source .env; set +a
fi

for _var in TEXTURES_DIR DATA_DIR MODEL_CACHE_DIR; do
  _val="${!_var:-}"
  if [ -n "$_val" ] && [[ "$_val" != /* ]]; then
    export "$_var"="$(pwd)/$_val"
  fi
done
mkdir -p data/uploads data/outputs data/masks data/model-cache

# ── Auto-install if needed ────────────────────────────────
if [ ! -f api/.venv/bin/uvicorn ]; then
  echo "→ Setting up Python venv…"
  python3 -m venv api/.venv
  api/.venv/bin/pip install -q -r api/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  echo "→ Installing npm deps…"
  (cd frontend && npm install --silent)
fi

echo ""
echo "═══ Stegu Visualizer ═══"
echo ""

# ── 1) API (FastAPI :8000) ────────────────────────────────
echo "→ API   http://localhost:8000"
(cd api && ../api/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning) &
PID_API=$!

# Wait for API
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "  ✓ API ready"
    break
  fi
  sleep 0.5
done

# ── 2) Frontend (Next.js :3000) ───────────────────────────
echo "→ Frontend http://localhost:3000"
(cd frontend && npm run dev 2>&1) &
PID_FRONTEND=$!

echo ""
echo "  ┌──────────────────────────────────────┐"
echo "  │  Open:  http://localhost:3000         │"
echo "  │  API:   http://localhost:8000         │"
echo "  │  Stop:  Ctrl+C                       │"
echo "  └──────────────────────────────────────┘"
echo ""

wait
