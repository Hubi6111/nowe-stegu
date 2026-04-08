PROJECT_ROOT := $(shell pwd)

.PHONY: setup dev stop clean

## ── Setup ──────────────────────────────────────────────────

setup:
	@echo "═══ Stegu Visualizer — Setup ═══"
	@mkdir -p data/uploads data/outputs data/masks data/model-cache
	@echo "→ Installing frontend dependencies..."
	cd frontend && npm install
	@echo "→ Creating API virtual environment..."
	python3 -m venv api/.venv
	api/.venv/bin/pip install --upgrade pip -q
	api/.venv/bin/pip install -r api/requirements.txt
	@echo "→ Creating inference virtual environment..."
	python3 -m venv inference/.venv
	inference/.venv/bin/pip install --upgrade pip -q
	inference/.venv/bin/pip install -r inference/requirements.txt
	@echo ""
	@echo "✓ Setup complete. Copy .env.example → .env and fill in your keys."
	@echo "  Then run: make dev"

## ── Development — full stack (inference :8001 + API :8000 + Next :3000) ─

dev:
	@bash scripts/start-all.sh

stop:
	@bash scripts/stop-all.sh

## ── Individual services ────────────────────────────────────

dev-frontend:
	cd frontend && npm run dev

dev-api:
	cd api && ../.env 2>/dev/null; .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

dev-inference:
	cd inference && ../.env 2>/dev/null; .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload

## ── Cleanup ────────────────────────────────────────────────

clean:
	rm -rf api/.venv inference/.venv frontend/node_modules frontend/.next
	rm -f .pid-*
	@echo "Cleaned. Run 'make setup' to reinstall."
