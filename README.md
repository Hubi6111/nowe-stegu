# Stegu Visualizer

Wall texture visualization tool. Upload a room photo, select the wall area, and preview Stegu decorative stone and brick products applied to your wall.

**Live (Vercel):** https://stegu-visualizer.vercel.app

---

## Quick Start — macOS

### 1. One-time setup

```bash
make setup
```

This installs all dependencies (Node, Python venvs) and creates `frontend/.env.local`.

### 2. Add your Gemini API key

Open **`.env`** in the project root and paste your key:

```
GEMINI_API_KEY=your-key-here
```

That is the **only file you need to edit**. Everything else is already configured.

> Get a key at https://aistudio.google.com/apikey

### 3. Start the app

**Option A — Double-click (macOS Finder):**

Double-click **`run.command`** in the project root.  
A Terminal window opens and all services start automatically.

**Option B — Terminal:**

```bash
bash scripts/start-all.sh
```

**Option C — Make:**

```bash
make dev
```

### 4. Open in browser

```
http://localhost:3000
```

### 5. Stop everything

```bash
bash scripts/stop-all.sh
```

Or: `make stop`

---

## Project Structure

```
stegu-visualizer/
├── run.command            ← double-click to start (macOS)
├── .env                   ← paste GEMINI_API_KEY here (never commit)
├── .env.example           ← env template (safe to commit)
├── Makefile               ← setup / dev / stop / clean
├── scripts/
│   ├── start-all.sh       ← starts all 3 services
│   └── stop-all.sh        ← stops all 3 services
│
├── frontend/              ← Next.js 15 · TypeScript · Tailwind · react-konva
│   └── .env.local         ← Next.js env (API_PROXY_URL, NEXT_PUBLIC_API_URL)
│
├── api/                   ← FastAPI orchestrator (port 8000)
│   ├── routers/           — products, pipeline endpoints
│   └── services/          — inference client, Gemini, texture projection
│
├── inference/             ← PyTorch inference service (port 8001)
│   └── models/            — OneFormer detector, SAM2 refiner
│
├── assets/
│   └── textures/stegu/    ← product texture library
│
├── data/
│   ├── uploads/           ← user photos (not committed)
│   ├── outputs/           ← generated results (not committed)
│   ├── masks/             ← wall masks (not committed)
│   └── model-cache/       ← ML model weights (not committed)
│
├── backend/               ← legacy FastAPI (Vercel serverless only)
└── vercel.json            ← Vercel deployment config
```

---

## Environment Variables

All variables live in **`.env`** (project root). This is the only file you edit locally.

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Server-side only — never sent to browser |
| `GEMINI_IMAGE_MODEL` | `gemini-3-pro-image-preview` | Final image refinement model |
| `GEMINI_REASONING_MODEL` | `gemini-3.1-pro-preview` | Orchestration / reasoning model |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base URL for browser-side code |
| `API_BASE_URL` | `http://localhost:8000` | API base URL for server-side code |
| `INFERENCE_BASE_URL` | `http://localhost:8001` | Inference service URL |

---

## Services

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | Next.js UI |
| API | http://localhost:8000/api/health | FastAPI orchestrator |
| Inference | http://localhost:8001/health | OneFormer + SAM 2 |

---

## Generation Pipeline

1. Upload room photo
2. Draw wall polygon
3. Click **Generate Visualization**
   - OneFormer detects wall pixels within the polygon
   - SAM 2 refines mask edges
   - Deterministic texture projection tiles the product onto the wall
   - Gemini Pro improves realism (lighting, shadows, edge blending)
4. Download final result

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service health check |
| GET | `/api/products` | Product list from texture library |
| GET | `/api/textures/:id` | Serves product albedo.jpg |
| POST | `/api/wall-detect` | Polygon → wall mask |
| POST | `/api/wall-refine` | Refine mask with SAM 2 |
| POST | `/api/texture-project` | Deterministic texture projection |
| POST | `/api/render-refine` | Gemini AI realism refinement |
| POST | `/api/generate-visualization` | Full pipeline in one call |

---

## Product Library

Each product is a folder in `assets/textures/stegu/{slug}/` containing:

- `albedo.jpg` — texture image
- `metadata.json` — product metadata

```json
{
  "name": "Cambridge 1",
  "moduleWidthMm": 245,
  "moduleHeightMm": 65,
  "jointMm": 10,
  "layoutType": "running-bond",
  "offsetRatio": 0.5
}
```

---

## Vercel Deployment

The app stays deployable to Vercel. The `backend/` directory is the serverless entry point.

```bash
vercel deploy --prod
```

---

## Troubleshooting

**"make setup not found"** — install Xcode Command Line Tools: `xcode-select --install`

**Port already in use** — run `bash scripts/stop-all.sh` to free ports 3000, 8000, 8001.

**Gemini returns text instead of image** — `gemini-3-pro-image-preview` may not be available on your API key tier. The pipeline falls back to the deterministic composite in that case.

**OneFormer/SAM2 not running** — inference is optional. The API falls back to using your drawn polygon as the wall mask.
