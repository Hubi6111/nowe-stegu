"""Stegu Visualizer — API Service (port 8000)

Central orchestrator: product catalog, pipeline coordination,
deterministic texture projection, and Gemini AI refinement.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _key in ("TEXTURES_DIR", "DATA_DIR", "MODEL_CACHE_DIR"):
    _val = os.environ.get(_key)
    if _val and not os.path.isabs(_val):
        os.environ[_key] = str(PROJECT_ROOT / _val)

from routers import products, pipeline, admin  # noqa: E402
from services.gemini import image_model_name, text_model_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("API service starting")
    logger.info("  TEXTURES_DIR  = %s", os.environ.get("TEXTURES_DIR", "(default)"))
    logger.info("  INFERENCE_URL = %s", os.environ.get("INFERENCE_URL", "(default)"))
    logger.info("  GEMINI_API_KEY = %s", "set" if os.environ.get("GEMINI_API_KEY") else "NOT SET")
    yield
    logger.info("API service shutting down")


app = FastAPI(
    title="Stegu Visualizer API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    inf_url = (
        os.environ.get("INFERENCE_BASE_URL")
        or os.environ.get("INFERENCE_URL")
        or "http://localhost:8001"
    ).rstrip("/")
    inference_reachable = False
    inference_ready = False
    inference_status = {}
    try:
        import httpx

        r = httpx.get(f"{inf_url}/health", timeout=3.0)
        inference_reachable = r.status_code == 200
        if inference_reachable:
            inference_status = r.json()
            inference_ready = (
                inference_status.get("warmup_complete", False)
                and inference_status.get("oneformer_loaded", False)
            )
    except Exception:
        pass

    return {
        "status": "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "gemini_text_model": text_model_name(),
        "gemini_image_model": image_model_name(),
        "inference_url": inf_url,
        "inference_reachable": inference_reachable,
        "inference_ready": inference_ready,
        "inference_status": inference_status,
    }


app.include_router(products.router)
app.include_router(pipeline.router)
app.include_router(admin.router)
