"""Stegu Visualizer — API Service (port 8000)

Central orchestrator: product catalog, pipeline coordination,
and deterministic texture projection.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("API service starting")
    logger.info("  TEXTURES_DIR  = %s", os.environ.get("TEXTURES_DIR", "(default)"))
    yield
    logger.info("API service shutting down")


app = FastAPI(
    title="Stegu Visualizer API",
    version="0.3.0",
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
    return {
        "status": "ok",
        "textures_dir": os.environ.get("TEXTURES_DIR", "(default)"),
    }


app.include_router(products.router)
app.include_router(pipeline.router)
app.include_router(admin.router)
