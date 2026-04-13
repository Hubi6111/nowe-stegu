"""CV Engine — FastAPI service for computer vision pipeline (port 8001).

Loads SAM2, GroundingDINO, Depth-Anything-V2, and SegFormer on startup.
Exposes endpoints for wall segmentation, foreground detection, depth estimation,
and a unified /analyze endpoint running the full pipeline.
"""

import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Add vendor dirs to path
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(VENDOR_DIR / "GroundingDINO"))
sys.path.insert(0, str(VENDOR_DIR / "Depth-Anything-V2"))

from engine import wall_prior, mask_refine, foreground, depth, compositor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("cv-engine")


# ── Models warmup status ──────────────────────────────────────────────────────

_warmup_status = {
    "wall_prior": False,
    "sam2": False,
    "grounding_dino": False,
    "depth": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CV Engine starting — loading models...")
    t0 = time.time()

    # Load models lazily on first request, or eagerly via WARMUP=1
    if os.environ.get("WARMUP", "0") == "1":
        try:
            wall_prior.load_model()
            _warmup_status["wall_prior"] = True
        except Exception as e:
            logger.error("Failed to load wall prior: %s", e)

        try:
            mask_refine.load_model()
            _warmup_status["sam2"] = True
        except Exception as e:
            logger.error("Failed to load SAM2: %s", e)

        try:
            foreground.load_model()
            _warmup_status["grounding_dino"] = True
        except Exception as e:
            logger.error("Failed to load GroundingDINO: %s", e)

        try:
            depth.load_model()
            _warmup_status["depth"] = True
        except Exception as e:
            logger.error("Failed to load depth model: %s", e)

        logger.info("All models loaded in %.1fs", time.time() - t0)
    else:
        logger.info("Lazy loading mode — models load on first request")

    yield
    logger.info("CV Engine shutting down")


app = FastAPI(title="CV Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PointSchema(BaseModel):
    x: float
    y: float


class AnalyzeRequest(BaseModel):
    image: str  # base64 encoded
    polygon: list[PointSchema] | None = None
    foreground_prompt: str | None = None


class WallSegmentRequest(BaseModel):
    image: str
    polygon: list[PointSchema] | None = None


class ForegroundRequest(BaseModel):
    image: str
    wall_mask: str | None = None  # base64 encoded mask
    prompt: str | None = None


class DepthRequest(BaseModel):
    image: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(data: str) -> Image.Image:
    raw = data.split(",", 1)[-1] if "," in data else data
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def decode_mask(data: str, w: int, h: int) -> np.ndarray:
    raw = data.split(",", 1)[-1] if "," in data else data
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    return np.array(img)


def mask_to_b64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def polygon_to_mask(polygon: list[PointSchema], w: int, h: int) -> np.ndarray:
    """Convert polygon points to a binary mask."""
    import cv2
    pts = np.array([[int(p.x), int(p.y)] for p in polygon], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": _warmup_status,
    }


@app.post("/wall-segment")
async def wall_segment(req: WallSegmentRequest):
    """Stage 1+2: Wall prior + SAM2 refinement."""
    t0 = time.time()
    image = decode_image(req.image)

    # Stage 1: Wall prior
    wall_prob = wall_prior.predict_wall_mask(image)
    _warmup_status["wall_prior"] = True

    # Combine with user polygon if provided
    if req.polygon and len(req.polygon) >= 3:
        poly_mask = polygon_to_mask(req.polygon, image.width, image.height)
        # Intersect: wall prior AND polygon
        wall_binary = (wall_prob > 0.2).astype(np.uint8) * 255
        combined = np.minimum(wall_binary, poly_mask)
    else:
        combined = (wall_prob > 0.3).astype(np.uint8) * 255

    # Stage 2: SAM2 refinement
    polygon_dicts = [{"x": p.x, "y": p.y} for p in req.polygon] if req.polygon else None
    refined = mask_refine.refine_mask(image, combined, polygon_dicts)
    _warmup_status["sam2"] = True

    return {
        "wall_prior": mask_to_b64((wall_prob * 255).astype(np.uint8)),
        "wall_mask": mask_to_b64(refined),
        "timing": round(time.time() - t0, 2),
    }


@app.post("/foreground-detect")
async def foreground_detect(req: ForegroundRequest):
    """Stage 3: Foreground occlusion masking."""
    t0 = time.time()
    image = decode_image(req.image)

    wall_mask = None
    if req.wall_mask:
        wall_mask = decode_mask(req.wall_mask, image.width, image.height)

    prompt = req.prompt or foreground.DEFAULT_PROMPT
    fg_mask = foreground.detect_foreground_mask(
        image, wall_mask=wall_mask, text_prompt=prompt
    )
    _warmup_status["grounding_dino"] = True

    # Also return detected objects list
    boxes = foreground.detect_foreground_boxes(image, text_prompt=prompt)

    return {
        "foreground_mask": mask_to_b64(fg_mask),
        "detections": boxes,
        "timing": round(time.time() - t0, 2),
    }


@app.post("/depth-estimate")
async def depth_estimate(req: DepthRequest):
    """Stage 4: Depth estimation."""
    t0 = time.time()
    image = decode_image(req.image)

    depth_map = depth.estimate_depth(image)
    _warmup_status["depth"] = True

    # Create colormap visualization
    depth_vis = depth.depth_to_colormap(depth_map)

    return {
        "depth_map": mask_to_b64((depth_map * 255).astype(np.uint8)),
        "depth_colormap": image_to_b64(depth_vis),
        "timing": round(time.time() - t0, 2),
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Full pipeline: wall + foreground + depth in one call."""
    t0 = time.time()
    image = decode_image(req.image)
    timings = {}

    # Stage 1+2: Wall segmentation
    t1 = time.time()
    wall_prob = wall_prior.predict_wall_mask(image)
    _warmup_status["wall_prior"] = True

    if req.polygon and len(req.polygon) >= 3:
        poly_mask = polygon_to_mask(req.polygon, image.width, image.height)
        wall_binary = (wall_prob > 0.2).astype(np.uint8) * 255
        combined = np.minimum(wall_binary, poly_mask)
    else:
        combined = (wall_prob > 0.3).astype(np.uint8) * 255

    polygon_dicts = [{"x": p.x, "y": p.y} for p in req.polygon] if req.polygon else None
    wall_mask = mask_refine.refine_mask(image, combined, polygon_dicts)
    _warmup_status["sam2"] = True
    timings["wall"] = round(time.time() - t1, 2)

    # Stage 3: Foreground detection
    t2 = time.time()
    prompt = req.foreground_prompt or foreground.DEFAULT_PROMPT
    fg_mask = foreground.detect_foreground_mask(
        image, wall_mask=wall_mask, text_prompt=prompt
    )
    _warmup_status["grounding_dino"] = True
    timings["foreground"] = round(time.time() - t2, 2)

    # Stage 4: Depth
    t3 = time.time()
    depth_map = depth.estimate_depth(image)
    _warmup_status["depth"] = True
    timings["depth"] = round(time.time() - t3, 2)

    timings["total"] = round(time.time() - t0, 2)

    depth_vis = depth.depth_to_colormap(depth_map)

    return {
        "wall_prior": mask_to_b64((wall_prob * 255).astype(np.uint8)),
        "wall_mask": mask_to_b64(wall_mask),
        "foreground_mask": mask_to_b64(fg_mask),
        "depth_map": mask_to_b64((depth_map * 255).astype(np.uint8)),
        "depth_colormap": image_to_b64(depth_vis),
        "timings": timings,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
