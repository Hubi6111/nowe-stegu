"""Stegu Visualizer — Inference Service (port 8001)

Hosts OneFormer wall detection and SAM 2 mask refinement.
Models are preloaded at startup in a background thread so the first
user request never triggers a multi-minute download/load.
"""

import base64
import io
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("inference")

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_cache = os.path.join(_project_root, "data", "model-cache")
MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", _default_cache)
if not os.path.isabs(MODEL_CACHE):
    MODEL_CACHE = os.path.join(_project_root, MODEL_CACHE)

# ── Model state ─────────────────────────────────────────────

_oneformer = None
_sam2 = None

_oneformer_loading = False
_oneformer_error: str | None = None
_sam2_loading = False
_sam2_error: str | None = None
_warmup_complete = False


def _load_oneformer():
    global _oneformer, _oneformer_loading, _oneformer_error
    _oneformer_loading = True
    _oneformer_error = None
    try:
        from models.oneformer_detector import OneFormerDetector
        _oneformer = OneFormerDetector(cache_dir=MODEL_CACHE)
        logger.info("OneFormer loaded successfully")
    except Exception as exc:
        _oneformer_error = f"{type(exc).__name__}: {exc}"
        logger.error("OneFormer failed to load: %s\n%s", exc, traceback.format_exc())
    finally:
        _oneformer_loading = False


def _load_sam2():
    global _sam2, _sam2_loading, _sam2_error
    _sam2_loading = True
    _sam2_error = None
    try:
        from models.sam2_refiner import SAM2Refiner
        _sam2 = SAM2Refiner(cache_dir=MODEL_CACHE)
        logger.info("SAM2 loaded — available: %s", _sam2.available if _sam2 else False)
    except Exception as exc:
        _sam2_error = f"{type(exc).__name__}: {exc}"
        logger.error("SAM2 failed to load: %s\n%s", exc, traceback.format_exc())
    finally:
        _sam2_loading = False


def _warmup():
    """Load all models sequentially in a background thread."""
    global _warmup_complete
    logger.info("Warmup started — loading models in background …")
    _load_oneformer()
    _load_sam2()
    _warmup_complete = True
    logger.info(
        "Warmup complete — oneformer=%s  sam2=%s (available=%s)",
        _oneformer is not None,
        _sam2 is not None,
        _sam2.available if _sam2 else False,
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Inference service starting")
    logger.info("Model cache: %s", os.path.abspath(MODEL_CACHE))
    os.makedirs(MODEL_CACHE, exist_ok=True)
    t = threading.Thread(target=_warmup, daemon=True, name="model-warmup")
    t.start()
    yield
    logger.info("Inference service shutting down")


app = FastAPI(title="Stegu Inference Service", lifespan=lifespan)


# ── Health & Readiness ──────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "warmup_complete": _warmup_complete,
        "oneformer_loaded": _oneformer is not None,
        "oneformer_loading": _oneformer_loading,
        "oneformer_error": _oneformer_error,
        "sam2_loaded": _sam2 is not None,
        "sam2_loading": _sam2_loading,
        "sam2_available": _sam2.available if _sam2 else False,
        "sam2_error": _sam2_error,
        "model_cache": os.path.abspath(MODEL_CACHE),
    }


@app.get("/ready")
def ready():
    """Returns 200 only when models are loaded and ready for requests."""
    if not _warmup_complete:
        raise HTTPException(
            status_code=503,
            detail="Models still loading — warmup in progress",
        )
    if _oneformer is None:
        raise HTTPException(
            status_code=503,
            detail=f"OneFormer not available: {_oneformer_error or 'unknown'}",
        )
    return {"ready": True}


@app.post("/warmup")
def warmup_endpoint():
    """Trigger model loading if not already done (idempotent)."""
    if _warmup_complete:
        return {"status": "already_complete"}
    if _oneformer_loading or _sam2_loading:
        return {"status": "in_progress"}
    t = threading.Thread(target=_warmup, daemon=True, name="model-warmup-manual")
    t.start()
    return {"status": "started"}


# ── Schemas ─────────────────────────────────────────────────


class PointSchema(BaseModel):
    x: float
    y: float


class WallDetectRequest(BaseModel):
    image: str
    polygon: list[PointSchema]
    canvas_width: int
    canvas_height: int


class WallRefineRequest(BaseModel):
    image: str
    coarse_mask: str
    canvas_width: int
    canvas_height: int
    polygon: list[PointSchema] | None = None


# ── Helpers ─────────────────────────────────────────────────


def decode_image(data: str) -> Image.Image:
    raw = data.split(",", 1)[-1] if "," in data else data
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def scale_polygon(
    polygon: list[PointSchema],
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
) -> list[dict]:
    sx = img_w / canvas_w
    sy = img_h / canvas_h
    return [{"x": p.x * sx, "y": p.y * sy} for p in polygon]


def _require_oneformer():
    """Return the detector or raise a clear 503."""
    if _oneformer_loading:
        raise HTTPException(503, detail="OneFormer is still loading — try again shortly")
    if _oneformer is None:
        raise HTTPException(503, detail=f"OneFormer not available: {_oneformer_error or 'not loaded'}")
    return _oneformer


def _require_sam2():
    """Return the refiner or raise a clear 503."""
    if _sam2_loading:
        raise HTTPException(503, detail="SAM2 is still loading — try again shortly")
    if _sam2 is None:
        raise HTTPException(503, detail=f"SAM2 not available: {_sam2_error or 'not loaded'}")
    return _sam2


# ── Endpoints ───────────────────────────────────────────────


@app.post("/wall-detect")
def wall_detect(req: WallDetectRequest):
    detector = _require_oneformer()

    try:
        image = decode_image(req.image)
    except Exception as exc:
        raise HTTPException(422, detail=f"Could not decode image: {exc}")

    try:
        polygon = scale_polygon(
            req.polygon, req.canvas_width, req.canvas_height,
            image.width, image.height,
        )
        result = detector.detect(image, roi_polygon=polygon)
    except Exception as exc:
        logger.error("OneFormer detect failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"Wall detection failed: {exc}")

    def _opt_mask_b64(key: str) -> str | None:
        m = result.get(key)
        if m is None or not m.any():
            return None
        return mask_to_base64(m)

    return {
        "wall_mask": mask_to_base64(result["wall_mask"]),
        "wall_mask_full": mask_to_base64(result["wall_mask_full"]),
        "occluder_mask": mask_to_base64(result["occluder_mask"]),
        "occluder_mask_full": mask_to_base64(result["occluder_mask_full"]),
        "ceiling_mask": _opt_mask_b64("ceiling_mask"),
        "floor_mask": _opt_mask_b64("floor_mask"),
        "image_width": image.width,
        "image_height": image.height,
        "model": "oneformer-ade20k-swin-large",
    }


@app.post("/wall-refine")
def wall_refine(req: WallRefineRequest):
    refiner = _require_sam2()

    try:
        image = decode_image(req.image)
        coarse_mask_img = decode_image(req.coarse_mask).convert("L")
        coarse_mask = (np.array(coarse_mask_img) > 127).astype(np.uint8)
    except Exception as exc:
        raise HTTPException(422, detail=f"Could not decode image/mask: {exc}")

    try:
        if coarse_mask.shape != (image.height, image.width):
            coarse_mask_pil = Image.fromarray(
                (coarse_mask * 255).astype(np.uint8)
            )
            coarse_mask = (
                np.array(
                    coarse_mask_pil.resize(
                        (image.width, image.height), Image.NEAREST
                    )
                )
                > 127
            ).astype(np.uint8)

        polygon_dicts: list[dict] | None = None
        if req.polygon:
            polygon_dicts = scale_polygon(
                req.polygon, req.canvas_width, req.canvas_height,
                image.width, image.height,
            )

        refined = refiner.refine_with_polygon(image, coarse_mask, polygon=polygon_dicts)
    except Exception as exc:
        logger.error("SAM2 refine failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"Mask refinement failed: {exc}")

    return {
        "refined_mask": mask_to_base64(refined),
        "model": "sam2-hiera-large" if refiner.available else "passthrough",
    }
