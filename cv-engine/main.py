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

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Add vendor dirs to path
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(VENDOR_DIR / "GroundingDINO"))
sys.path.insert(0, str(VENDOR_DIR / "Depth-Anything-V2"))

from engine import wall_prior, mask_refine, foreground, depth, compositor, scale_calibrate

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


class BoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class WallMaskRequest(BaseModel):
    """Smart wall masking: image + selection box → clean wall-only mask."""
    image: str  # base64 encoded
    box: BoxSchema  # user selection rectangle in image coordinates


class AnalyzeRequest(BaseModel):
    image: str
    polygon: list[PointSchema] | None = None
    foreground_prompt: str | None = None


class WallSegmentRequest(BaseModel):
    image: str
    polygon: list[PointSchema] | None = None


class ForegroundRequest(BaseModel):
    image: str
    wall_mask: str | None = None
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
    pts = np.array([[int(p.x), int(p.y)] for p in polygon], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def render_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create a semi-transparent overlay showing the mask on the image."""
    img_np = np.array(image).copy()
    overlay = img_np.copy()
    # Red tint for wall area
    overlay[mask > 127] = (
        overlay[mask > 127] * 0.5 + np.array([180, 40, 40]) * 0.5
    ).astype(np.uint8)
    # Green outline for boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return Image.fromarray(overlay)


# ── MAIN ENDPOINT: Smart Wall Mask ────────────────────────────────────────────

@app.post("/wall-mask")
async def wall_mask_endpoint(req: WallMaskRequest):
    """Smart wall masking: selection box → wall-only mask.

    Pipeline:
    1. SegFormer-B5 → wall prob + floor/ceiling/door exclusion
    2. Intersect wall prior with user box, subtract exclusions
    3. SAM2 iterative refinement with grid points + exclusion negatives
    4. GroundingDINO + SAM2 → detect & subtract foreground objects
    5. Morphological cleanup + largest-component selection
    """
    t0 = time.time()
    timings = {}
    image = decode_image(req.image)
    w, h = image.size

    # Clamp box to image bounds
    x1 = max(0, int(req.box.x1))
    y1 = max(0, int(req.box.y1))
    x2 = min(w, int(req.box.x2))
    y2 = min(h, int(req.box.y2))

    if x2 - x1 < 10 or y2 - y1 < 10:
        raise HTTPException(400, detail="Selection box too small")

    logger.info("Wall mask request: box=[%d,%d,%d,%d] image=%dx%d", x1, y1, x2, y2, w, h)

    # ── Stage 1: Full semantic segmentation (SegFormer-B5) ────────────────
    t1 = time.time()
    seg_result = wall_prior.predict_full(image)
    wall_prob = seg_result["wall_prob"]
    exclude_mask = seg_result["exclude_mask"]
    floor_prob = seg_result["floor_prob"]
    ceiling_prob = seg_result["ceiling_prob"]
    _warmup_status["wall_prior"] = True
    timings["wall_prior"] = round(time.time() - t1, 2)
    logger.info("Stage 1 (wall prior B5): %.2fs", timings["wall_prior"])

    # Box mask
    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255
    box_pixels = (y2 - y1) * (x2 - x1)

    # ── Build coarse wall mask ────────────────────────────────────────────
    wall_in_box = wall_prob.copy()
    wall_in_box[box_mask == 0] = 0

    # Remove exclusion zones from wall probability
    wall_in_box[exclude_mask > 127] = 0

    # Suppress floor/ceiling with soft threshold
    wall_in_box[floor_prob > 0.10] *= 0.1
    wall_in_box[ceiling_prob > 0.10] *= 0.1

    # Adaptive threshold based on peak probability in box
    max_prob = wall_in_box.max()
    if max_prob > 0.5:
        threshold = max(0.15, max_prob * 0.3)
    else:
        threshold = 0.10

    coarse_mask = (wall_in_box > threshold).astype(np.uint8) * 255

    # If wall prior found very little, fall back to box minus exclusions
    wall_pixel_count = coarse_mask.sum() / 255
    if wall_pixel_count < box_pixels * 0.08:
        logger.warning("Wall prior found <8%% wall in box, using box - exclusions")
        coarse_mask = box_mask.copy()
        coarse_mask[exclude_mask > 127] = 0

    logger.info(
        "Coarse mask: threshold=%.3f, pixels=%d (%.1f%% of box)",
        threshold, coarse_mask.sum() // 255,
        (coarse_mask.sum() / 255) / max(box_pixels, 1) * 100,
    )

    # ── Stage 2: SAM2 iterative refinement ────────────────────────────────
    t2 = time.time()
    refined = mask_refine.refine_mask(
        image,
        coarse_mask,
        box=(x1, y1, x2, y2),
        exclude_mask=exclude_mask,
        n_iterations=2,
    )
    _warmup_status["sam2"] = True
    timings["sam2_refine"] = round(time.time() - t2, 2)
    logger.info("Stage 2 (SAM2 refine 2-iter): %.2fs", timings["sam2_refine"])

    # Clip to box and re-apply exclusion zones
    refined = np.minimum(refined, box_mask)
    refined[exclude_mask > 127] = 0

    # ── Stage 3: Foreground occlusion ─────────────────────────────────────
    t3 = time.time()
    try:
        fg_mask = foreground.detect_foreground_mask(
            image,
            wall_mask=refined,
            text_prompt=foreground.DEFAULT_PROMPT,
        )
        _warmup_status["grounding_dino"] = True
        timings["foreground"] = round(time.time() - t3, 2)
        logger.info("Stage 3 (foreground): %.2fs, %d fg pixels",
                     timings["foreground"], fg_mask.sum() // 255)

        if fg_mask.sum() > 0:
            kernel = np.ones((7, 7), np.uint8)
            fg_dilated = cv2.dilate(fg_mask, kernel, iterations=2)
            refined = np.clip(
                refined.astype(np.int16) - fg_dilated.astype(np.int16),
                0, 255,
            ).astype(np.uint8)
    except Exception as e:
        logger.warning("Foreground detection failed (non-fatal): %s", e)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        timings["foreground"] = -1

    # ── Stage 4: Morphological cleanup + largest component ────────────────
    t4 = time.time()

    # Connected components — remove small noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined, connectivity=8
    )
    min_area = max(200, int(box_pixels * 0.01))
    clean_mask = np.zeros_like(refined)

    if num_labels > 1:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        largest_area = max(areas) if areas else 0
        for i in range(1, num_labels):
            # Keep component if it's large enough (at least 20% of largest)
            if stats[i, cv2.CC_STAT_AREA] >= max(min_area, largest_area * 0.2):
                clean_mask[labels == i] = 255

    # Morphological close to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    # Fill internal holes (holes inside the wall boundary)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_filled = np.zeros_like(clean_mask)
    cv2.drawContours(hole_filled, contours, -1, 255, -1)
    clean_mask = hole_filled

    # Final exclusion pass
    clean_mask[exclude_mask > 127] = 0

    # Smooth edges
    clean_mask = cv2.GaussianBlur(clean_mask, (3, 3), 0)
    clean_mask = (clean_mask > 127).astype(np.uint8) * 255

    timings["cleanup"] = round(time.time() - t4, 2)
    timings["total"] = round(time.time() - t0, 2)

    logger.info("Wall mask complete: total=%.2fs, mask pixels=%d (%.1f%% of box)",
                timings["total"], clean_mask.sum() // 255,
                (clean_mask.sum() / 255) / max(box_pixels, 1) * 100)
    # ── Stage 5: Scale calibration ──────────────────────────────────────────
    t5 = time.time()
    calibration = scale_calibrate.calibrate_scale(
        image,
        segmentation_argmax=seg_result["argmax"],
        wall_mask=clean_mask,
        box=(x1, y1, x2, y2),
    )
    timings["scale_calibrate"] = round(time.time() - t5, 2)
    timings["total"] = round(time.time() - t0, 2)

    logger.info(
        "Scale calibration: px/cm=%.3f, wall=%.0fcm, confidence=%s, method=%s",
        calibration["px_per_cm"], calibration["wall_height_cm"],
        calibration["confidence"], calibration["method"],
    )

    # ── Build overlay visualization ───────────────────────────────────────
    overlay = render_mask_overlay(image, clean_mask)

    return {
        "wall_mask": mask_to_b64(clean_mask),
        "wall_prior_raw": mask_to_b64((wall_prob * 255).astype(np.uint8)),
        "foreground_mask": mask_to_b64(fg_mask),
        "exclude_mask": mask_to_b64(exclude_mask),
        "overlay": image_to_b64(overlay),
        "calibration": calibration,
        "timings": timings,
        "stats": {
            "wall_pixels": int(clean_mask.sum() // 255),
            "total_pixels": w * h,
            "coverage_pct": round(float(clean_mask.sum() / 255) / (w * h) * 100, 1),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        },
    }



# ── Other endpoints ───────────────────────────────────────────────────────────

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

    wall_prob = wall_prior.predict_wall_mask(image)
    _warmup_status["wall_prior"] = True

    if req.polygon and len(req.polygon) >= 3:
        poly_mask = polygon_to_mask(req.polygon, image.width, image.height)
        wall_binary = (wall_prob > 0.2).astype(np.uint8) * 255
        combined = np.minimum(wall_binary, poly_mask)
    else:
        combined = (wall_prob > 0.3).astype(np.uint8) * 255

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
    t0 = time.time()
    image = decode_image(req.image)

    wall_mask_arr = None
    if req.wall_mask:
        wall_mask_arr = decode_mask(req.wall_mask, image.width, image.height)

    prompt = req.prompt or foreground.DEFAULT_PROMPT
    fg_mask = foreground.detect_foreground_mask(
        image, wall_mask=wall_mask_arr, text_prompt=prompt
    )
    _warmup_status["grounding_dino"] = True

    boxes = foreground.detect_foreground_boxes(image, text_prompt=prompt)

    return {
        "foreground_mask": mask_to_b64(fg_mask),
        "detections": boxes,
        "timing": round(time.time() - t0, 2),
    }


@app.post("/depth-estimate")
async def depth_estimate(req: DepthRequest):
    t0 = time.time()
    image = decode_image(req.image)

    depth_map = depth.estimate_depth(image)
    _warmup_status["depth"] = True
    depth_vis = depth.depth_to_colormap(depth_map)

    return {
        "depth_map": mask_to_b64((depth_map * 255).astype(np.uint8)),
        "depth_colormap": image_to_b64(depth_vis),
        "timing": round(time.time() - t0, 2),
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.time()
    image = decode_image(req.image)
    timings = {}

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

    t2 = time.time()
    prompt = req.foreground_prompt or foreground.DEFAULT_PROMPT
    fg_mask = foreground.detect_foreground_mask(
        image, wall_mask=wall_mask, text_prompt=prompt
    )
    _warmup_status["grounding_dino"] = True
    timings["foreground"] = round(time.time() - t2, 2)

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
