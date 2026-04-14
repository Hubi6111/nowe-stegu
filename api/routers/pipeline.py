"""Pipeline — smart wall masking (CV engine) + deterministic texture projection."""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image, ImageDraw as PILImageDraw
from pydantic import BaseModel

from fastapi import Request
from fastapi.responses import StreamingResponse

from services.texture import (
    polygon_to_mask,
    exclusions_to_mask,
    compute_final_mask,
    render_mask_overlay,
    project_texture,
    masked_composite,
    mask_to_b64,
    decode_mask_b64,
    image_to_b64,
)
from services.admin_store import (
    check_rate_limit,
    increment_usage,
    get_watermark_config,
    load_watermark_image,
    load_config,
    save_generation,
    get_daily_limit,
)

logger = logging.getLogger(__name__)
router = APIRouter()

CV_ENGINE_URL = os.environ.get("CV_ENGINE_URL", "http://localhost:8001")


def _apply_watermark(img: Image.Image) -> Image.Image:
    """Overlay watermark in bottom-right corner if enabled."""
    wm_cfg = get_watermark_config()
    if not wm_cfg["enabled"] or not wm_cfg["has_file"]:
        return img
    wm = load_watermark_image()
    if wm is None:
        return img

    opacity = float(load_config().get("watermark_opacity", 0.3))
    W, H = img.size

    max_wm_w = int(W * 0.18)
    max_wm_h = int(H * 0.10)
    wm_scale = min(max_wm_w / max(wm.width, 1), max_wm_h / max(wm.height, 1), 1.0)
    if wm_scale < 1.0:
        wm = wm.resize((int(wm.width * wm_scale), int(wm.height * wm_scale)), Image.LANCZOS)

    if opacity < 1.0:
        alpha = wm.split()[3] if wm.mode == "RGBA" else None
        if alpha:
            import numpy as _np
            a = _np.array(alpha).astype(float) * opacity
            wm.putalpha(Image.fromarray(a.clip(0, 255).astype("uint8"), "L"))

    margin = int(min(W, H) * 0.02)
    pos = (W - wm.width - margin, H - wm.height - margin)

    result = img.copy().convert("RGBA")
    result.paste(wm, pos, wm)
    return result.convert("RGB")


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


TEXTURES_DIR = Path(
    os.environ.get(
        "TEXTURES_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "assets" / "textures" / "stegu"),
    )
)


class PointSchema(BaseModel):
    x: float
    y: float


class RectSchema(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectMaskRequest(BaseModel):
    image: str
    polygon: list[PointSchema]
    exclusions: list[RectSchema] = []
    canvas_width: float
    canvas_height: float


class RenderFinalRequest(BaseModel):
    image: str
    confirmed_mask: str | None = None
    polygon: list[PointSchema]
    product_id: str
    canvas_width: float
    canvas_height: float
    calibration: dict | None = None


class WallDetectRequest(BaseModel):
    image: str
    polygon: list[PointSchema]
    canvas_width: float
    canvas_height: float


class TextureProjectRequest(BaseModel):
    image: str
    mask: str
    product_id: str
    canvas_width: float
    canvas_height: float


class GenerateRequest(BaseModel):
    image: str
    polygon: list[PointSchema]
    exclusions: list[RectSchema] = []
    product_id: str
    canvas_width: float
    canvas_height: float


def decode_image(data: str) -> Image.Image:
    raw = data.split(",", 1)[-1] if "," in data else data
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def load_product(product_id: str) -> tuple[dict, Image.Image]:
    safe = product_id.replace("..", "").replace("/", "")
    folder = TEXTURES_DIR / safe
    meta_path = folder / "metadata.json"
    albedo_path = folder / "albedo.jpg"
    if not meta_path.is_file():
        raise HTTPException(404, detail=f"Product not found: {safe}")
    if not albedo_path.is_file():
        raise HTTPException(404, detail=f"Texture not found: {safe}")
    meta = json.loads(meta_path.read_text())
    texture = Image.open(albedo_path).convert("RGB")
    return meta, texture


def _build_polygon_mask(
    image: Image.Image,
    poly_mask: np.ndarray,
    excl_mask: np.ndarray,
) -> dict:
    """Build wall mask directly from user polygon."""
    final_mask = compute_final_mask(
        poly_mask,
        None,
        excl_mask if excl_mask.any() else None,
    )
    overlay = render_mask_overlay(image, final_mask)

    return {
        "wall_mask": poly_mask,
        "occluder_mask": None,
        "final_mask": final_mask,
        "detect_model": "polygon-only",
        "overlay": overlay,
    }


# ── Public endpoints ──────────────────────────────────────────────────────────

@router.get("/api/remaining-generations")
async def remaining_generations(request: Request):
    """Public endpoint: how many generations does this user have left today."""
    client_ip = _get_client_ip(request)
    limit = get_daily_limit()
    if limit <= 0:
        return {"remaining": -1, "limit": 0, "used": 0, "unlimited": True}
    _, used, _ = check_rate_limit(client_ip)
    return {"remaining": max(0, limit - used), "limit": limit, "used": used, "unlimited": False}


@router.post("/api/smart-mask")
async def smart_mask(req: DetectMaskRequest):
    """Smart wall masking via CV Engine.

    Takes the same polygon selection as detect-mask, but calls cv-engine
    to produce a wall-only mask that excludes ceiling, floor, and foreground objects.
    """
    t0 = time.time()
    try:
        canvas_w = max(int(req.canvas_width), 1)
        canvas_h = max(int(req.canvas_height), 1)
        image = decode_image(req.image)
        sx = image.width / canvas_w
        sy = image.height / canvas_h

        # Convert polygon points to image-space bounding box
        scaled_polygon = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]
        xs = [p["x"] for p in scaled_polygon]
        ys = [p["y"] for p in scaled_polygon]
        box = {
            "x1": max(0, min(xs)),
            "y1": max(0, min(ys)),
            "x2": min(image.width, max(xs)),
            "y2": min(image.height, max(ys)),
        }

        # Call cv-engine
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{CV_ENGINE_URL}/wall-mask",
                json={"image": req.image, "box": box},
            )

        if resp.status_code != 200:
            detail = resp.json().get("detail", "CV Engine error")
            raise HTTPException(resp.status_code, detail=detail)

        cv_result = resp.json()
        wall_mask_b64 = cv_result["wall_mask"]
        overlay_b64 = cv_result.get("overlay")
        fg_mask_b64 = cv_result.get("foreground_mask")

        return {
            "wall_mask": wall_mask_b64,
            "final_mask": wall_mask_b64,
            "foreground_mask": fg_mask_b64,
            "mask_overlay": overlay_b64,
            "wall_model": "deterministic",
            "image_width": image.width,
            "image_height": image.height,
            "calibration": cv_result.get("calibration"),
            "timings": {
                **cv_result.get("timings", {}),
                "proxy": round(time.time() - t0, 2),
            },
            "stats": cv_result.get("stats"),
        }
    except HTTPException:
        raise
    except httpx.ConnectError:
        raise HTTPException(
            503,
            detail="CV Engine niedostępny. Uruchom: conda activate cv-engine && cd cv-engine && python main.py"
        )
    except Exception as exc:
        logger.error("smart_mask failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"Smart mask failed: {exc}")


@router.post("/api/detect-mask")
async def detect_mask(req: DetectMaskRequest):
    """Build wall mask from user polygon (instant — no AI call)."""
    t0 = time.time()
    try:
        canvas_w = max(int(req.canvas_width), 1)
        canvas_h = max(int(req.canvas_height), 1)
        image = decode_image(req.image)
        sx = image.width / canvas_w
        sy = image.height / canvas_h

        scaled_polygon = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]
        poly_mask = polygon_to_mask(scaled_polygon, image.width, image.height)

        excl_dicts = [{"x": e.x, "y": e.y, "w": e.w, "h": e.h} for e in req.exclusions]
        excl_mask = exclusions_to_mask(
            excl_dicts, image.width, image.height, canvas_w, canvas_h
        )

        m = _build_polygon_mask(image, poly_mask, excl_mask)

        return {
            "wall_mask": mask_to_b64(m["wall_mask"]),
            "occluder_mask": None,
            "final_mask": mask_to_b64(m["final_mask"]),
            "exclusion_mask": mask_to_b64(excl_mask) if excl_mask.any() else None,
            "mask_overlay": image_to_b64(m["overlay"]),
            "wall_model": "polygon-only",
            "image_width": image.width,
            "image_height": image.height,
            "timings": {"detect_mask": round(time.time() - t0, 2)},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("detect_mask failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"Mask build failed: {exc}")


@router.post("/api/render-final")
async def render_final(req: RenderFinalRequest, request: Request):
    """Deterministic texture projection on confirmed mask."""
    client_ip = _get_client_ip(request)
    allowed, used, limit = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            429,
            detail=f"Limit {limit} generacji/dzień wyczerpany ({used}/{limit})."
        )

    t0 = time.time()
    results: dict = {"timings": {}}

    image = decode_image(req.image)
    canvas_w = max(int(req.canvas_width), 1)
    canvas_h = max(int(req.canvas_height), 1)
    sx = image.width / canvas_w
    sy = image.height / canvas_h
    scaled_polygon = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]

    if req.confirmed_mask:
        final_mask = decode_mask_b64(req.confirmed_mask, image.width, image.height)
    else:
        final_mask = polygon_to_mask(scaled_polygon, image.width, image.height)

    meta, texture = load_product(req.product_id)
    product_name = meta.get("name", req.product_id)

    # Deterministic texture projection
    t1 = time.time()
    composite = project_texture(
        image, final_mask, texture, meta=meta, polygon=scaled_polygon
    )
    composite_b64 = image_to_b64(composite)
    results["composite"] = composite_b64
    results["timings"]["texture_project"] = round(time.time() - t1, 2)

    # Use composite as final result (no AI)
    results["refined"] = composite_b64

    # ── Save images to disk for refine endpoint ───────────────────────────
    try:
        refine_dir = Path(os.environ.get("DATA_DIR", "./data")) / "refine-temp"
        refine_dir.mkdir(parents=True, exist_ok=True)
        image.save(refine_dir / "original.jpg", "JPEG", quality=92)
        composite.save(refine_dir / "composite.jpg", "JPEG", quality=92)
        logger.info("Saved refine images: %s", refine_dir)
    except Exception as save_exc:
        logger.warning("Failed to save refine images: %s", save_exc)

    # ── Watermark ─────────────────────────────────────────────────────────
    if results.get("refined"):
        try:
            refined_str = results["refined"]
            raw_b = refined_str.split(",", 1)[-1] if "," in refined_str else refined_str
            refined_img = Image.open(io.BytesIO(base64.b64decode(raw_b))).convert("RGB")
            watermarked = _apply_watermark(refined_img)
            if watermarked is not refined_img:
                results["refined"] = image_to_b64(watermarked)
        except Exception as wm_exc:
            logger.warning("Watermark failed (non-fatal): %s", wm_exc)

    results["timings"]["total"] = round(time.time() - t0, 2)
    results["image_width"] = image.width
    results["image_height"] = image.height

    # ── Save to generation gallery ────────────────────────────────────────
    try:
        thumbnail_b64 = None
        if results.get("refined"):
            refined_str = results["refined"]
            raw_r = refined_str.split(",", 1)[-1] if "," in refined_str else refined_str
            thumb_img = Image.open(io.BytesIO(base64.b64decode(raw_r)))
            thumb_img.thumbnail((300, 300), Image.LANCZOS)
            buf = io.BytesIO()
            thumb_img.save(buf, "JPEG", quality=60)
            thumbnail_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        save_generation(
            client_ip=client_ip,
            product_id=req.product_id,
            product_name=product_name,
            gemini_model="deterministic",
            timings=results.get("timings", {}),
            refined_b64=results.get("refined"),
            thumbnail_b64=thumbnail_b64,
        )
    except Exception as gen_exc:
        logger.warning("Failed to save generation record: %s", gen_exc)

    return results


# ── SSE Streaming endpoint ────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/render-stream")
async def render_stream(req: RenderFinalRequest, request: Request):
    """SSE streaming pipeline — decode → texture → done."""
    client_ip = _get_client_ip(request)
    allowed, used, limit = check_rate_limit(client_ip)
    if not allowed:
        error_payload = json.dumps({
            "stage": "done", "ok": False,
            "error": f"Limit {limit} generacji/dzień wyczerpany ({used}/{limit})."
        })
        return StreamingResponse(
            iter([f"data: {error_payload}\n\n"]),
            media_type="text/event-stream",
        )

    def _pipeline():
        t0 = time.time()
        timings: dict = {}

        # ── Stage 1: Decode inputs ────────────────────────────────────────
        yield _sse({"stage": "decode", "status": "running",
                     "message": "Przygotowywanie…"})
        try:
            image = decode_image(req.image)
            canvas_w = max(int(req.canvas_width), 1)
            canvas_h = max(int(req.canvas_height), 1)
            sx = image.width / canvas_w
            sy = image.height / canvas_h
            scaled_polygon = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]

            if req.confirmed_mask:
                final_mask = decode_mask_b64(req.confirmed_mask, image.width, image.height)
            else:
                final_mask = polygon_to_mask(scaled_polygon, image.width, image.height)
            td = round(time.time() - t0, 2)
            timings["decode"] = td
            yield _sse({"stage": "decode", "status": "done", "timing": td,
                         "detail": f"{image.width}×{image.height} px"})
        except Exception as exc:
            yield _sse({"stage": "decode", "status": "error", "error": str(exc)})
            yield _sse({"stage": "done", "ok": False, "error": str(exc)})
            return

        # ── Stage 2: Deterministic texture projection ─────────────────────
        t1 = time.time()
        yield _sse({"stage": "texture", "status": "running",
                     "message": "Tworzenie wizualizacji…"})
        try:
            meta, texture = load_product(req.product_id)

            # Build analysis from calibration data (precise px_per_cm)
            analysis = None
            if req.calibration and req.calibration.get("px_per_cm"):
                px_per_cm = req.calibration["px_per_cm"]
                wall_h_cm = req.calibration.get("wall_height_cm", 270)
                analysis = {
                    "wallHeightCm": wall_h_cm,
                    "px_per_cm": px_per_cm,
                    "confidence": req.calibration.get("confidence", "medium"),
                    "calibration_method": req.calibration.get("method", "unknown"),
                    "references": req.calibration.get("references", []),
                }
                logger.info(
                    "Using CV calibration: px/cm=%.3f, wall=%.0fcm (%s)",
                    px_per_cm, wall_h_cm, analysis["confidence"],
                )

            composite = project_texture(
                image, final_mask, texture, meta=meta,
                polygon=scaled_polygon, analysis=analysis,
            )
            composite_b64 = image_to_b64(composite)

            td = round(time.time() - t1, 2)
            timings["texture"] = td
            yield _sse({"stage": "texture", "status": "done", "timing": td,
                         "detail": meta.get("name", req.product_id)})
        except Exception as exc:
            yield _sse({"stage": "texture", "status": "error", "error": str(exc)})
            yield _sse({"stage": "done", "ok": False, "error": str(exc)})
            return

        # ── Final output ──────────────────────────────────────────────────
        final_output = composite
        refined_b64 = image_to_b64(final_output)

        # ── Save images to disk for refine endpoint ────────────────────
        try:
            refine_dir = Path(os.environ.get("DATA_DIR", "./data")) / "refine-temp"
            refine_dir.mkdir(parents=True, exist_ok=True)
            image.save(refine_dir / "original.jpg", "JPEG", quality=92)
            composite.save(refine_dir / "composite.jpg", "JPEG", quality=92)
            logger.info("Saved refine images to %s", refine_dir)
        except Exception as save_exc:
            logger.warning("Failed to save refine images: %s", save_exc)

        # ── Watermark ─────────────────────────────────────────────────────
        try:
            watermarked = _apply_watermark(final_output)
            if watermarked is not final_output:
                refined_b64 = image_to_b64(watermarked)
        except Exception:
            pass

        # ── Save & finalize ───────────────────────────────────────────────
        timings["total"] = round(time.time() - t0, 2)

        try:
            thumb_img = final_output.copy()
            thumb_img.thumbnail((300, 300), Image.LANCZOS)
            buf = io.BytesIO()
            thumb_img.save(buf, "JPEG", quality=60)
            thumbnail_b64 = "data:image/jpeg;base64," + base64.b64encode(
                buf.getvalue()
            ).decode()
            save_generation(
                client_ip=client_ip,
                product_id=req.product_id,
                product_name=meta.get("name", req.product_id),
                gemini_model="deterministic",
                timings=timings,
                refined_b64=refined_b64,
                thumbnail_b64=thumbnail_b64,
            )
        except Exception as gen_exc:
            logger.warning("Failed to save generation record: %s", gen_exc)

        # ── Final result ──────────────────────────────────────────────────
        yield _sse({"stage": "done", "ok": True,
            "result": {
                "composite": composite_b64,
                "refined": refined_b64,
                "gemini_model": "deterministic",
                "timings": timings,
            }})

    return StreamingResponse(
        _pipeline(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/wall-detect")
async def wall_detect(req: WallDetectRequest):
    canvas_w = max(int(req.canvas_width), 1)
    canvas_h = max(int(req.canvas_height), 1)
    image = decode_image(req.image)
    sx = image.width / canvas_w
    sy = image.height / canvas_h
    scaled = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]
    poly_mask = polygon_to_mask(scaled, image.width, image.height)
    excl_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    m = _build_polygon_mask(image, poly_mask, excl_mask)
    return {
        "wall_mask": mask_to_b64(m["wall_mask"]),
        "occluder_mask": None,
        "image_width": image.width,
        "image_height": image.height,
        "model": "polygon-only",
    }


@router.post("/api/texture-project")
async def texture_project(req: TextureProjectRequest):
    image = decode_image(req.image)
    mask = decode_mask_b64(req.mask, image.width, image.height)
    meta, texture = load_product(req.product_id)
    composite = project_texture(image, mask, texture, meta=meta)
    return {"composite": image_to_b64(composite)}


@router.post("/api/generate-visualization")
async def generate_visualization(req: GenerateRequest):
    detect_req = DetectMaskRequest(
        image=req.image,
        polygon=req.polygon,
        exclusions=req.exclusions,
        canvas_width=req.canvas_width,
        canvas_height=req.canvas_height,
    )
    mask_result = await detect_mask(detect_req)

    render_req = RenderFinalRequest(
        image=req.image,
        confirmed_mask=mask_result["final_mask"],
        polygon=req.polygon,
        product_id=req.product_id,
        canvas_width=req.canvas_width,
        canvas_height=req.canvas_height,
    )
    # We need a mock request for render_final
    from starlette.testclient import TestClient
    from unittest.mock import MagicMock
    mock_request = MagicMock()
    mock_request.client.host = "internal"
    mock_request.headers = {}
    render_result = await render_final(render_req, mock_request)

    return {
        **mask_result,
        **render_result,
        "wall_model": mask_result.get("wall_model", "unknown"),
    }
