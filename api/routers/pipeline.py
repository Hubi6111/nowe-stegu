"""Pipeline — wall mask detection (user polygon + Gemini) and texture render."""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image, ImageDraw as PILImageDraw
from pydantic import BaseModel

from fastapi import Request
from fastapi.responses import StreamingResponse

from services.gemini import (
    analyze_wall_scene,
    generate_photorealistic_render,
    verify_and_correct_render,
    image_model_name,
    refine_image,
    image_to_b64,
)
from services.texture import (
    polygon_to_mask,
    exclusions_to_mask,
    compute_final_mask,
    refine_mask_with_boundaries,
    render_mask_overlay,
    project_texture,
    masked_composite,
    mask_to_b64,
    decode_mask_b64,
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


class RenderRefineRequest(BaseModel):
    composite: str
    mask: str | None = None
    original_image: str | None = None
    product_name: str


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
    """Build wall mask directly from user polygon (no AI — Gemini handles
    scene analysis during the final render step instead)."""
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


@router.get("/api/remaining-generations")
async def remaining_generations(request: Request):
    """Public endpoint: how many generations does this user have left today."""
    client_ip = _get_client_ip(request)
    limit = get_daily_limit()
    if limit <= 0:
        return {"remaining": -1, "limit": 0, "used": 0, "unlimited": True}
    _, used, _ = check_rate_limit(client_ip)
    return {"remaining": max(0, limit - used), "limit": limit, "used": used, "unlimited": False}


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
    """Texture on confirmed mask + Gemini photorealistic pass."""
    client_ip = _get_client_ip(request)
    allowed, used, limit = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            429,
            detail=f"Dzienny limit generacji ({limit}) został osiągnięty. Spróbuj jutro.",
        )

    t0 = time.time()
    results: dict = {"timings": {}}

    try:
        canvas_w = max(int(req.canvas_width), 1)
        canvas_h = max(int(req.canvas_height), 1)
        image = decode_image(req.image)
        sx = image.width / canvas_w
        sy = image.height / canvas_h
        scaled_polygon = [{"x": p.x * sx, "y": p.y * sy} for p in req.polygon]

        if req.confirmed_mask:
            final_mask = decode_mask_b64(req.confirmed_mask, image.width, image.height)
        else:
            final_mask = polygon_to_mask(scaled_polygon, image.width, image.height)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("render_final decode failed: %s", exc, exc_info=True)
        raise HTTPException(422, detail=f"Could not decode inputs: {exc}")

    # ── Deterministic texture projection (heuristic scale — for analysis) ─
    t1 = time.time()
    try:
        meta, texture = load_product(req.product_id)
        composite = project_texture(
            image, final_mask, texture, meta=meta, polygon=scaled_polygon
        )
        results["composite"] = image_to_b64(composite)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Texture projection failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"Texture projection failed: {exc}")
    results["timings"]["texture_project"] = round(time.time() - t1, 2)

    mask_overlay = render_mask_overlay(
        image, final_mask, alpha=0.55, max_long_side=2048
    )

    # ── Stage 1: Gemini scene analysis (measurement + occlusion + lighting) ─
    t2 = time.time()
    if os.environ.get("GEMINI_API_KEY"):
        try:
            product_name = meta.get("name", req.product_id)
            material_type = meta.get("materialType", meta.get("layoutType", "decorative stone/brick"))

            analysis: dict = {}
            try:
                analysis = analyze_wall_scene(image, composite, mask_overlay)
                results["timings"]["scene_analysis"] = round(time.time() - t2, 2)
                results["analysis"] = {
                    "wallHeightCm": analysis.get("wallHeightCm"),
                    "wallWidthCm": analysis.get("wallWidthCm"),
                    "referenceObjects": [
                        r.get("name") for r in analysis.get("referenceObjects", [])
                    ],
                    "occluders": [
                        o.get("label") for o in analysis.get("occluders", [])
                    ],
                    "confidence": analysis.get("confidence"),
                }
            except Exception as exc_a:
                logger.warning("Gemini analysis failed (non-fatal): %s", exc_a)

            # ── Refine mask with wall boundaries from analysis ────────────
            wall_boundaries = analysis.get("wallBoundaries")
            if wall_boundaries:
                refined = refine_mask_with_boundaries(final_mask, wall_boundaries)
                if refined.any() and not np.array_equal(refined, final_mask):
                    logger.info(
                        "Mask refined by wall boundaries — trimmed %d px",
                        int(final_mask.sum() - refined.sum()),
                    )
                    final_mask = refined

            # ── Re-tile with AI-calibrated dimensions ─────────────────────
            t_retile = time.time()
            try:
                composite = project_texture(
                    image, final_mask, texture, meta=meta,
                    polygon=scaled_polygon, analysis=analysis,
                )
                results["composite"] = image_to_b64(composite)
                mask_overlay = render_mask_overlay(
                    image, final_mask, alpha=0.55, max_long_side=2048
                )
            except Exception as exc_rt:
                logger.warning("Re-tile after analysis failed (non-fatal): %s", exc_rt)
            results["timings"]["retile"] = round(time.time() - t_retile, 2)

            # ── Stage 2: Photorealistic render ────────────────────────────
            t3 = time.time()
            raw_rendered = generate_photorealistic_render(
                original=image,
                composite=composite,
                mask_overlay=mask_overlay,
                product_name=product_name,
                product_texture=texture,
                analysis=analysis,
                material_type=material_type,
                product_meta=meta,
            )
            results["timings"]["render"] = round(time.time() - t3, 2)

            if raw_rendered:
                # Safety: clip to mask so areas outside the polygon are preserved
                stage2_output = masked_composite(
                    image, raw_rendered, final_mask, feather_radius=3
                )

                # ── Stage 3: Verification & correction ────────────────────
                t4 = time.time()
                skip_verify = os.environ.get("SKIP_VERIFICATION", "").lower() in ("1", "true")
                if not skip_verify:
                    try:
                        verified = verify_and_correct_render(
                            original=image,
                            rendered=stage2_output,
                            mask_overlay=mask_overlay,
                            product_texture=texture,
                        )
                        if verified:
                            final_output = masked_composite(
                                image, verified, final_mask, feather_radius=3
                            )
                        else:
                            final_output = stage2_output
                    except Exception as exc_v:
                        logger.warning("Verification pass failed (non-fatal): %s", exc_v)
                        final_output = stage2_output
                else:
                    final_output = stage2_output
                results["timings"]["verification"] = round(time.time() - t4, 2)

                results["refined"] = image_to_b64(final_output)
                results["gemini_model"] = image_model_name()
            else:
                results["refined"] = results["composite"]
                results["gemini_model"] = "no-image-output"
        except Exception as exc:
            logger.error("Gemini render failed: %s", exc)
            results["refined"] = results["composite"]
            results["gemini_model"] = f"error: {exc}"
    else:
        results["refined"] = results["composite"]
        results["gemini_model"] = "not-configured"
    results["timings"]["gemini_total"] = round(time.time() - t2, 2)

    # ── Watermark ─────────────────────────────────────────────────────────
    if results.get("refined"):
        try:
            from services.gemini import image_to_b64 as _b64
            refined_str = results["refined"]
            raw_b = refined_str.split(",", 1)[-1] if "," in refined_str else refined_str
            refined_img = Image.open(io.BytesIO(base64.b64decode(raw_b))).convert("RGB")
            watermarked = _apply_watermark(refined_img)
            if watermarked is not refined_img:
                results["refined"] = image_to_b64(watermarked)
        except Exception as wm_exc:
            logger.warning("Watermark failed (non-fatal): %s", wm_exc)

    increment_usage(client_ip)

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
            product_name=meta.get("name", req.product_id),
            gemini_model=results.get("gemini_model", "unknown"),
            timings=results.get("timings", {}),
            refined_b64=results.get("refined"),
            thumbnail_b64=thumbnail_b64,
        )
    except Exception as gen_exc:
        logger.warning("Failed to save generation record: %s", gen_exc)

    return results


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/render-stream")
async def render_stream(req: RenderFinalRequest, request: Request):
    """Full rendering pipeline with SSE real-time progress events."""
    client_ip = _get_client_ip(request)
    allowed, used, limit = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            429,
            detail=f"Dzienny limit generacji ({limit}) został osiągnięty. Spróbuj jutro.",
        )

    def _pipeline():
        t0 = time.time()
        timings: dict = {}
        analysis_summary: dict = {}

        # ── Stage: Decode inputs ──────────────────────────────────────────
        yield _sse({"stage": "decode", "status": "running",
                     "message": "Dekodowanie obrazu…"})
        try:
            canvas_w = max(int(req.canvas_width), 1)
            canvas_h = max(int(req.canvas_height), 1)
            image = decode_image(req.image)
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

        # ── Stage: Texture projection ─────────────────────────────────────
        t1 = time.time()
        yield _sse({"stage": "texture", "status": "running",
                     "message": "Projekcja tekstury na ścianę…"})
        try:
            meta, texture = load_product(req.product_id)
            composite = project_texture(
                image, final_mask, texture, meta=meta, polygon=scaled_polygon
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

        mask_overlay = render_mask_overlay(
            image, final_mask, alpha=0.55, max_long_side=2048
        )

        # No Gemini key — skip AI stages
        if not os.environ.get("GEMINI_API_KEY"):
            yield _sse({"stage": "analysis", "status": "skipped",
                         "message": "Brak klucza GEMINI_API_KEY"})
            yield _sse({"stage": "render", "status": "skipped"})
            yield _sse({"stage": "verify", "status": "skipped"})
            timings["total"] = round(time.time() - t0, 2)
            yield _sse({"stage": "done", "ok": True,
                "result": {"composite": composite_b64, "refined": composite_b64,
                           "gemini_model": "not-configured", "timings": timings}})
            return

        product_name = meta.get("name", req.product_id)
        material_type = meta.get(
            "materialType", meta.get("layoutType", "decorative stone/brick")
        )

        # ── Stage: Scene analysis (AI) ────────────────────────────────────
        t2 = time.time()
        yield _sse({"stage": "analysis", "status": "running",
                     "message": "Analiza sceny — pomiary, oświetlenie, przeszkody (Gemini)…"})
        analysis: dict = {}
        try:
            analysis = analyze_wall_scene(image, composite, mask_overlay)
            td = round(time.time() - t2, 2)
            timings["analysis"] = td

            # Build rich summary for frontend
            analysis_summary = {
                "wallHeightCm": analysis.get("wallHeightCm"),
                "wallWidthCm": analysis.get("wallWidthCm"),
                "ceilingHeightCm": analysis.get("ceilingHeightCm"),
                "confidence": analysis.get("confidence"),
                "measurementMethod": analysis.get("measurementMethod"),
                "textureScaleCorrect": analysis.get("textureScaleCorrect"),
                "scaleNote": analysis.get("scaleNote"),
                "referenceObjects": [
                    {"name": r.get("name"), "realHeightCm": r.get("realHeightCm"),
                     "pxPerCm": r.get("pxPerCm"), "confidence": r.get("confidence")}
                    for r in analysis.get("referenceObjects", [])
                ],
                "occluders": [o.get("label") for o in analysis.get("occluders", [])],
                "wallBoundaries": analysis.get("wallBoundaries"),
                "lighting": {
                    k: analysis.get("lighting", {}).get(k)
                    for k in ("primarySource", "temperature", "temperatureKelvin",
                              "gradient", "gradientIntensity", "shadowType",
                              "shadowIntensity", "colorCast", "exposure")
                },
                "perspective": analysis.get("perspective", {}),
            }

            conf = analysis.get("confidence", "?")
            wh = analysis.get("wallHeightCm", "?")
            ww = analysis.get("wallWidthCm", "?")
            refs = [r.get("name") for r in analysis.get("referenceObjects", [])]
            occ = [o.get("label") for o in analysis.get("occluders", [])]
            yield _sse({
                "stage": "analysis", "status": "done", "timing": td,
                "detail": f"Ściana: {wh}×{ww} cm · pewność: {conf}",
                "analysis": analysis_summary,
            })
        except Exception as exc:
            td = round(time.time() - t2, 2)
            timings["analysis"] = td
            yield _sse({"stage": "analysis", "status": "error",
                         "error": str(exc), "timing": td})

        # ── Mask refinement with wall boundaries ──────────────────────────
        wall_boundaries = analysis.get("wallBoundaries")
        if wall_boundaries:
            refined = refine_mask_with_boundaries(final_mask, wall_boundaries)
            if refined.any() and not np.array_equal(refined, final_mask):
                trimmed = int(final_mask.sum() - refined.sum())
                final_mask = refined
                yield _sse({"stage": "mask_refine", "status": "done",
                             "detail": f"Przycięto maskę: {trimmed} px"})

        # ── Re-tile with AI-calibrated dimensions ─────────────────────────
        t_retile = time.time()
        yield _sse({"stage": "retile", "status": "running",
                     "message": "Ponowne kafelkowanie z kalibracją AI…"})
        try:
            composite = project_texture(
                image, final_mask, texture, meta=meta,
                polygon=scaled_polygon, analysis=analysis,
            )
            composite_b64 = image_to_b64(composite)
            mask_overlay = render_mask_overlay(
                image, final_mask, alpha=0.55, max_long_side=2048
            )
            td = round(time.time() - t_retile, 2)
            timings["retile"] = td
            yield _sse({"stage": "retile", "status": "done", "timing": td})
        except Exception as exc:
            td = round(time.time() - t_retile, 2)
            timings["retile"] = td
            yield _sse({"stage": "retile", "status": "warning",
                         "error": str(exc), "timing": td})

        # ── Stage: Photorealistic render (AI) ─────────────────────────────
        t3 = time.time()
        yield _sse({"stage": "render", "status": "running",
                     "message": "Renderowanie fotorealistyczne (Gemini)…"})
        raw_rendered = None
        try:
            raw_rendered = generate_photorealistic_render(
                original=image,
                composite=composite,
                mask_overlay=mask_overlay,
                product_name=product_name,
                product_texture=texture,
                analysis=analysis,
                material_type=material_type,
                product_meta=meta,
            )
            td = round(time.time() - t3, 2)
            timings["render"] = td
            if raw_rendered:
                yield _sse({"stage": "render", "status": "done", "timing": td,
                             "detail": f"model: {image_model_name()}"})
            else:
                yield _sse({"stage": "render", "status": "warning", "timing": td,
                             "error": "Gemini nie zwrócił obrazu"})
        except Exception as exc:
            td = round(time.time() - t3, 2)
            timings["render"] = td
            yield _sse({"stage": "render", "status": "error",
                         "error": str(exc), "timing": td})

        if not raw_rendered:
            # Fall back to composite
            increment_usage(client_ip)
            timings["total"] = round(time.time() - t0, 2)
            yield _sse({"stage": "verify", "status": "skipped",
                         "message": "Pominięto — brak obrazu z rendera"})
            yield _sse({"stage": "done", "ok": True,
                "result": {"composite": composite_b64, "refined": composite_b64,
                           "gemini_model": "no-image-output",
                           "timings": timings, "analysis": analysis_summary}})
            return

        # Clip render to mask
        stage2_output = masked_composite(
            image, raw_rendered, final_mask, feather_radius=3
        )

        # ── Stage: Verification & correction (AI) ────────────────────────
        t4 = time.time()
        skip_verify = os.environ.get("SKIP_VERIFICATION", "").lower() in ("1", "true")
        final_output = stage2_output

        if not skip_verify:
            yield _sse({"stage": "verify", "status": "running",
                         "message": "Weryfikacja i korekta — przeszkody, granice, oświetlenie (Gemini)…"})
            try:
                verified = verify_and_correct_render(
                    original=image,
                    rendered=stage2_output,
                    mask_overlay=mask_overlay,
                    product_texture=texture,
                )
                td = round(time.time() - t4, 2)
                timings["verify"] = td
                if verified:
                    final_output = masked_composite(
                        image, verified, final_mask, feather_radius=3
                    )
                    yield _sse({"stage": "verify", "status": "done", "timing": td,
                                 "detail": "Skorygowano"})
                else:
                    yield _sse({"stage": "verify", "status": "warning", "timing": td,
                                 "error": "Weryfikacja nie zwróciła obrazu"})
            except Exception as exc:
                td = round(time.time() - t4, 2)
                timings["verify"] = td
                yield _sse({"stage": "verify", "status": "error",
                             "error": str(exc), "timing": td})
        else:
            yield _sse({"stage": "verify", "status": "skipped",
                         "message": "Pominięto (SKIP_VERIFICATION=true)"})

        # ── Watermark ─────────────────────────────────────────────────────
        refined_b64 = image_to_b64(final_output)
        try:
            watermarked = _apply_watermark(final_output)
            if watermarked is not final_output:
                refined_b64 = image_to_b64(watermarked)
        except Exception:
            pass

        # ── Save & finalize ───────────────────────────────────────────────
        increment_usage(client_ip)
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
                gemini_model=image_model_name(),
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
                "gemini_model": image_model_name(),
                "timings": timings,
                "analysis": analysis_summary,
                "image_width": image.width,
                "image_height": image.height,
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


@router.post("/api/render-refine")
async def render_refine(req: RenderRefineRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(501, detail="GEMINI_API_KEY not configured.")
    composite = decode_image(req.composite)
    try:
        raw_refined = refine_image(composite, req.product_name)
    except Exception as exc:
        raise HTTPException(502, detail=str(exc))
    if raw_refined is None:
        raise HTTPException(502, detail="Gemini returned no image")
    if req.mask and req.original_image:
        original = decode_image(req.original_image)
        mask = decode_mask_b64(req.mask, original.width, original.height)
        final = masked_composite(original, raw_refined, mask, feather_radius=3)
    else:
        if raw_refined.size != composite.size:
            raw_refined = raw_refined.resize(composite.size, Image.LANCZOS)
        final = raw_refined
    return {"refined": image_to_b64(final)}


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
    render_result = await render_final(render_req)

    return {
        **mask_result,
        **render_result,
        "wall_model": mask_result.get("wall_model", "unknown"),
    }
