"""Refine endpoint — Gemini-powered photorealistic refinement.

Takes the composite render + original photo as base64, loads the
texture from disk by product_id, and uses Gemini 3.1 Flash Image
(nanobanana 2) to make it photorealistic.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["refine"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-3.1-flash-image-preview"

MAX_DIM = 1024
JPEG_QUALITY = 85


class RefineRequest(BaseModel):
    composite: str      # base64 data URL of the generated composite
    original: str       # base64 data URL of the original photo
    product_id: str     # product ID — texture loaded from disk
    product_name: str
    material_type: str


def _load_texture_from_disk(product_id: str) -> Image.Image:
    """Load the albedo texture directly from disk."""
    textures_dir = os.environ.get("TEXTURES_DIR", "")
    if not textures_dir:
        raise ValueError("TEXTURES_DIR not set")

    albedo_path = Path(textures_dir) / product_id / "albedo.jpg"
    if not albedo_path.exists():
        albedo_path = Path(textures_dir) / product_id / "albedo.png"
    if not albedo_path.exists():
        raise FileNotFoundError(f"Texture not found: {product_id}")

    logger.info("Loading texture from disk: %s", albedo_path)
    return Image.open(albedo_path).convert("RGB")


def _decode_data_url(data_url: str) -> Image.Image:
    """Decode a data URL to PIL Image, handling all edge cases."""
    if not data_url:
        raise ValueError("Empty image data")

    # Find base64 data after the comma
    if "," in data_url:
        raw_b64 = data_url.split(",", 1)[1]
    else:
        raw_b64 = data_url

    # Clean up
    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")

    # Fix padding
    pad = len(raw_b64) % 4
    if pad:
        raw_b64 += "=" * (4 - pad)

    raw_bytes = base64.b64decode(raw_b64)

    if len(raw_bytes) < 100:
        raise ValueError(f"Image data too small: {len(raw_bytes)} bytes")

    logger.info("Decoded data URL: %d bytes, starts with %s", len(raw_bytes), raw_bytes[:4].hex())
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _img_to_gemini_part(img: Image.Image, max_dim: int = MAX_DIM) -> dict:
    """Resize a PIL Image and encode as Gemini inline_data part."""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    logger.info("Prepared for Gemini: %dx%d → %d KB", img.width, img.height, len(b64) // 1024)

    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": b64,
        }
    }


REFINE_PROMPT = """You are a professional architectural visualization artist. You have been given three images:

1. **COMPOSITE** — A computer-generated visualization where a decorative wall texture ({product_name} — {material_type}) has been digitally projected onto a wall in an interior photo. This is a rough composite that needs refinement.

2. **ORIGINAL** — The original, unmodified interior photograph showing the room as it actually looks.

3. **TEXTURE** — The raw texture/material sample ({product_name}) that was applied to the wall.

Your task is to produce a SINGLE photorealistic result image that:

**CRITICAL REQUIREMENTS:**
- The wall area that was textured in the COMPOSITE must remain covered with the SAME texture pattern ({product_name}), maintaining the exact same scale, placement, and tiling pattern
- Fix any visible seams, gaps, or imperfections in the texture coverage — the wall should look like it was genuinely clad with this material
- Add realistic shadows and ambient occlusion where the textured wall meets furniture, floor, ceiling, door frames, and other objects
- Match the lighting conditions from the ORIGINAL photo — if there's warm light from a window, the textured wall should reflect that same warmth and direction
- Preserve ALL non-wall elements exactly as they appear in the ORIGINAL: furniture, doors, windows, floor, ceiling, decorations, plants — do not alter anything except the wall surface
- The edges where texture meets objects (furniture, frames, outlets) should be clean and natural with proper shadow transitions
- Add subtle light reflections on the texture surface consistent with the material properties (matte for brick, slight sheen for stone)
- The final result should be completely indistinguishable from a real photograph of a room with this wall cladding installed

**OUTPUT:** Generate exactly one refined, photorealistic image. Do NOT include any text, labels, or watermarks."""


@router.post("/refine")
async def refine_render(req: RefineRequest):
    """Refine a composite render using Gemini image editing."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY nie ustawiony w .env")

    import asyncio
    import httpx

    prompt = REFINE_PROMPT.format(
        product_name=req.product_name,
        material_type=req.material_type,
    )

    # ── Load and prepare images ──────────────────────────────────────
    try:
        logger.info("Decoding composite image...")
        composite_img = _decode_data_url(req.composite)
        logger.info("Composite: %dx%d", composite_img.width, composite_img.height)

        logger.info("Decoding original image...")
        original_img = _decode_data_url(req.original)
        logger.info("Original: %dx%d", original_img.width, original_img.height)

        logger.info("Loading texture from disk for product: %s", req.product_id)
        texture_img = _load_texture_from_disk(req.product_id)
        logger.info("Texture: %dx%d", texture_img.width, texture_img.height)

        composite_part = _img_to_gemini_part(composite_img, max_dim=MAX_DIM)
        original_part = _img_to_gemini_part(original_img, max_dim=MAX_DIM)
        texture_part = _img_to_gemini_part(texture_img, max_dim=512)

    except Exception as exc:
        logger.error("Image preparation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Błąd przygotowania obrazów: {exc}")

    # ── Build Gemini request ─────────────────────────────────────────
    payload = {
        "contents": [{
            "parts": [
                {"text": "COMPOSITE (computer-generated visualization):"},
                composite_part,
                {"text": "ORIGINAL photograph:"},
                original_part,
                {"text": "TEXTURE sample:"},
                texture_part,
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.2,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    )

    # ── Retry up to 3 times ──────────────────────────────────────────
    last_error = None
    for attempt in range(3):
        t0 = time.time()
        logger.info("Gemini refine attempt %d/3", attempt + 1)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)

            elapsed = round(time.time() - t0, 1)
            logger.info("Gemini response: HTTP %d in %.1fs", resp.status_code, elapsed)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = "Brak wyników od modelu"
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        mime = part["inlineData"]["mimeType"]
                        img_b64 = part["inlineData"]["data"]
                        return {"image": f"data:{mime};base64,{img_b64}", "elapsed": elapsed}

                finish_reason = candidates[0].get("finishReason", "unknown")
                last_error = f"Model nie wygenerował obrazu ({finish_reason})"
                continue

            elif resp.status_code == 429:
                last_error = "Zbyt wiele żądań — spróbuj za chwilę"
                await asyncio.sleep(5 * (attempt + 1))
                continue

            elif resp.status_code == 400:
                logger.error("Gemini 400: %s", resp.text[:300])
                if attempt < 2:
                    # Retry with smaller images
                    composite_part = _img_to_gemini_part(composite_img, max_dim=768)
                    original_part = _img_to_gemini_part(original_img, max_dim=768)
                    payload["contents"][0]["parts"][1] = composite_part
                    payload["contents"][0]["parts"][3] = original_part
                    last_error = "Ponawiam z mniejszymi obrazami…"
                    continue
                last_error = "Nie udało się przetworzyć obrazów"
                break
            else:
                last_error = f"Błąd API (HTTP {resp.status_code})"
                continue

        except httpx.TimeoutException:
            last_error = "Przekroczono czas oczekiwania — spróbuj ponownie"
            continue
        except Exception as exc:
            last_error = f"Błąd połączenia: {exc}"
            continue

    raise HTTPException(status_code=502, detail=last_error or "Nie udało się wygenerować renderingu")
