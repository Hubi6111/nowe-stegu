"""Refine endpoint — Gemini-powered photorealistic refinement.

Takes the composite render, original photo, and texture, and uses
Gemini 3.1 Flash Image (nanobanana 2) to make it photorealistic.
"""

import base64
import io
import logging
import os
import re
import time

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["refine"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-3.1-flash-image-preview"

# Max image dimension sent to Gemini (keeps payload under limits)
MAX_DIM = 1024
JPEG_QUALITY = 85


class RefineRequest(BaseModel):
    composite: str      # base64 data URL of the generated composite
    original: str       # base64 data URL of the original photo
    texture: str        # base64 data URL of the selected texture
    product_name: str   # e.g. "Cambridge 1"
    material_type: str  # e.g. "decorative brick cladding"


def _decode_image(data_url: str) -> Image.Image:
    """Decode a data URL or raw base64 string into a PIL Image."""
    # Strip data URL prefix if present
    if "base64," in data_url:
        raw_b64 = data_url.split("base64,", 1)[1]
    else:
        raw_b64 = data_url

    raw_bytes = base64.b64decode(raw_b64)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _prepare_for_gemini(data_url: str, max_dim: int = MAX_DIM) -> dict:
    """Decode, resize, re-encode as JPEG, and format as Gemini inline_data part."""
    img = _decode_image(data_url)

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resized image %dx%d → %dx%d for Gemini", w, h, new_w, new_h)

    # Encode as JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

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
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Set it in .env",
        )

    import httpx

    prompt = REFINE_PROMPT.format(
        product_name=req.product_name,
        material_type=req.material_type,
    )

    # Prepare images — resize and re-encode as JPEG
    try:
        composite_part = _prepare_for_gemini(req.composite, max_dim=MAX_DIM)
        original_part = _prepare_for_gemini(req.original, max_dim=MAX_DIM)
        texture_part = _prepare_for_gemini(req.texture, max_dim=512)  # texture is smaller
    except Exception as exc:
        logger.error("Failed to prepare images: %s", exc)
        raise HTTPException(status_code=400, detail=f"Błąd przygotowania obrazów: {exc}")

    # Build the request payload for Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "COMPOSITE (computer-generated visualization):"},
                    composite_part,
                    {"text": "ORIGINAL photograph:"},
                    original_part,
                    {"text": "TEXTURE sample:"},
                    texture_part,
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.2,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    )

    # Retry up to 3 times
    last_error = None
    for attempt in range(3):
        t0 = time.time()
        logger.info(
            "Refine request attempt %d/3 (model=%s, product=%s)",
            attempt + 1, MODEL_ID, req.product_name,
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)

            elapsed = round(time.time() - t0, 1)
            logger.info("Gemini refine response: %d in %.1fs", resp.status_code, elapsed)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])

                if not candidates:
                    last_error = "Brak wyników od modelu"
                    logger.warning("No candidates, retrying...")
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        mime = part["inlineData"]["mimeType"]
                        img_b64 = part["inlineData"]["data"]
                        result_url = f"data:{mime};base64,{img_b64}"
                        logger.info("Refine complete: %.1fs", elapsed)
                        return {"image": result_url, "elapsed": elapsed}

                # No image in response — might be safety filter or text-only
                finish_reason = candidates[0].get("finishReason", "unknown")
                if finish_reason == "IMAGE_SAFETY":
                    last_error = "Obraz został odrzucony przez filtr bezpieczeństwa"
                    logger.warning("Safety filter triggered")
                    continue
                else:
                    last_error = f"Model nie wygenerował obrazu (reason: {finish_reason})"
                    logger.warning("No image in parts, finish=%s", finish_reason)
                    continue

            elif resp.status_code == 429:
                last_error = "Zbyt wiele żądań — spróbuj za chwilę"
                wait = 5 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds...", wait)
                import asyncio
                await asyncio.sleep(wait)
                continue

            elif resp.status_code == 400:
                detail = resp.text[:300]
                logger.error("Gemini 400 error: %s", detail)
                # Try reducing image size further
                if attempt < 2:
                    logger.info("Retrying with smaller images...")
                    composite_part = _prepare_for_gemini(req.composite, max_dim=768)
                    original_part = _prepare_for_gemini(req.original, max_dim=768)
                    last_error = "Błąd przetwarzania — ponawiam z mniejszymi obrazami"
                    continue
                last_error = "Nie udało się przetworzyć obrazów"
                break

            else:
                last_error = f"Błąd API (HTTP {resp.status_code})"
                logger.error("Gemini API error %d: %s", resp.status_code, resp.text[:300])
                if attempt < 2:
                    continue
                break

        except httpx.TimeoutException:
            elapsed = round(time.time() - t0, 1)
            last_error = "Przekroczono czas oczekiwania — spróbuj ponownie"
            logger.warning("Timeout after %.1fs on attempt %d", elapsed, attempt + 1)
            continue
        except Exception as exc:
            last_error = f"Błąd połączenia: {exc}"
            logger.error("Request failed: %s", exc)
            if attempt < 2:
                continue
            break

    raise HTTPException(status_code=502, detail=last_error or "Nie udało się wygenerować renderingu")
