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
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["refine"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-3.1-flash-image-preview"


class RefineRequest(BaseModel):
    composite: str      # base64 data URL of the generated composite
    original: str       # base64 data URL of the original photo
    texture: str        # base64 data URL of the selected texture
    product_name: str   # e.g. "Cambridge 1"
    material_type: str  # e.g. "decorative brick cladding"


def _strip_data_url(data_url: str) -> tuple[str, bytes]:
    """Extract mime type and raw bytes from a data URL."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
    if match:
        mime = match.group(1)
        raw = base64.b64decode(match.group(2))
        return mime, raw
    # fallback: maybe it's just raw base64
    raw = base64.b64decode(data_url)
    return "image/jpeg", raw


def _to_gemini_part(data_url: str) -> dict:
    """Convert a data URL to a Gemini inline_data part."""
    mime, raw = _strip_data_url(data_url)
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(raw).decode("ascii"),
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

    # Build the request payload for Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "This is the COMPOSITE (computer-generated visualization):"},
                    _to_gemini_part(req.composite),
                    {"text": "This is the ORIGINAL photograph:"},
                    _to_gemini_part(req.original),
                    {"text": "This is the TEXTURE sample:"},
                    _to_gemini_part(req.texture),
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

    t0 = time.time()
    logger.info("Refine request started (model=%s, product=%s)", MODEL_ID, req.product_name)

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)

    elapsed = round(time.time() - t0, 1)
    logger.info("Gemini refine response: %d in %.1fs", resp.status_code, elapsed)

    if resp.status_code != 200:
        detail = resp.text[:500]
        logger.error("Gemini API error: %s", detail)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}")

    data = resp.json()

    # Extract the image from the response
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in response")

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                mime = part["inlineData"]["mimeType"]
                img_b64 = part["inlineData"]["data"]
                result_url = f"data:{mime};base64,{img_b64}"
                logger.info("Refine complete: %.1fs, image returned", elapsed)
                return {"image": result_url, "elapsed": elapsed}

        raise ValueError("No image in response parts")

    except Exception as exc:
        logger.error("Failed to parse Gemini response: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Gemini response: {exc}",
        )
