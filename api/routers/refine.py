"""Refine endpoint — Gemini-powered photorealistic refinement.

Loads the composite and original images from disk (saved by pipeline),
loads the texture from disk by product_id, then sends all three to
Gemini 3.1 Flash Image (nanobanana 2) for photorealistic refinement.

NO images are sent through JSON — everything is loaded from disk.
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
    product_id: str
    product_name: str
    material_type: str


def _load_image(path: Path) -> Image.Image:
    """Load an image from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    logger.info("Loading image: %s", path)
    return Image.open(path).convert("RGB")


def _img_to_gemini_part(img: Image.Image, max_dim: int = MAX_DIM) -> dict:
    """Resize a PIL Image and encode as Gemini inline_data part."""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    logger.info("Prepared for Gemini: %dx%d → %d KB JPEG", img.width, img.height, len(buf.getvalue()) // 1024)

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


@router.get("/refine-status")
async def refine_status():
    """Check if refine images are saved and ready."""
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    refine_dir = data_dir / "refine-temp"
    textures_dir = Path(os.environ.get("TEXTURES_DIR", ""))

    composite_path = refine_dir / "composite.jpg"
    original_path = refine_dir / "original.jpg"

    result = {
        "data_dir": str(data_dir),
        "refine_dir": str(refine_dir),
        "textures_dir": str(textures_dir),
        "composite_exists": composite_path.exists(),
        "original_exists": original_path.exists(),
        "composite_size": composite_path.stat().st_size if composite_path.exists() else 0,
        "original_size": original_path.stat().st_size if original_path.exists() else 0,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "model": MODEL_ID,
    }
    logger.info("Refine status: %s", result)
    return result


@router.post("/refine")
async def refine_render(req: RefineRequest):
    """Refine a composite render using Gemini image editing.

    All images are loaded from disk:
    - composite: data/refine-temp/composite.jpg (saved by pipeline)
    - original: data/refine-temp/original.jpg (saved by pipeline)
    - texture: assets/textures/stegu/{product_id}/albedo.jpg (on disk)
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY nie ustawiony w .env")

    import asyncio
    import httpx

    prompt = REFINE_PROMPT.format(
        product_name=req.product_name,
        material_type=req.material_type,
    )

    # ── Load all images from disk ────────────────────────────────────
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    refine_dir = data_dir / "refine-temp"
    textures_dir = Path(os.environ.get("TEXTURES_DIR", ""))

    try:
        composite_img = _load_image(refine_dir / "composite.jpg")
        logger.info("Composite loaded: %dx%d", composite_img.width, composite_img.height)

        original_img = _load_image(refine_dir / "original.jpg")
        logger.info("Original loaded: %dx%d", original_img.width, original_img.height)

        # Find texture albedo
        tex_path = textures_dir / req.product_id / "albedo.jpg"
        if not tex_path.exists():
            tex_path = textures_dir / req.product_id / "albedo.png"
        texture_img = _load_image(tex_path)
        logger.info("Texture loaded: %dx%d", texture_img.width, texture_img.height)

        composite_part = _img_to_gemini_part(composite_img, max_dim=MAX_DIM)
        original_part = _img_to_gemini_part(original_img, max_dim=MAX_DIM)
        texture_part = _img_to_gemini_part(texture_img, max_dim=512)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        raise HTTPException(status_code=400, detail=f"Brak zapisanych obrazów — wygeneruj wizualizację ponownie")
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
        logger.info("Gemini refine attempt %d/3 (product=%s)", attempt + 1, req.product_name)

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
