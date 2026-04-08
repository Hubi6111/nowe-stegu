"""Gemini AI integration — wall detection + photorealistic render.

detect_wall_mask_with_gemini():
  Vision model analyses user's polygon selection on the room photo and returns
  the precise wall-only polygon + object exclusion rectangles.

generate_photorealistic_render():
  Image model receives original + composite + mask overlay + texture tile
  and produces a photorealistic result strictly within the mask boundary.
"""

import base64
import io
import json
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_FLASH_TEXT = "gemini-2.5-pro"
DEFAULT_GEMINI_FLASH_IMAGE = "gemini-3.1-flash-image-preview"


def text_model_name() -> str:
    return (
        os.environ.get("GEMINI_TEXT_MODEL")
        or os.environ.get("GEMINI_REASONING_MODEL")
        or os.environ.get("GEMINI_FLASH_MODEL")
        or DEFAULT_GEMINI_FLASH_TEXT
    )


def image_model_name() -> str:
    return os.environ.get("GEMINI_IMAGE_MODEL") or DEFAULT_GEMINI_FLASH_IMAGE


# ── Prompts ───────────────────────────────────────────────────────────────────

_RENDER_PROMPT_TEMPLATE = """\
You are an expert photorealistic interior rendering engine with MAXIMUM
decision-making authority over the final result. Your goal: make the textured
wall look indistinguishable from a real renovation photograph taken by a
professional camera.

YOU RECEIVE FOUR IMAGES (in order):
  1. ORIGINAL — the unmodified room photo (ground truth for perspective,
     lighting, objects, and spatial reasoning)
  2. COMPOSITE — room with texture algorithmically placed on the wall
     (use as a rough guide for texture placement and tiling pattern)
  3. MASK OVERLAY — ORIGINAL with ORANGE highlight = user's selection area
  4. PRODUCT TEXTURE TILE — the real decorative material

═══ YOUR PRIMARY AUTHORITY — SCENE ANALYSIS & RENDERING ═══
YOU are the sole intelligence deciding what gets textured and how. The user
drew a rough polygon selection (orange area in image 3). The algorithm placed
texture within that area. Now YOU must:

  1. ANALYSE THE SCENE in the ORIGINAL (image 1):
     - Identify what is WALL SURFACE vs what is NOT wall within the orange zone
     - Identify all objects ON or IN FRONT of the wall: furniture, clocks,
       frames, mirrors, lamps, switches, sockets, shelves, plants, TVs, etc.
     - Identify boundaries: ceiling line, floor/baseboard, side wall corners
     - If the selection extends onto ceiling, floor, or adjacent walls —
       you must NOT place texture there

  2. APPLY TEXTURE ONLY ON THE FLAT WALL SURFACE within the selection:
     - Where the orange zone covers actual wall → apply texture
     - Where the orange zone covers ceiling/floor/side wall → keep ORIGINAL
     - Where the orange zone covers furniture or objects → keep ORIGINAL
     - The texture should exist ONLY on the wall plane the user intended

  3. PERSPECTIVE — study the ORIGINAL carefully:
     - Identify vanishing points and camera angle
     - The texture must follow the same perspective foreshortening
     - Bricks/tiles farther from camera appear smaller
     - Horizontal mortar lines converge toward vanishing points
     - If wall is at an angle, texture must foreshorten accordingly

  4. SCALE CONSISTENCY:
     - The algorithmic composite already placed texture at approximately
       correct real-world scale — preserve that scale
     - Every brick/tile/panel must have the SAME physical size
     - Between renders, brick dimensions must remain consistent

═══ PHYSICAL DIMENSIONS — CRITICAL FOR REALISM ═══
{dimension_instructions}
Use reference objects in the ORIGINAL to validate scale: standard door frames
are ~200 cm, light switches ~120 cm from floor, power sockets ~30 cm from
floor, standard ceiling height ~270-285 cm.

═══ PHOTOREALISTIC RENDERING ═══
Make the textured wall look like a real renovation photograph:
  • Texture boundary: the ORANGE zone is the maximum extent — texture MUST NOT
    extend beyond it. If anything non-wall is inside the zone, do NOT texture
    it. When in doubt, pull texture slightly INWARD.
  • Apply {lighting_temperature} lighting from {lighting_direction}
  • Add realistic ambient occlusion at ceiling/floor junctions
  • Add contact shadows where furniture meets the textured wall
  • Add convincing surface relief and 3D depth for real {material_type}:
    - Individual bricks/tiles should have subtle edge shadows
    - Mortar/grout lines should appear recessed with micro-shadows
    - Surface should show slight roughness variation matching the product
  • Blend edges naturally at boundaries: {blend_notes}
  • Match the room's color temperature, exposure, and white balance exactly
  • EVERYTHING outside the textured wall area must be identical to ORIGINAL

═══ ABSOLUTE RESTRICTIONS ═══
  • Do NOT zoom, crop, pan, reframe, or change image dimensions
  • Do NOT extend texture beyond the orange selection boundary
  • Do NOT texture any non-wall surface (ceiling, floor, adjacent walls)
  • Do NOT cover furniture, objects on the wall, or items in front of the wall
  • Do NOT change room geometry or camera angle
  • KEEP the same tiling pattern and brick/tile layout as in the COMPOSITE
  • Do NOT resize or rescale individual bricks/tiles vs. the COMPOSITE

PRODUCT: {product_name}

Return ONLY the final photorealistic image. No text.
"""

# ── Dimension instruction builder ─────────────────────────────────────────────


def _build_dimension_instructions(meta: dict | None, material_type: str) -> str:
    """Build human-readable physical dimension text for the Gemini render prompt."""
    if not meta:
        return (
            f"This is a decorative {material_type} product. Ensure all modules "
            "appear at consistent, physically correct size across the wall."
        )

    layout = meta.get("layoutType", "running-bond")
    mod_h = float(meta.get("moduleHeightMm", 80))
    mod_w = float(meta.get("moduleWidthMm", 245))
    joint = float(meta.get("jointMm", 10))

    if layout in ("running-bond", "stretcher-bond"):
        return (
            f"Product: decorative brick cladding.\n"
            f"  • Each brick: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(≈ {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Mortar joint between courses: {joint:.0f} mm "
            f"(≈ {joint/10:.1f} cm)\n"
            f"  • Course height (brick + joint): {mod_h + joint:.0f} mm "
            f"(≈ {(mod_h + joint)/10:.1f} cm)\n"
            f"All bricks MUST appear this exact physical size. "
            f"A standard doorway (~200 cm) should fit roughly "
            f"{2000 / (mod_h + joint):.0f} courses next to it. "
            f"Maintain identical brick dimensions across the entire wall — "
            f"only natural perspective foreshortening is allowed."
        )
    elif layout in ("stack-bond",):
        return (
            f"Product: decorative lamels/wall panels.\n"
            f"  • Each panel: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(≈ {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Gap between panels: {joint:.0f} mm "
            f"(≈ {joint/10:.1f} cm)\n"
            f"The spacing between lamels MUST be exactly {joint:.0f} mm "
            f"(~{joint/10:.1f} cm). Panels must be evenly spaced with "
            f"visible gaps. Maintain consistent panel size and gap width "
            f"across the entire wall."
        )
    else:
        return (
            f"Product: decorative {material_type}.\n"
            f"  • Module: {mod_w:.0f} × {mod_h:.0f} mm, joint: {joint:.0f} mm\n"
            f"Maintain consistent module dimensions across the wall."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _parse_json_from_text(text: str) -> dict:
    """Robustly extract the first JSON object from a Gemini text response."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _safe_float(val, default: float = 0.6) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _response_text(response) -> str:
    """Concatenate text parts; never raises on empty / blocked responses."""
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    parts = getattr(response, "parts", None)
    if not parts:
        return ""
    out = []
    for part in parts:
        tx = getattr(part, "text", None)
        if isinstance(tx, str):
            out.append(tx)
    return "".join(out)


def _extract_first_image_from_response(response) -> Image.Image | None:
    """First inline image in the first candidate, or None."""
    if not response.candidates:
        return None
    cand = response.candidates[0]
    if cand.content is None or not cand.content.parts:
        return None
    for part in cand.content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            img_bytes = inline.data
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            try:
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                logger.warning("Could not decode Gemini image part: %s", exc)
    return None


def _gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")
    return genai.Client(api_key=api_key)


import time as _time

def _retry_generate(client, *, model: str, contents, config=None, max_attempts: int = 3):
    """Call client.models.generate_content with automatic retry on transient errors."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: dict = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(k in msg for k in (
                "429", "500", "503", "overloaded", "deadline", "timeout",
                "resource exhausted", "unavailable", "internal",
            ))
            if not transient or attempt == max_attempts:
                logger.error("Gemini call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
                raise
            wait = 2 ** attempt
            logger.warning("Gemini transient error (attempt %d/%d): %s — retrying in %ds",
                           attempt, max_attempts, exc, wait)
            _time.sleep(wait)
    raise last_exc  # unreachable but keeps type checker happy


# ── Scene analysis (lighting + extra exclusions) ──────────────────────────────

_SCENE_ANALYSIS_PROMPT = """\
You are analysing a room photo where a wall texture has been applied.

You receive THREE images:
  1. ORIGINAL room photo
  2. COMPOSITE — room with texture applied on wall
  3. MASK OVERLAY — orange highlight showing where texture is placed

OUTPUT a single JSON object (no markdown):
{
  "extra_exclusions": [{"x":0.5,"y":0.2,"w":0.1,"h":0.15,"label":"clock"}],
  "lighting_direction": "left",
  "lighting_temperature": "warm",
  "ambient_level": 0.65,
  "shadow_intensity": "medium",
  "blend_notes": "soft fade at ceiling; crisp edge at shelf"
}

extra_exclusions: objects VISIBLY covered by the orange mask that should NOT
be textured. Only list truly covered objects. Empty list if mask is clean.
All coordinates normalised 0–1.
"""


def analyze_wall_scene(
    original: Image.Image,
    composite: Image.Image,
    mask_overlay: Image.Image,
) -> dict:
    """Ask Gemini about lighting and any missed exclusions."""
    from google.genai import types

    model_name = text_model_name()
    client = _gemini_client()

    parts: list = [
        types.Part.from_text(text=_SCENE_ANALYSIS_PROMPT),
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(composite), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(mask_overlay), mime_type="image/jpeg"),
    ]

    defaults = {
        "extra_exclusions": [],
        "lighting_direction": "diffuse",
        "lighting_temperature": "neutral",
        "ambient_level": 0.6,
        "shadow_intensity": "medium",
        "blend_notes": "natural fade at edges",
    }

    logger.info("Gemini scene analysis — model: %s", model_name)
    try:
        response = _retry_generate(client, model=model_name, contents=parts)
    except Exception as exc:
        logger.warning("Gemini scene analysis failed: %s", exc)
        return defaults

    raw_text = _response_text(response)
    result = _parse_json_from_text(raw_text)
    result["ambient_level"] = _safe_float(result.get("ambient_level"), 0.6)
    if not isinstance(result.get("extra_exclusions"), list):
        result["extra_exclusions"] = []
    defaults.update(result)
    return defaults


# ── Stage 2: Photorealistic render ────────────────────────────────────────────


def generate_photorealistic_render(
    original: Image.Image,
    composite: Image.Image,
    mask_overlay: Image.Image,
    product_name: str,
    product_texture: Image.Image | None = None,
    analysis: dict | None = None,
    material_type: str = "decorative stone/brick",
    product_meta: dict | None = None,
) -> Image.Image | None:
    """Generate a photorealistic render. Sends 4 images: original, composite,
    mask overlay (orange = texture zone), and product texture tile.
    The mask overlay is critical — it shows Gemini the exact boundary.
    """
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    dim_instructions = _build_dimension_instructions(product_meta, material_type)

    a = analysis or {}
    prompt = _RENDER_PROMPT_TEMPLATE.format(
        lighting_direction=str(a.get("lighting_direction") or "diffuse"),
        lighting_temperature=str(a.get("lighting_temperature") or "neutral"),
        ambient_level=_safe_float(a.get("ambient_level"), 0.6),
        shadow_intensity=str(a.get("shadow_intensity") or "medium"),
        blend_notes=str(a.get("blend_notes") or "natural fade at all edges"),
        material_type=str(material_type or "decorative stone/brick"),
        product_name=str(product_name or "product"),
        dimension_instructions=dim_instructions,
    )

    parts: list = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(composite), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(mask_overlay), mime_type="image/jpeg"),
    ]
    if product_texture:
        parts.append(
            types.Part.from_bytes(
                data=_pil_to_bytes(product_texture), mime_type="image/jpeg",
            )
        )

    logger.info("Gemini photorealistic render — model: %s, images: %d", model_name, len(parts) - 1)
    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
    except Exception as exc:
        logger.warning("Gemini render with modalities failed (%s); retrying without config", exc)
        try:
            response = _retry_generate(client, model=model_name, contents=parts)
        except Exception as exc2:
            logger.error("Gemini render failed after retries: %s", exc2)
            return None

    if response is None:
        return None

    img = _extract_first_image_from_response(response)
    if img is not None:
        return img

    tx = _response_text(response)
    if tx:
        logger.warning("Gemini returned text instead of image: %s", tx[:400])
    return None


# ── Legacy single-call refine (used by /api/render-refine) ────────────────────


_LEGACY_REFINE_PROMPT = """\
You are a photorealistic interior rendering engine.

INPUT: A composite image of a room with a decorative Stegu wall texture
projected onto a specific wall area.  The texture placement, dimensions,
pattern, and boundaries are FINAL — do not change them.

CRITICAL FRAMING: Output must have the EXACT same dimensions, framing, and
perspective as the input.  Do NOT zoom, crop, pan, or reframe.

YOUR ONLY TASK:
• Match room lighting (colour temperature, brightness gradients)
• Add contact shadows at ceiling, floor, and furniture edges
• Add subtle surface relief to the texture
• Blend texture edges naturally into surroundings

ABSOLUTE RESTRICTIONS:
• Do NOT change brick/tile scale, spacing, or pattern
• Do NOT move or expand the textured area
• Do NOT cover objects (clocks, pictures, lamps, mirrors, etc.)
• Do NOT edit anything outside the textured area
• Do NOT zoom, crop, pan, or reframe

PRODUCT: PRODUCT_NAME

Return ONLY the refined image at the exact same resolution.  No text."""


def refine_image(
    composite: Image.Image,
    product_name: str,
    product_texture: Image.Image | None = None,
) -> Image.Image | None:
    """Legacy single-call refinement (kept for /api/render-refine endpoint)."""
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    prompt = _LEGACY_REFINE_PROMPT.replace("PRODUCT_NAME", product_name)
    parts: list = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=_pil_to_bytes(composite), mime_type="image/jpeg"),
    ]
    if product_texture:
        parts.append(
            types.Part.from_bytes(
                data=_pil_to_bytes(product_texture), mime_type="image/jpeg",
            )
        )

    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
    except Exception as exc:
        logger.warning("Gemini refine with modalities failed (%s); retrying without config", exc)
        try:
            response = _retry_generate(client, model=model_name, contents=parts)
        except Exception as exc2:
            logger.error("Gemini refine failed after retries: %s", exc2)
            return None

    if response is None:
        return None

    img = _extract_first_image_from_response(response)
    if img is not None:
        return img

    tx = _response_text(response)
    if tx:
        logger.warning("Gemini refine returned text instead of image: %s", tx[:200])
    return None


# ── Utility ───────────────────────────────────────────────────────────────────


def image_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()
