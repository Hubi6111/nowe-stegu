"""Gemini AI integration — 3-stage photorealistic rendering pipeline.

Stage 1: Scene measurement — multi-reference calibration for real-world dimensions
Stage 2: Photorealistic render — texture placement with perfect lighting
Stage 3: Verification — compare with original, fix occlusion & lighting
"""

import base64
import io
import json
import logging
import os
import re
import time as _time

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _parse_json_from_text(text: str) -> dict:
    """Robustly extract the first JSON object from a Gemini text response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Scene Measurement & Analysis
# Multi-reference calibration + lighting + occlusion detection
# ══════════════════════════════════════════════════════════════════════════════

_SCENE_ANALYSIS_PROMPT = """\
You are a forensic architectural measurement system. Your task: measure the \
EXACT real-world dimensions of a wall area and catalog everything visible in \
the scene for photorealistic rendering.

You receive THREE images:
  1. ORIGINAL — unmodified room/building photograph
  2. COMPOSITE — same photo with decorative texture tiled on the wall
  3. MASK OVERLAY — ORIGINAL with ORANGE semi-transparent highlight = target wall area

═══ TASK A: MULTI-REFERENCE DIMENSIONAL CALIBRATION ═══

You MUST find and use EVERY reference object visible anywhere in the photo to \
triangulate the wall dimensions. Known real-world sizes (use as many as apply):

DOORS (PRIMARY — most reliable, error ±3%):
  • Standard interior door leaf: 200 cm × 80 cm
  • Door frame/architrave adds: 6-8 cm each side, 5-8 cm top
  • Total opening with frame: ~210 cm × ~95 cm
  • French/balcony door: 200-220 cm × 80-180 cm
  • Closet sliding door: 230-250 cm × 60-90 cm per panel
  • Front/exterior door: 200-210 cm × 90-100 cm

WINDOWS (error ±5%):
  • Standard window: 120-150 cm tall × 80-120 cm wide
  • Window sill from floor: 85-95 cm (interior), 90-100 cm (exterior)
  • Curtain rod to ceiling: 10-20 cm
  • Roller blind cassette: 7-10 cm tall

ARCHITECTURAL (error ±3%):
  • Standard EU ceiling height: 250-280 cm (most common: 260-270 cm)
  • Standard US ceiling height: 244 cm (8 ft) or 274 cm (9 ft)
  • Baseboard/skirting: 8-12 cm tall
  • Crown molding: 5-10 cm
  • Stair riser: 17-19 cm, tread: 25-30 cm
  • Standard brick (visible outside): 6.5 cm × 25 cm
  • Concrete block: 19 cm × 39 cm
  • Radiator height: 30 cm (low), 50-60 cm (standard), 90+ cm (tall)
  • Radiator from floor: 10-15 cm

ELECTRICAL (error ±5%):
  • Light switch plate: 8 × 8 cm, center 110-120 cm from floor
  • Double switch plate: 8 × 15 cm
  • Power outlet plate: 8 × 8 cm, center 25-30 cm from floor
  • Kitchen counter outlet: 100-110 cm from floor
  • Thermostat: 8 × 12 cm, center ~150 cm from floor

FURNITURE (error ±8%):
  • Dining table height: 73-76 cm
  • Kitchen counter/island: 85-90 cm tall
  • Bar counter: 100-110 cm tall
  • Chair seat: 43-47 cm from floor
  • Chair backrest top: 85-100 cm from floor
  • Sofa seat: 40-45 cm, backrest: 80-90 cm total
  • Coffee table: 40-50 cm tall
  • Bookshelf: 180-200 cm (standard), 70-80 cm (low)
  • Desk: 72-76 cm tall
  • Bed frame headboard: 90-120 cm from floor
  • Bedside table: 55-65 cm tall

APPLIANCES/OBJECTS (error ±3%):
  • TV 55": ~68 × 122 cm. TV 65": ~81 × 144 cm
  • Refrigerator: 170-185 cm tall (standard), 60-70 cm wide
  • Washing machine: 85 cm tall × 60 cm wide
  • Microwave: 28-30 cm tall
  • Standard A4 frame: 21 × 30 cm
  • Wall clock: 25-40 cm diameter
  • Wine bottle on shelf: 30 cm tall
  • Standard door handle from floor: 100-110 cm

HUMAN PROPORTIONS (if people visible, error ±5%):
  • Adult standing: 165-180 cm
  • Shoulder height: 140-155 cm
  • Elbow height: 100-110 cm

MEASUREMENT PROCEDURE (execute ALL steps):
  1. SCAN the entire image systematically — left to right, top to bottom
  2. LIST every reference object you can identify with its approximate pixel span
  3. For the 3 most reliable references, calculate px/cm ratio independently
  4. CROSS-CHECK: all ratios should agree within 15%. Use the median.
  5. Apply the calibrated px/cm to the orange-highlighted wall area
  6. VERIFY: does the resulting wall height make sense given ceiling height?
  7. VERIFY: would a standard door (200cm) fit proportionally?

═══ TASK B: OCCLUSION CATALOG ═══

List EVERY object that is IN FRONT OF or ON the wall surface within the \
orange-highlighted area. These objects must NOT be covered by texture:
  • Furniture touching or near the wall (shelves, TV, frames, mirrors)
  • Architectural elements (columns, beams, window frames, door frames)
  • Electrical (switches, outlets, thermostats)
  • Decorative (clocks, plants, lamps, sconces, artwork)
  • Structural (pipes, vents, radiators)

For each occluder, provide its bounding box in normalized 0-1 coordinates \
relative to the full image.

═══ TASK C: WALL BOUNDARY ANALYSIS ═══

The orange mask is the USER's rough selection. But texture must ONLY go on \
the actual wall surface. Identify precisely:
  • Where does the wall meet the CEILING? (y coordinate, normalized 0-1)
  • Where does the wall meet the FLOOR/baseboard? (y coordinate, normalized 0-1)
  • Where are LEFT and RIGHT wall boundaries? (corners, door frames, windows)
  • Does the orange selection extend BEYOND the wall onto ceiling/floor/other walls?
  • If yes: which parts of the selection should be EXCLUDED?

═══ TASK D: LIGHTING ANALYSIS ═══

Analyze the EXACT lighting on the wall surface:
  • Primary light source position relative to wall (e.g. "window left, 45° angle")
  • Secondary/fill light sources
  • Color temperature: warm (2700-3500K), neutral (4000-5000K), cool (5500K+)
  • Brightness gradient across the wall (which region is brightest/darkest?)
  • Shadow characteristics: hard (direct sun) vs soft (overcast/diffuse)
  • Ambient occlusion darkness at ceiling/floor/corner junctions
  • Any color cast from nearby colored surfaces (warm floor, colored wall, etc.)
  • Overall exposure level: underexposed / correct / overexposed

═══ TASK E: PERSPECTIVE ANALYSIS ═══

  • Camera angle to wall: frontal (0-10°), moderate (10-30°), strong (30°+)
  • If angled: which side recedes? (left/right)
  • Vanishing point direction for horizontal lines
  • Vertical convergence (looking up/down)?
  • Lens distortion visible? (barrel/pincushion)

═══ TASK F: TEXTURE SCALE VALIDATION ═══

Look at image 2 (COMPOSITE). Do the texture elements appear the correct \
physical size compared to reference objects? Are they too big or too small?

OUTPUT a single JSON object (no markdown, no backticks):
{
  "wallHeightCm": 255,
  "wallWidthCm": 340,
  "ceilingHeightCm": 265,
  "measurementMethod": "door frame at left edge used as primary (200cm), light switch confirmed (115cm from floor), ceiling height cross-checked",
  "referenceObjects": [
    {"name": "interior door", "pixelHeight": 520, "realHeightCm": 200, "pxPerCm": 2.6, "confidence": "high"},
    {"name": "light switch", "pixelHeight": 21, "realHeightCm": 8, "pxPerCm": 2.63, "confidence": "medium"},
    {"name": "baseboard", "pixelHeight": 26, "realHeightCm": 10, "pxPerCm": 2.6, "confidence": "medium"}
  ],
  "calibratedPxPerCm": 2.6,
  "occluders": [
    {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.65, "label": "bookshelf", "depth": "touching_wall"},
    {"x": 0.7, "y": 0.5, "w": 0.08, "h": 0.04, "label": "light_switch", "depth": "on_wall"}
  ],
  "wallBoundaries": {
    "ceilingLineY": 0.02,
    "floorLineY": 0.95,
    "leftEdgeX": 0.0,
    "rightEdgeX": 1.0,
    "selectionExceedsCeiling": false,
    "selectionExceedsFloor": false,
    "selectionExceedsLeftWall": false,
    "selectionExceedsRightWall": false,
    "ceilingExclusionZone": null,
    "floorExclusionZone": null
  },
  "lighting": {
    "primarySource": "window left, natural daylight",
    "secondarySource": "ceiling fixture, warm LED",
    "temperature": "neutral-warm",
    "temperatureKelvin": 4500,
    "gradient": "brighter-left-darker-right",
    "gradientIntensity": 0.3,
    "shadowType": "soft-diffuse",
    "shadowIntensity": 0.4,
    "ambientOcclusion": "visible at ceiling junction and left corner",
    "colorCast": "slight warm cast from wooden floor",
    "exposure": "correct"
  },
  "perspective": {
    "type": "frontal",
    "angleDeg": 5,
    "recedes": "none",
    "horizontalConvergence": "negligible",
    "verticalConvergence": "none",
    "lensDistortion": "none"
  },
  "textureScaleCorrect": true,
  "scaleNote": "bricks correctly sized — ~27 courses visible matches expected 28 for 255cm wall",
  "confidence": "high"
}"""


def analyze_wall_scene(
    original: Image.Image,
    composite: Image.Image,
    mask_overlay: Image.Image,
) -> dict:
    """Stage 1: Comprehensive scene analysis with multi-reference measurement."""
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
        "wallHeightCm": 260,
        "wallWidthCm": 350,
        "ceilingHeightCm": 265,
        "calibratedPxPerCm": 2.0,
        "referenceObjects": [],
        "occluders": [],
        "wallBoundaries": {
            "ceilingLineY": 0.0,
            "floorLineY": 1.0,
            "leftEdgeX": 0.0,
            "rightEdgeX": 1.0,
            "selectionExceedsCeiling": False,
            "selectionExceedsFloor": False,
            "selectionExceedsLeftWall": False,
            "selectionExceedsRightWall": False,
        },
        "lighting": {
            "primarySource": "diffuse ambient",
            "temperature": "neutral",
            "temperatureKelvin": 5000,
            "gradient": "even",
            "gradientIntensity": 0.1,
            "shadowType": "soft-diffuse",
            "shadowIntensity": 0.3,
            "ambientOcclusion": "subtle at junctions",
            "colorCast": "none",
            "exposure": "correct",
        },
        "perspective": {
            "type": "frontal",
            "angleDeg": 0,
            "recedes": "none",
            "horizontalConvergence": "none",
            "verticalConvergence": "none",
        },
        "textureScaleCorrect": True,
        "scaleNote": "",
        "confidence": "low",
        # Legacy keys for backward compat with render prompt
        "lighting_direction": "diffuse",
        "lighting_temperature": "neutral",
        "ambient_level": 0.6,
        "shadow_intensity": "medium",
        "blend_notes": "natural fade at edges",
        "extra_exclusions": [],
    }

    logger.info("Gemini scene analysis — model: %s", model_name)
    try:
        response = _retry_generate(client, model=model_name, contents=parts)
    except Exception as exc:
        logger.warning("Gemini scene analysis failed: %s", exc)
        return defaults

    raw_text = _response_text(response)
    result = _parse_json_from_text(raw_text)

    if not result:
        logger.warning("Gemini scene analysis returned no parseable JSON")
        return defaults

    # Merge results into defaults, preserving structure
    for key in ("wallHeightCm", "wallWidthCm", "ceilingHeightCm", "calibratedPxPerCm",
                "textureScaleCorrect", "scaleNote", "confidence", "measurementMethod"):
        if key in result:
            defaults[key] = result[key]

    if isinstance(result.get("referenceObjects"), list):
        defaults["referenceObjects"] = result["referenceObjects"]

    if isinstance(result.get("occluders"), list):
        defaults["occluders"] = result["occluders"]
        # Populate legacy extra_exclusions for backward compat
        defaults["extra_exclusions"] = [
            {"x": o.get("x", 0), "y": o.get("y", 0),
             "w": o.get("w", 0), "h": o.get("h", 0),
             "label": o.get("label", "object")}
            for o in result["occluders"]
        ]

    if isinstance(result.get("wallBoundaries"), dict):
        defaults["wallBoundaries"].update(result["wallBoundaries"])

    if isinstance(result.get("lighting"), dict):
        defaults["lighting"].update(result["lighting"])
        # Populate legacy keys
        lt = defaults["lighting"]
        defaults["lighting_direction"] = lt.get("primarySource", "diffuse")
        defaults["lighting_temperature"] = lt.get("temperature", "neutral")
        defaults["shadow_intensity"] = lt.get("shadowType", "medium")
        gi = _safe_float(lt.get("gradientIntensity"), 0.1)
        defaults["ambient_level"] = max(0.3, 1.0 - gi)
        defaults["blend_notes"] = lt.get("ambientOcclusion", "natural fade at edges")

    if isinstance(result.get("perspective"), dict):
        defaults["perspective"].update(result["perspective"])

    # Sanity: wall can't be taller than ceiling
    wh = _safe_float(defaults.get("wallHeightCm"), 260)
    ch = _safe_float(defaults.get("ceilingHeightCm"), 265)
    if wh > ch + 10:
        defaults["wallHeightCm"] = ch

    logger.info(
        "Scene analysis: wall=%s×%scm, refs=%s, occluders=%d, confidence=%s",
        defaults.get("wallHeightCm"), defaults.get("wallWidthCm"),
        [r.get("name") for r in defaults.get("referenceObjects", [])],
        len(defaults.get("occluders", [])),
        defaults.get("confidence"),
    )

    return defaults


# ══════════════════════════════════════════════════════════════════════════════
# Dimension instruction builder (enhanced)
# ══════════════════════════════════════════════════════════════════════════════


def _build_dimension_instructions(
    meta: dict | None,
    material_type: str,
    analysis: dict | None = None,
) -> str:
    """Build dimension text using product specs + scene measurements."""
    if not meta:
        return (
            f"This is a decorative {material_type} product. Ensure all modules "
            "appear at consistent, physically correct size across the wall."
        )

    layout = meta.get("layoutType", "running-bond")
    mod_h = float(meta.get("moduleHeightMm", 80))
    mod_w = float(meta.get("moduleWidthMm", 245))
    joint = float(meta.get("jointMm", 10))

    wall_h_cm = 260
    wall_w_cm = 350
    refs_text = "estimated from room proportions"
    if analysis:
        wall_h_cm = _safe_float(analysis.get("wallHeightCm"), 260)
        wall_w_cm = _safe_float(analysis.get("wallWidthCm"), 350)
        ref_objs = analysis.get("referenceObjects", [])
        if ref_objs:
            refs_text = ", ".join(
                f"{r.get('name', '?')} (~{r.get('realHeightCm', '?')}cm)"
                for r in ref_objs[:4]
            )

    course_h_mm = mod_h + joint
    unit_w_mm = mod_w + joint
    courses_in_wall = (wall_h_cm * 10) / course_h_mm
    units_in_row = (wall_w_cm * 10) / unit_w_mm

    if layout in ("running-bond", "stretcher-bond"):
        return (
            f"PRODUCT: decorative brick cladding\n"
            f"  • Each brick: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(≈ {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Mortar joint: {joint:.0f} mm (≈ {joint/10:.1f} cm)\n"
            f"  • Course height (brick + joint): {course_h_mm:.0f} mm "
            f"(≈ {course_h_mm/10:.1f} cm)\n"
            f"\n"
            f"WALL DIMENSIONS (measured using: {refs_text}):\n"
            f"  • Wall height: ~{wall_h_cm:.0f} cm → should contain ~{courses_in_wall:.0f} courses\n"
            f"  • Wall width: ~{wall_w_cm:.0f} cm → should contain ~{units_in_row:.0f} bricks per row\n"
            f"\n"
            f"VERIFICATION CROSS-CHECKS:\n"
            f"  • A standard door (200 cm) should span ~{2000/course_h_mm:.0f} courses alongside it\n"
            f"  • A light switch (at 115 cm from floor) should be at ~{1150/course_h_mm:.0f} courses from floor\n"
            f"  • 25 courses from the floor = ~{25*course_h_mm/10:.0f} cm = about knee height on an adult\n"
            f"  • All bricks MUST appear this exact physical size. Only natural perspective "
            f"foreshortening is allowed — bricks farther from camera appear smaller."
        )
    elif layout in ("stack-bond", "vertical-stack"):
        return (
            f"PRODUCT: decorative panels/lamels\n"
            f"  • Each panel: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(≈ {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Gap between panels: {joint:.0f} mm (≈ {joint/10:.1f} cm)\n"
            f"\n"
            f"WALL DIMENSIONS (measured using: {refs_text}):\n"
            f"  • Wall height: ~{wall_h_cm:.0f} cm\n"
            f"  • Wall width: ~{wall_w_cm:.0f} cm → should fit ~{units_in_row:.0f} panels\n"
            f"\n"
            f"The spacing between panels MUST be exactly {joint:.0f} mm "
            f"(~{joint/10:.1f} cm). Panels must be evenly spaced with visible gaps."
        )
    else:
        return (
            f"PRODUCT: decorative {material_type}\n"
            f"  • Module: {mod_w:.0f} × {mod_h:.0f} mm, joint: {joint:.0f} mm\n"
            f"\n"
            f"WALL: ~{wall_h_cm:.0f} × {wall_w_cm:.0f} cm (measured using: {refs_text})\n"
            f"Maintain consistent module dimensions across the wall."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Photorealistic Render
# ══════════════════════════════════════════════════════════════════════════════

_RENDER_PROMPT_TEMPLATE = """\
You are a photorealistic PHOTO EDITOR. Your job is to take an existing \
COMPOSITE image and make the textured wall area look like a real photograph. \
This is IMAGE EDITING — you are NOT generating a new image from scratch.

YOU RECEIVE FIVE IMAGES (in order):
  1. ORIGINAL — unmodified room photo (ground truth for everything outside wall)
  2. COMPOSITE — room with "{product_name}" texture already placed at CORRECT \
real-world scale
  3. MASK OVERLAY — ORIGINAL with ORANGE highlight = target wall area
  4. PRODUCT TEXTURE TILE — material swatch (color/detail reference only)
  5. ORIGINAL again (for direct comparison)

╔══════════════════════════════════════════════════════════════════╗
║  #1 CRITICAL RULE — PATTERN LOCK                                ║
║                                                                  ║
║  The COMPOSITE (image 2) has SCIENTIFICALLY CORRECT texture.    ║
║  It was computed from real product dimensions in millimetres.   ║
║  You MUST preserve the EXACT pattern:                            ║
║                                                                  ║
║  → COUNT every {unit_label} in the COMPOSITE image              ║
║  → Your output MUST have the IDENTICAL number of {unit_label}s  ║
║  → Each {unit_label} must be the SAME WIDTH as in the COMPOSITE ║
║  → The {joint_label} spacing must be IDENTICAL                   ║
║                                                                  ║
║  DO NOT re-draw, re-tile, or re-imagine the texture.            ║
║  DO NOT make {unit_label}s wider, narrower, taller, or shorter. ║
║  If the COMPOSITE shows narrow {unit_label}s — keep them narrow.║
║  If it shows many {unit_label}s — keep the same count.          ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  #2 WALL ONLY — never ceiling, floor, adjacent walls,      ║
║  furniture, or objects in front of the wall.                ║
╚══════════════════════════════════════════════════════════════╝

═══ YOUR TASK (photo editing, NOT generation) ═══

Start with the COMPOSITE. Apply ONLY these photorealistic adjustments:
  1. Match the ORIGINAL's lighting/shadows onto the textured wall
  2. Add surface relief (micro-shadows on {shadow_edge}, recessed {joint_label}s)
  3. Blend wall edges naturally (thin AO shadow lines at junctions)
  4. Restore foreground objects from ORIGINAL on top of texture
  5. Keep everything outside wall pixel-identical to ORIGINAL

You are NOT allowed to:
  ✗ Change the number of {unit_label}s (count them — keep the count)
  ✗ Change the width or height of any {unit_label}
  ✗ Re-draw or re-generate the texture pattern
  ✗ Change the {joint_label} spacing or width
  ✗ Zoom, crop, pan, reframe, or change dimensions

═══ FOREGROUND OBJECTS (restore from ORIGINAL) ═══

{occluder_list}
These appear IN FRONT of the texture — copy from ORIGINAL onto the textured wall.

═══ PRODUCT DIMENSIONS ═══

{dimension_instructions}

═══ SURFACE PHYSICS ═══

{surface_rendering}
Relief: {bump_depth}mm → micro-shadows on {shadow_edge}
{joint_label}s recessed → appear in shadow, darker than faces

═══ LIGHTING (copy from ORIGINAL) ═══

  • Primary: {lighting_primary} (~{lighting_kelvin}K {lighting_temperature})
  • Gradient: {lighting_gradient}
  • Shadows: {shadow_type}, intensity {shadow_intensity}
  • AO: {ambient_occlusion}
  • Color cast: {color_cast}
  • {temperature_effect}

═══ PERSPECTIVE ═══

{perspective_instructions}

═══ FINAL VERIFICATION ═══

  ☐ COUNT {unit_label}s in output — SAME number as COMPOSITE?
  ☐ Each {unit_label} SAME width as in COMPOSITE?
  ☐ Texture on wall ONLY — not ceiling/floor/other walls?
  ☐ ALL foreground objects from ORIGINAL preserved?
  ☐ Brightness gradient matches ORIGINAL?
  ☐ Same resolution/aspect — NO crop/zoom/pan?

Output ONLY the final image. No text.
"""


def _build_occluder_list(analysis: dict | None) -> str:
    """Format the occluder list for the render prompt."""
    if not analysis:
        return "   (Analyze the ORIGINAL image for any objects in front of the wall)"

    occluders = analysis.get("occluders", [])
    if not occluders:
        extra = analysis.get("extra_exclusions", [])
        if extra:
            occluders = extra

    if not occluders:
        return "   (No major foreground objects detected — but double-check the ORIGINAL)"

    lines = []
    for occ in occluders:
        label = occ.get("label", "object")
        depth = occ.get("depth", "in front of wall")
        lines.append(f"   - {label} ({depth})")
    return "\n".join(lines)


def _build_perspective_instructions(analysis: dict | None) -> str:
    """Format perspective section for the render prompt."""
    if not analysis:
        return "Match the perspective visible in the ORIGINAL photo exactly."

    persp = analysis.get("perspective", {})
    ptype = persp.get("type", "frontal")
    angle = persp.get("angleDeg", 0)
    recedes = persp.get("recedes", "none")
    h_conv = persp.get("horizontalConvergence", "none")

    if ptype == "frontal" or angle < 10:
        return (
            f"Wall is viewed nearly head-on (~{angle}°). "
            f"Mortar/gap lines must be straight and parallel — "
            f"horizontals truly horizontal, verticals truly vertical. "
            f"Minimal perspective distortion."
        )
    else:
        return (
            f"Wall is viewed at ~{angle}° angle, receding to the {recedes}. "
            f"Mortar/gap lines converge toward the vanishing point ({h_conv}). "
            f"Far-side elements appear narrower due to foreshortening. "
            f"Follow the SAME convergence visible in the ORIGINAL."
        )


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
    """Stage 2: Generate photorealistic render with 5 reference images."""
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    meta = product_meta or {}
    layout = meta.get("layoutType", "running-bond")
    category = meta.get("category", "brick")
    is_brick = layout in ("running-bond", "stretcher-bond")
    is_stone = layout == "random-stone"
    is_wood = category == "wood"

    unit_label = "stone" if is_stone else "panel" if is_wood else "brick"
    joint_label = "mortar" if (is_brick or is_stone) else "gap"
    course_label = "stone rows" if is_stone else "panel rows" if is_wood else "brick courses"

    roughness = float(meta.get("roughness", 0.7))
    bump_depth = float(meta.get("bumpDepthMm", 5))
    specular = float(meta.get("specularIntensity", 0.05))
    surface_desc = meta.get("surfaceDescription", "matte surface with subtle texture")

    if roughness > 0.7:
        surface_rendering = (
            f"ROUGH MATTE material ({roughness*100:.0f}% roughness). "
            f"ZERO specular highlights, ZERO shine. "
            f"Surface: {surface_desc}"
        )
    elif roughness > 0.5:
        surface_rendering = (
            f"Near-matte surface ({roughness*100:.0f}% roughness). "
            f"Extremely subtle light variation, but NO specular spots. "
            f"Surface: {surface_desc}"
        )
    else:
        surface_rendering = (
            f"Subtle satin sheen ({roughness*100:.0f}% roughness, {specular*100:.0f}% specular). "
            f"Soft diffuse reflection visible only at oblique angles. "
            f"Surface: {surface_desc}"
        )

    shadow_edge = "bottom and right edges" if is_brick or is_stone else "gap edges"

    # Extract lighting from analysis
    a = analysis or {}
    lt = a.get("lighting", {})
    lighting_primary = lt.get("primarySource", str(a.get("lighting_direction", "diffuse")))
    lighting_temp = lt.get("temperature", str(a.get("lighting_temperature", "neutral")))
    lighting_kelvin = lt.get("temperatureKelvin", 5000)
    lighting_gradient = lt.get("gradient", "even")
    shadow_type = lt.get("shadowType", str(a.get("shadow_intensity", "soft-diffuse")))
    shadow_intensity = lt.get("shadowIntensity", 0.3)
    ambient_occlusion = lt.get("ambientOcclusion", "subtle at junctions")
    color_cast = lt.get("colorCast", "none")
    exposure = lt.get("exposure", "correct")

    if lighting_temp in ("warm", "neutral-warm"):
        temperature_effect = "Warm light tints the wall surface slightly orange/yellow — apply subtly"
    elif lighting_temp in ("cool", "neutral-cool"):
        temperature_effect = "Cool light tints the wall slightly blue — apply subtly"
    else:
        temperature_effect = "Neutral light — minimal color shift on the wall surface"

    dim_instructions = _build_dimension_instructions(meta, material_type, analysis)
    occluder_list = _build_occluder_list(analysis)
    perspective_instructions = _build_perspective_instructions(analysis)

    prompt = _RENDER_PROMPT_TEMPLATE.format(
        dimension_instructions=dim_instructions,
        occluder_list=occluder_list,
        perspective_instructions=perspective_instructions,
        unit_label=unit_label,
        joint_label=joint_label,
        course_label=course_label,
        lighting_primary=lighting_primary,
        lighting_temperature=lighting_temp,
        lighting_kelvin=lighting_kelvin,
        lighting_gradient=lighting_gradient,
        shadow_type=shadow_type,
        shadow_intensity=shadow_intensity,
        ambient_occlusion=ambient_occlusion,
        color_cast=color_cast,
        exposure=exposure,
        temperature_effect=temperature_effect,
        surface_rendering=surface_rendering,
        bump_depth=bump_depth,
        shadow_edge=shadow_edge,
        product_name=str(product_name or "product"),
        material_type=str(material_type),
    )

    # Send 5 images: original, composite, mask, texture tile, original again
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
    # Add original again as reference #5 for comparison
    parts.append(
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
    )

    logger.info("Gemini photorealistic render — model: %s, images: %d", model_name, len(parts) - 1)
    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.1,
            ),
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Verification & Correction
# Compare render with original to fix occlusion and lighting
# ══════════════════════════════════════════════════════════════════════════════

_VERIFICATION_PROMPT = """\
You are a photorealistic rendering QA engine. Compare the RENDERED image \
with the ORIGINAL photograph and fix issues. This is CORRECTION, not re-creation.

YOU RECEIVE FOUR IMAGES:
  1. ORIGINAL — unmodified room photograph (ground truth)
  2. RENDERED — AI visualization to be corrected
  3. MASK OVERLAY — ORANGE highlight = target wall area
  4. PRODUCT TEXTURE TILE — the material being applied

╔══════════════════════════════════════════════════════════════╗
║  PATTERN LOCK: Do NOT change the texture pattern.           ║
║  Keep the SAME number, size, and spacing of elements.       ║
║  Do NOT make bricks/panels wider, narrower, or fewer.       ║
╚══════════════════════════════════════════════════════════════╝

═══ CHECK AND FIX ═══

1. FOREGROUND OBJECTS: Is any furniture/frame/switch covered by texture?
   → Restore it from ORIGINAL. Texture goes BEHIND objects.

2. BOUNDARIES: Has texture leaked onto ceiling/floor/adjacent wall?
   → Remove it, show original surface.

3. LIGHTING: Does brightness gradient match ORIGINAL?
   → Fix mismatches. Add contact shadows where furniture meets wall.

4. TEXTURE COUNT: Does the number of bricks/panels/stones match RENDERED?
   → Do NOT change the count. Only fix colour/lighting issues.

5. EDGES: Smooth transitions at wall junctions?
   → Thin shadow lines, no harsh cuts.

═══ RULES ═══
  • SAME resolution/aspect — NO crop/zoom/pan
  • Everything outside wall = pixel-identical to ORIGINAL
  • Do NOT change texture scale, count, or pattern
  • If no issues found, return RENDERED unchanged

Return ONLY the corrected image. No text.
"""


def verify_and_correct_render(
    original: Image.Image,
    rendered: Image.Image,
    mask_overlay: Image.Image,
    product_texture: Image.Image | None = None,
) -> Image.Image | None:
    """Stage 3: Compare render with original, fix occlusion & lighting issues."""
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    parts: list = [
        types.Part.from_text(text=_VERIFICATION_PROMPT),
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(rendered), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(mask_overlay), mime_type="image/jpeg"),
    ]
    if product_texture:
        parts.append(
            types.Part.from_bytes(
                data=_pil_to_bytes(product_texture), mime_type="image/jpeg",
            )
        )

    logger.info("Gemini verification pass — model: %s", model_name)
    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.1,
            ),
        )
    except Exception as exc:
        logger.warning("Gemini verification failed (%s); skipping", exc)
        return None

    if response is None:
        return None

    img = _extract_first_image_from_response(response)
    if img is not None:
        return img

    tx = _response_text(response)
    if tx:
        logger.warning("Gemini verification returned text: %s", tx[:300])
    return None


# ── Legacy single-call refine (kept for /api/render-refine) ────────────────


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
