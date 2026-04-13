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
You are a forensic architectural measurement system with expertise in \
construction, interior design, and photogrammetry. Your task: determine \
the EXACT real-world dimensions of a wall area and catalog every element \
in the scene for photorealistic material rendering.

You receive THREE images:
  1. ORIGINAL — unmodified room/building photograph
  2. COMPOSITE — same photo with decorative texture tiled on the wall \
(use this to verify texture scale correctness)
  3. MASK OVERLAY — ORIGINAL with ORANGE semi-transparent highlight \
showing the user's wall selection area

════════════════════════════════════════════════════════════════
 TASK A: MULTI-REFERENCE DIMENSIONAL CALIBRATION
 You MUST use AT LEAST 3 independent reference objects to triangulate.
════════════════════════════════════════════════════════════════

Scan the ENTIRE image systematically. Identify EVERY object with known \
real-world dimensions from ANY of these categories:

─── CATEGORY 1: DOORS (most reliable, ±3%) ───
  • Standard interior door leaf height: 200 cm (ALWAYS — this is building code)
  • Standard interior door leaf width: 70–90 cm (most common: 80 cm)
  • Door frame/architrave adds ~7 cm each side and ~6 cm top
  • Total opening with frame: ~213 cm × ~94 cm
  • French/balcony door: 200–220 cm × 80–180 cm
  • Sliding wardrobe door panel: 230–250 cm × 60–90 cm
  • Front/exterior door: 200–210 cm × 90–100 cm
  • A door handle is ALWAYS at 100–110 cm from the floor

─── CATEGORY 2: WINDOWS (±5%) ───
  • Standard window: 120–150 cm tall × 80–120 cm wide
  • Window sill height from floor: 85–95 cm (interior), 90–100 cm (exterior)
  • Curtain rod to ceiling gap: 10–20 cm
  • Roller blind cassette height: 7–10 cm
  • Standard windowpane in residential: 50–70 cm × 40–60 cm per section

─── CATEGORY 3: ARCHITECTURAL FIXED ELEMENTS (±3%) ───
  • EU ceiling height: 250–280 cm (most common in Poland/EU: 260–270 cm)
  • US ceiling height: 244 cm (8 ft) or 274 cm (9 ft)
  • Baseboard/skirting board: 8–15 cm tall (most common: 10 cm)
  • Crown molding: 5–10 cm
  • Stair riser height: 17–19 cm; tread depth: 25–30 cm
  • Standard facing brick (if visible): 6.5 cm × 25 cm × 12 cm
  • Concrete block: 19 cm × 39 cm
  • Radiator height: 30 cm (low), 50–60 cm (standard), 90 cm (tall)
  • Radiator clearance from floor: 10–15 cm
  • Kitchen backsplash tile (standard): 10 × 10 cm or 15 × 15 cm
  • Floor tile (standard): 30 × 30 cm, 45 × 45 cm, or 60 × 60 cm

─── CATEGORY 4: ELECTRICAL ELEMENTS (±5%) ───
  • Light switch plate: 8 × 8 cm, center at 105–120 cm from floor
  • Double switch plate: 8 × 15 cm, same height
  • Power outlet plate: 8 × 8 cm, center at 25–30 cm from floor
  • Kitchen counter outlet: center at 100–115 cm from floor
  • Thermostat: 8 × 12 cm, center ~150 cm from floor
  • Light switch to door frame gap: typically 10–15 cm

─── CATEGORY 5: FURNITURE (±8%) ───
  • Dining table height: 73–76 cm
  • Kitchen counter/island top: 85–90 cm
  • Bar counter height: 100–115 cm
  • Standard dining chair seat: 43–47 cm from floor
  • Chair backrest top: 85–100 cm from floor
  • Sofa seat height: 40–45 cm; backrest height: 80–95 cm total
  • Coffee table: 40–50 cm tall
  • TV console/sideboard: 40–55 cm tall
  • Bookshelf (full): 180–210 cm; bookshelf (low): 70–85 cm
  • Desk height: 72–76 cm
  • Nightstand/bedside table: 55–65 cm
  • Bed headboard top: 90–130 cm from floor
  • Kitchen upper cabinet bottom edge: 140–150 cm from floor
  • Kitchen upper cabinet top: 210–230 cm from floor

─── CATEGORY 6: APPLIANCES & OBJECTS (±3–5%) ───
  • TV diagonal → 55": ~68 × 122 cm;  65": ~81 × 144 cm;  75": ~93 × 166 cm
  • Refrigerator: 170–190 cm tall × 60–70 cm wide
  • Washing machine/dishwasher: 85 cm tall × 60 cm wide
  • Standard microwave: 28–30 cm tall × 50–55 cm wide
  • Wall clock diameter: 25–40 cm
  • A4 picture frame: 21 × 30 cm (with mat/frame: ~30 × 40 cm)
  • Wine bottle standing: 30 cm tall
  • Standard dinner plate: 27 cm diameter
  • Floor lamp shade center: 150–170 cm from floor

─── CATEGORY 7: HUMAN PROPORTIONS (if visible, ±5%) ───
  • Adult standing height: 165–180 cm
  • Adult shoulder height: 140–155 cm
  • Adult elbow height: 100–110 cm
  • Seated adult head height: 120–130 cm from floor

═══ MANDATORY MEASUREMENT PROCEDURE ═══

Execute ALL of these steps IN ORDER — do not skip any:

STEP 1 — SYSTEMATIC SCAN: Examine every region of the image (top-left, \
top-center, top-right, middle-left, … bottom-right). List every reference \
object you spot, no matter how small.

STEP 2 — PIXEL MEASUREMENT: For the 3–5 best reference objects, measure \
their pixel span (height or width) as precisely as possible in the image.

STEP 3 — INDEPENDENT px/cm CALCULATION: For EACH reference, compute:
   px_per_cm = pixel_span / known_real_cm
Record all values. They should agree within ±15%.

STEP 4 — CROSS-VALIDATION:
   • If a door is visible (200 cm): the room height from floor to ceiling \
should be between 240–280 cm. If your measurement says 400 cm, something \
is wrong — re-check.
   • If a light switch is visible (115 cm from floor): it should be at \
roughly 43% of the way up a 265 cm wall.
   • If a baseboard is visible (10 cm): it should be about 3.7% of 270 cm.
   • If furniture shows a table at 75 cm: it should be ~28% of a 270 cm wall.
   • Use at least TWO of these sanity checks.

STEP 5 — TAKE THE MEDIAN px/cm value, discarding outliers that deviate \
more than 20% from the median.

STEP 6 — APPLY to the orange-highlighted wall area:
   wall_height_cm = wall_pixel_height / calibrated_px_per_cm
   wall_width_cm = wall_pixel_width / calibrated_px_per_cm

STEP 7 — FINAL SANITY CHECK:
   • Does the wall height make sense? (Indoor: 200–300 cm typical)
   • Could a person (170 cm) stand against it proportionally?
   • Would a standard door (200 cm) fit?
   • Is the width reasonable for the scene? (Room widths: 200–800 cm)

════════════════════════════════════════════════════════════════
 TASK B: COMPLETE OCCLUSION CATALOG
 Every object between the camera and the wall surface
════════════════════════════════════════════════════════════════

Examine the orange-highlighted wall area in image 3. List EVERY object \
that is IN FRONT OF the wall surface or ATTACHED to it. These objects \
must NOT be covered by the decorative texture. Include:

  • Furniture touching/near the wall (shelves, TV, cabinets, frames, mirrors)
  • Architectural elements crossing the wall (columns, beams, window frames, \
door frames, reveals, molding)
  • Electrical fixtures (switches, outlets, thermostats, intercom panels)
  • Decorative items (clocks, sconces, wall art, pendant lights, plants)
  • Structural elements (exposed pipes, vents, radiators, AC units)
  • People or animals in front of the wall
  • ANY object — even partial — that overlaps the orange area

For EACH occluder provide its bounding box as normalized 0.0–1.0 \
coordinates relative to the full image dimensions: {"x", "y", "w", "h"}.

════════════════════════════════════════════════════════════════
 TASK C: PRECISE WALL BOUNDARY DETECTION
 Where does the actual wall surface begin and end?
════════════════════════════════════════════════════════════════

The orange mask is the USER's rough rectangle selection. It may extend \
BEYOND the actual wall onto the ceiling, floor, or adjacent walls. \
You MUST identify the exact boundaries of the physical wall surface:

  • CEILING LINE — the exact y-coordinate (normalized 0.0–1.0) where the \
wall meets the ceiling. Look for: color change, shadow line, crown \
molding lower edge, or where paint changes.
  • FLOOR LINE — the exact y-coordinate where the wall meets the floor \
or the TOP of the baseboard/skirting board. The texture should NOT cover \
the baseboard — it goes to its top edge only.
  • LEFT BOUNDARY — where the wall ends on the left (corner, door frame, \
window frame, or image edge).
  • RIGHT BOUNDARY — same for the right side.

Check each boundary: does the orange selection extend past it?
Flag any overrun so the system can clip the mask.

════════════════════════════════════════════════════════════════
 TASK D: COMPREHENSIVE LIGHTING ANALYSIS
════════════════════════════════════════════════════════════════

Analyze ALL light sources affecting the wall:

  • Primary source: direction, type (natural/artificial), angle
  • Secondary/fill sources (ceiling fixtures, table lamps, reflected light)
  • Color temperature: warm (2700–3500K), neutral (4000–5000K), cool (5500K+)
  • Brightness gradient across the wall (which side/region is brightest?)
  • Shadow characteristics: hard edges (direct sun) vs soft (diffuse/overcast)
  • Ambient occlusion: darkness in ceiling–wall junction, floor–wall junction, \
corners, and behind furniture
  • Color cast from nearby colored surfaces (warm wood floor, colored wall)
  • Overall exposure: underexposed / correct / slightly overexposed
  • Specular reflections visible on wall surface?

════════════════════════════════════════════════════════════════
 TASK E: PERSPECTIVE ANALYSIS
════════════════════════════════════════════════════════════════

  • Camera angle to the wall surface: frontal (0–10°), moderate (10–30°), \
oblique (30°+)
  • If angled: which side recedes? (left or right)
  • Vanishing point direction for horizontal mortar/grout lines
  • Vertical convergence (camera tilted up/down)?
  • Lens distortion: barrel / pincushion / none?
  • Is the camera at eye level (~155 cm), low, or high?

════════════════════════════════════════════════════════════════
 TASK F: TEXTURE SCALE VALIDATION (check COMPOSITE image)
════════════════════════════════════════════════════════════════

Look at image 2 (COMPOSITE). The texture has been tiled at a computed \
scale. Evaluate:
  • Do the texture elements appear the correct real-world size compared \
to the reference objects you identified?
  • Count the number of horizontal courses (rows) of texture visible.
  • Given the wall height and the texture module height, is the count correct?
  • Are textures too large? Too small? Just right?

═══ OUTPUT FORMAT ═══

Output a single JSON object. No markdown, no backticks, no commentary. \
Start directly with the opening brace:

{
  "wallHeightCm": 255,
  "wallWidthCm": 340,
  "ceilingHeightCm": 265,
  "measurementMethod": "door frame primary (200cm=520px → 2.6 px/cm), baseboard confirmed (10cm=26px → 2.6 px/cm), chair seat cross-check (45cm=117px → 2.6 px/cm). Median px/cm = 2.6",
  "referenceObjects": [
    {"name": "interior door", "pixelHeight": 520, "realHeightCm": 200, "pxPerCm": 2.6, "confidence": "high"},
    {"name": "baseboard", "pixelHeight": 26, "realHeightCm": 10, "pxPerCm": 2.6, "confidence": "medium"},
    {"name": "dining chair seat", "pixelHeight": 117, "realHeightCm": 45, "pxPerCm": 2.6, "confidence": "medium"}
  ],
  "calibratedPxPerCm": 2.6,
  "occluders": [
    {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.65, "label": "bookshelf", "depth": "touching_wall"},
    {"x": 0.7, "y": 0.5, "w": 0.08, "h": 0.04, "label": "light_switch", "depth": "on_wall"},
    {"x": 0.3, "y": 0.8, "w": 0.4, "h": 0.2, "label": "sofa_top", "depth": "30cm_from_wall"}
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
    "primarySource": "window left, natural daylight, 45° angle",
    "secondarySource": "ceiling fixture, warm LED 3000K",
    "temperature": "neutral-warm",
    "temperatureKelvin": 4500,
    "gradient": "brighter-left-darker-right",
    "gradientIntensity": 0.3,
    "shadowType": "soft-diffuse",
    "shadowIntensity": 0.4,
    "ambientOcclusion": "visible at ceiling junction and left corner, darker at baseboard",
    "colorCast": "slight warm cast from wooden floor",
    "exposure": "correct",
    "specularOnWall": false
  },
  "perspective": {
    "type": "frontal",
    "angleDeg": 5,
    "recedes": "none",
    "horizontalConvergence": "negligible",
    "verticalConvergence": "none",
    "lensDistortion": "none",
    "cameraHeight": "eye-level"
  },
  "textureScaleCorrect": true,
  "scaleNote": "bricks correctly sized — 28 courses counted, expected 28 for 255cm wall with 9cm course height",
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

    # Additional cross-checks for the prompt
    courses_in_door = 2000 / course_h_mm  # standard 200cm door
    courses_in_meter = 1000 / course_h_mm

    if layout in ("running-bond", "stretcher-bond"):
        return (
            f"PRODUCT: Decorative brick/stone cladding\n"
            f"  • Each brick module: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(= {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Mortar joint: {joint:.0f} mm (= {joint/10:.1f} cm)\n"
            f"  • Course height (brick + joint): {course_h_mm:.0f} mm "
            f"(= {course_h_mm/10:.1f} cm)\n"
            f"  • Unit width (brick + joint): {unit_w_mm:.0f} mm\n"
            f"\n"
            f"WALL DIMENSIONS (calibrated using: {refs_text}):\n"
            f"  • Wall height: ~{wall_h_cm:.0f} cm → MUST contain ~{courses_in_wall:.0f} courses (rows of bricks)\n"
            f"  • Wall width: ~{wall_w_cm:.0f} cm → MUST contain ~{units_in_row:.0f} bricks per row\n"
            f"\n"
            f"CRITICAL SCALE CROSS-CHECKS:\n"
            f"  • A standard door (200 cm tall) should span exactly {courses_in_door:.0f} brick courses alongside it\n"
            f"  • 1 meter of wall height = exactly {courses_in_meter:.1f} courses\n"
            f"  • A light switch (115 cm from floor) should be at course ~{1150/course_h_mm:.0f} from floor\n"
            f"  • An adult person (170 cm) standing against the wall = ~{1700/course_h_mm:.0f} courses tall\n"
            f"  • A dining table (75 cm tall) against the wall = ~{750/course_h_mm:.0f} courses tall\n"
            f"\n"
            f"  ⚠️ If the number of courses in the COMPOSITE differs from ~{courses_in_wall:.0f}, the scale is WRONG.\n"
            f"     The COMPOSITE texture has been pre-computed at the correct scale — TRUST ITS COUNT.\n"
            f"  ⚠️ All bricks MUST appear this exact physical size. Only natural\n"
            f"     perspective foreshortening is allowed — bricks farther from camera appear smaller."
        )
    elif layout in ("stack-bond", "vertical-stack"):
        return (
            f"PRODUCT: Decorative panels / lamels (vertically-oriented)\n"
            f"  • Each panel: {mod_w:.0f} mm wide × {mod_h:.0f} mm tall "
            f"(= {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Gap between panels: {joint:.0f} mm (= {joint/10:.1f} cm)\n"
            f"  • Panel + gap width: {unit_w_mm:.0f} mm (= {unit_w_mm/10:.1f} cm)\n"
            f"\n"
            f"WALL DIMENSIONS (calibrated using: {refs_text}):\n"
            f"  • Wall height: ~{wall_h_cm:.0f} cm\n"
            f"  • Wall width: ~{wall_w_cm:.0f} cm → should fit ~{units_in_row:.0f} panels across\n"
            f"\n"
            f"CRITICAL SCALE CROSS-CHECKS:\n"
            f"  • A standard door (200 cm) should have {2000/unit_w_mm:.0f} panels across its width if adjacent\n"
            f"  • 1 meter of wall width = {1000/unit_w_mm:.1f} panels\n"
            f"  • The spacing between panels MUST be exactly {joint:.0f} mm "
            f"(~{joint/10:.1f} cm). Panels must be evenly spaced.\n"
            f"\n"
            f"  ⚠️ The COMPOSITE has the correct number of panels — preserve that count EXACTLY."
        )
    elif layout == "random-stone":
        return (
            f"PRODUCT: Decorative random stone cladding\n"
            f"  • Average stone module: {mod_w:.0f} × {mod_h:.0f} mm "
            f"(= {mod_w/10:.1f} × {mod_h/10:.1f} cm)\n"
            f"  • Mortar joint: {joint:.0f} mm (= {joint/10:.1f} cm)\n"
            f"  • Average course height: {course_h_mm:.0f} mm\n"
            f"\n"
            f"WALL DIMENSIONS (calibrated using: {refs_text}):\n"
            f"  • Wall height: ~{wall_h_cm:.0f} cm → ~{courses_in_wall:.0f} stone rows\n"
            f"  • Wall width: ~{wall_w_cm:.0f} cm → ~{units_in_row:.0f} stones per row\n"
            f"\n"
            f"  ⚠️ Stone pattern is IRREGULAR but the AVERAGE size must match.\n"
            f"     The COMPOSITE has the correct stone count — preserve it EXACTLY."
        )
    else:
        return (
            f"PRODUCT: Decorative {material_type}\n"
            f"  • Module: {mod_w:.0f} × {mod_h:.0f} mm, joint: {joint:.0f} mm\n"
            f"  • Course height: {course_h_mm:.0f} mm\n"
            f"\n"
            f"WALL: ~{wall_h_cm:.0f} × {wall_w_cm:.0f} cm "
            f"(calibrated using: {refs_text})\n"
            f"  • Expected ~{courses_in_wall:.0f} courses × ~{units_in_row:.0f} modules per row\n"
            f"  ⚠️ The COMPOSITE has the correct module count — preserve it EXACTLY."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Photorealistic Render
# ══════════════════════════════════════════════════════════════════════════════

_RENDER_PROMPT_TEMPLATE = """\
You are a professional photo compositor. You receive 3 images:

IMAGE 1 — "BEZ AI" (COMPOSITE): The original room photo with "{product_name}" \
({material_type}) texture provisionally laid onto one wall. This defines \
the EXACT SIZE, SCALE, POSITION, and TILING of the texture. The texture \
dimensions in this image are FINAL — do not change them at all.

IMAGE 2 — ORIGINAL: The unmodified room photograph. This is your reference \
for lighting, shadows, color temperature, and all objects/surfaces.

IMAGE 3 — PRODUCT TEXTURE: A close-up tile/swatch of "{product_name}". \
This defines the EXACT APPEARANCE of the texture — colors, grain, surface \
detail, 3D relief, material look. The texture on the wall must look \
IDENTICAL to this swatch.

TWO CRITICAL 1:1 RULES:
  ★ APPEARANCE 1:1 with IMAGE 3 — The texture on the wall must reproduce \
the exact same colors, grain pattern, surface detail, and material look \
as shown in IMAGE 3 (product texture). Every brick, slat, stone, or \
plank must look exactly like the swatch.
  ★ SIZE 1:1 with IMAGE 1 — The texture scale, tile count, element \
spacing, and overall dimensions must be EXACTLY the same as in IMAGE 1 \
(the "Bez AI" composite). Count the elements (bricks/slats/planks) in \
IMAGE 1 — your output must have the SAME count in the SAME positions.

YOUR TASK: Create a photorealistic photograph where the texture looks \
physically installed on the wall.

STEP 1 — ANALYZE (look at IMAGE 2):
  • Lighting: direction, intensity, color temperature, shadows
  • Objects IN FRONT of the wall: furniture, TV, shelves, plants, lamps, \
frames, switches, cables, curtains, people
  • Wall boundaries: ceiling, floor/baseboard, corners, door/window frames

STEP 2 — COMPOSE THE TEXTURE:
  • Use IMAGE 1 as your base — keep the texture exactly where it is, \
at exactly the same size
  • Make the texture look like IMAGE 3 — same colors, same grain, \
same surface detail, same material quality
  • Apply the room's lighting from IMAGE 2 SUBTLY — only slight \
brightness/shadow adjustments, NOT color changes
  • Add natural contact shadows where furniture meets the wall
  • Add ambient occlusion at ceiling/floor/corner junctions

STEP 3 — RESTORE FOREGROUND:
  • Every object IN FRONT of the wall in IMAGE 2 must appear ON TOP \
of the texture — copy them pixel-perfectly from IMAGE 2
  • Texture must NOT cover ceiling, floor, adjacent walls, baseboards, \
crown molding, door/window frames
  • Everything outside the textured area = IDENTICAL to IMAGE 2

ABSOLUTE RULES:
  ⛔ Do NOT change the texture SIZE — it must match IMAGE 1 exactly
  ⛔ Do NOT change the texture LOOK — it must match IMAGE 3 exactly
  ⛔ Do NOT re-tile, re-scale, re-draw, or re-imagine the texture
  ⛔ Do NOT change resolution, framing, or aspect ratio
  ⛔ Do NOT crop or zoom

Output ONLY the final image. No text.
"""


def generate_photorealistic_render(
    original: Image.Image,
    composite: Image.Image,
    mask_overlay: Image.Image | None = None,
    product_name: str = "product",
    product_texture: Image.Image | None = None,
    analysis: dict | None = None,
    material_type: str = "decorative stone/brick",
    product_meta: dict | None = None,
) -> Image.Image | None:
    """Generate photorealistic render from 3 images.

    Sends 3 images to Gemini (in this order):
      1. COMPOSITE ("Bez AI") — texture provisionally laid on the wall
      2. ORIGINAL — unmodified room photo (ground truth)
      3. PRODUCT TEXTURE — close-up tile/swatch of the selected product

    The AI analyzes all 3 images and produces a photorealistic blend.
    No mask needed — the AI sees the difference between composite and original.
    """
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    meta = product_meta or {}

    prompt = _RENDER_PROMPT_TEMPLATE.format(
        product_name=str(product_name or "product"),
        material_type=str(material_type),
    )

    # Send 3 images: composite ("bez ai"), original, product texture
    parts: list = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=_pil_to_bytes(composite), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
    ]
    if product_texture:
        parts.append(
            types.Part.from_bytes(data=_pil_to_bytes(product_texture), mime_type="image/jpeg")
        )

    img_count = len(parts) - 1  # subtract text part
    logger.info("Gemini photorealistic render — model: %s, images: %d", model_name, img_count)
    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.2,
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
# Compare render with original to fix occlusion, boundaries, and lighting
# ══════════════════════════════════════════════════════════════════════════════

_VERIFICATION_PROMPT = """\
You are a photo QA engineer. Compare the RENDERED image with the ORIGINAL \
photograph and fix any remaining issues. This is surgical correction only.

IMAGE 1 — ORIGINAL: The unmodified room photo (ground truth)
IMAGE 2 — RENDERED: The AI visualization to check and correct
IMAGE 3 — MASK: ORIGINAL with ORANGE overlay showing the wall area

FIX THESE ISSUES IF FOUND:

1. COVERED OBJECTS — If any object visible in the ORIGINAL (furniture, TV, \
frames, plants, lamps, switches, people) is now hidden behind the texture, \
RESTORE it from the ORIGINAL. Objects go IN FRONT of the texture.

2. BOUNDARY LEAKS — If texture appears on the ceiling, floor, adjacent \
walls, baseboards, or door/window frames, REMOVE it and show the ORIGINAL \
surface instead. Texture covers ONLY the wall.

3. LIGHTING — If the wall brightness or color doesn't match the ORIGINAL, \
adjust it. The textured wall should have the same lighting gradient and \
color temperature as the ORIGINAL.

4. EDGES — Transitions at wall boundaries should look natural with thin \
shadow lines at ceiling/floor junctions.

RULES:
• Do NOT change the texture pattern, scale, count, or spacing
• Same resolution and aspect ratio — no crop/zoom/pan
• Everything outside the wall = ORIGINAL unchanged
• If no issues found, return the RENDERED image unchanged

Output ONLY the corrected image. No text.
"""


def verify_and_correct_render(
    original: Image.Image,
    rendered: Image.Image,
    mask_overlay: Image.Image,
    product_texture: Image.Image | None = None,
) -> Image.Image | None:
    """Stage 3: Compare render with original, fix remaining issues."""
    from google.genai import types

    model_name = image_model_name()
    client = _gemini_client()

    parts: list = [
        types.Part.from_text(text=_VERIFICATION_PROMPT),
        types.Part.from_bytes(data=_pil_to_bytes(original), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(rendered), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_pil_to_bytes(mask_overlay), mime_type="image/jpeg"),
    ]

    logger.info("Gemini verification pass — model: %s", model_name)
    response = None
    try:
        response = _retry_generate(
            client, model=model_name, contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.2,
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
