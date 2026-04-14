"""Scale Calibration — estimate real-world px/cm ratio from scene references.

Uses SegFormer ADE20K semantic segmentation to detect many architectural and
furniture reference elements and compute a highly accurate px-per-cm conversion.

Multi-reference cross-validation: detects doors, windows, ceiling-floor span,
furniture, and spatial relationships between elements to triangulate the true
room scale with minimal deviation from reality.

Standard European interior dimensions used as ground truth:
- Door height: 200cm, width: 80-90cm
- Window height: 120-150cm, sill at 85-100cm from floor
- Ceiling height: 260-280cm (standard 270cm)
- Sofa/couch height: 80-90cm
- Table/desk height: 72-78cm
- Chair seat height: 42-48cm, back height: 80-100cm
- Bed height: 50-60cm
- Cabinet height: 200-220cm (tall), 85cm (counter)
- Radiator height: 55-65cm, bottom 10-15cm from floor
- Person standing: 165-180cm
- Light switch: 110-120cm from floor
- Electrical outlet: 25-35cm from floor
- Baseboard/skirting: 8-12cm
- Painting/picture center: 145-165cm from floor
- Curtain rod: 255-270cm from floor
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K 150-class indices for reference elements
ADE20K = {
    "wall": 0, "building": 1, "sky": 2, "floor": 3, "tree": 4,
    "ceiling": 5, "road": 6, "bed": 7, "window": 8, "grass": 9,
    "cabinet": 10, "sidewalk": 11, "person": 12, "earth": 13,
    "door": 14, "table": 15, "mountain": 16, "plant": 17,
    "curtain": 18, "chair": 19, "car": 20, "water": 21,
    "painting": 22, "sofa": 23, "shelf": 24, "house": 25,
    "sea": 26, "mirror": 27, "rug": 28, "field": 29,
    "armchair": 30, "seat": 31, "fence": 32, "desk": 33,
    "rock": 34, "wardrobe": 35, "lamp": 36, "bathtub": 37,
    "railing": 38, "cushion": 39, "base": 40, "column": 41,
    "signboard": 42, "chest": 43, "counter": 44, "sand": 45,
    "sink": 46, "skyscraper": 47, "fireplace": 48, "refrigerator": 49,
    "grandstand": 50, "path": 51, "stairs": 52, "runway": 53,
    "screen": 54, "stool": 55, "radiator": 56,
    "bookcase": 57, "booth": 58, "tv": 59, "computer": 60,
    "towel": 61, "toilet": 62, "flower": 63, "book": 64,
}


@dataclass
class ReferenceDetection:
    """A detected reference element with its measurements."""
    type: str           # element name
    height_px: int      # bounding box height in pixels
    width_px: int       # bounding box width in pixels
    x: int              # bbox top-left x
    y: int              # bbox top-left y
    y_bottom: int       # bbox bottom y (y + height)
    area_px: int        # contour area
    expected_cm: float  # expected real-world dimension (cm)
    dimension: str      # which dimension: "height" or "width"
    px_per_cm: float    # computed px/cm from this element
    confidence: float   # weight for averaging (0..1)


# Each reference: (ADE20K class, expected_height_cm, expected_width_cm, weight)
# weight = how reliable this reference is (1.0 = very reliable)
REFERENCE_SPECS = {
    # === ARCHITECTURAL (most reliable) ===
    "door": {
        "class": "door",
        "height_cm": 200, "width_cm": 85,
        "weight": 1.0,
        "min_aspect": 1.3,  # taller than wide
    },
    "window": {
        "class": "window",
        "height_cm": 140, "width_cm": 120,
        "weight": 0.75,
        "min_aspect": 0.5,
    },

    # === FURNITURE (moderately reliable) ===
    "sofa": {
        "class": "sofa",
        "height_cm": 85, "width_cm": None,
        "weight": 0.65,
        "min_aspect": 0.0,
    },
    "armchair": {
        "class": "armchair",
        "height_cm": 90, "width_cm": 80,
        "weight": 0.60,
        "min_aspect": 0.0,
    },
    "table": {
        "class": "table",
        "height_cm": 75, "width_cm": None,
        "weight": 0.55,
        "min_aspect": 0.0,
    },
    "desk": {
        "class": "desk",
        "height_cm": 75, "width_cm": None,
        "weight": 0.55,
        "min_aspect": 0.0,
    },
    "chair": {
        "class": "chair",
        "height_cm": 90, "width_cm": 45,
        "weight": 0.50,
        "min_aspect": 0.0,
    },
    "bed": {
        "class": "bed",
        "height_cm": 55, "width_cm": None,
        "weight": 0.50,
        "min_aspect": 0.0,
    },
    "cabinet": {
        "class": "cabinet",
        "height_cm": 200, "width_cm": None,
        "weight": 0.55,
        "min_aspect": 0.8,  # tall cabinets only
    },
    "wardrobe": {
        "class": "wardrobe",
        "height_cm": 210, "width_cm": None,
        "weight": 0.60,
        "min_aspect": 1.0,
    },
    "bookcase": {
        "class": "bookcase",
        "height_cm": 200, "width_cm": None,
        "weight": 0.55,
        "min_aspect": 0.8,
    },
    "refrigerator": {
        "class": "refrigerator",
        "height_cm": 180, "width_cm": 60,
        "weight": 0.70,
        "min_aspect": 1.5,
    },
    "radiator": {
        "class": "radiator",
        "height_cm": 60, "width_cm": None,
        "weight": 0.60,
        "min_aspect": 0.0,
    },
    "fireplace": {
        "class": "fireplace",
        "height_cm": 110, "width_cm": 120,
        "weight": 0.55,
        "min_aspect": 0.0,
    },

    # === OBJECTS WITH KNOWN SIZES ===
    "tv_screen": {
        "class": "tv",
        "height_cm": 55, "width_cm": 100,  # ~55" TV
        "weight": 0.45,
        "min_aspect": 0.0,
    },
    "mirror": {
        "class": "mirror",
        "height_cm": 100, "width_cm": None,
        "weight": 0.40,
        "min_aspect": 0.0,
    },
    "toilet": {
        "class": "toilet",
        "height_cm": 40, "width_cm": 38,
        "weight": 0.65,
        "min_aspect": 0.0,
    },
    "bathtub": {
        "class": "bathtub",
        "height_cm": 55, "width_cm": 170,
        "weight": 0.60,
        "min_aspect": 0.0,
    },
    "sink": {
        "class": "sink",
        "height_cm": 15, "width_cm": 50,
        "weight": 0.40,
        "min_aspect": 0.0,
    },
    "stool": {
        "class": "stool",
        "height_cm": 45, "width_cm": 35,
        "weight": 0.45,
        "min_aspect": 0.0,
    },
    "column": {
        "class": "column",
        "height_cm": 270, "width_cm": 30,
        "weight": 0.50,
        "min_aspect": 2.0,
    },
    "person": {
        "class": "person",
        "height_cm": 172, "width_cm": None,
        "weight": 0.55,
        "min_aspect": 1.3,
    },
}


def calibrate_scale(
    image: Image.Image,
    segmentation_argmax: np.ndarray,
    wall_mask: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> dict:
    """Estimate px_per_cm from all detectable architectural references.

    Uses multi-reference cross-validation: detects as many reference objects
    as possible, measures them, computes individual px/cm estimates, filters
    outliers, and returns a weighted average.

    Returns:
        dict with px_per_cm, wall_height_cm, confidence, references, method
    """
    h, w = segmentation_argmax.shape
    detections: list[ReferenceDetection] = []

    # ── 1. Detect individual reference objects ───────────────────────────
    for ref_name, spec in REFERENCE_SPECS.items():
        cls_name = spec["class"]
        cls_idx = ADE20K.get(cls_name)
        if cls_idx is None:
            continue

        class_mask = (segmentation_argmax == cls_idx).astype(np.uint8)
        if class_mask.sum() < 80:
            continue

        # Find ALL instances (contours)
        contours, _ = cv2.findContours(
            class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Measure each instance
        min_area = h * w * 0.001  # at least 0.1% of image
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = bh / max(bw, 1)

            # Filter by aspect ratio if specified
            if spec.get("min_aspect", 0) > 0 and aspect < spec["min_aspect"]:
                continue

            # Compute px_per_cm from height
            if spec["height_cm"] and bh > 10:
                px_per_cm_h = float(bh) / spec["height_cm"]
                detections.append(ReferenceDetection(
                    type=ref_name,
                    height_px=int(bh), width_px=int(bw),
                    x=int(bx), y=int(by), y_bottom=int(by + bh),
                    area_px=int(area),
                    expected_cm=spec["height_cm"],
                    dimension="height",
                    px_per_cm=px_per_cm_h,
                    confidence=spec["weight"],
                ))

            # Also from width if available
            if spec.get("width_cm") and bw > 10:
                px_per_cm_w = float(bw) / spec["width_cm"]
                detections.append(ReferenceDetection(
                    type=f"{ref_name}_width",
                    height_px=int(bh), width_px=int(bw),
                    x=int(bx), y=int(by), y_bottom=int(by + bh),
                    area_px=int(area),
                    expected_cm=spec["width_cm"],
                    dimension="width",
                    px_per_cm=px_per_cm_w,
                    confidence=spec["weight"] * 0.7,  # width less reliable
                ))

    # ── 2. Ceiling-to-floor measurement ──────────────────────────────────
    ceiling_mask = (segmentation_argmax == ADE20K["ceiling"]).astype(np.uint8)
    floor_mask = (segmentation_argmax == ADE20K["floor"]).astype(np.uint8)

    ceil_bottom = None
    floor_top = None

    if ceiling_mask.sum() > 50:
        ceil_rows = np.where(ceiling_mask.any(axis=1))[0]
        if len(ceil_rows) > 0:
            ceil_bottom = int(ceil_rows.max())

    if floor_mask.sum() > 50:
        floor_rows = np.where(floor_mask.any(axis=1))[0]
        if len(floor_rows) > 0:
            floor_top = int(floor_rows.min())

    if ceil_bottom is not None and floor_top is not None:
        room_height_px = floor_top - ceil_bottom
        if room_height_px > 50:
            px_per_cm = float(room_height_px) / 270.0
            detections.append(ReferenceDetection(
                type="ceiling_to_floor",
                height_px=room_height_px, width_px=w,
                x=0, y=ceil_bottom, y_bottom=floor_top,
                area_px=room_height_px * w,
                expected_cm=270.0,
                dimension="height",
                px_per_cm=px_per_cm,
                confidence=0.90,
            ))

    # ── 3. Spatial cross-validation ──────────────────────────────────────
    # If we detected a door AND ceiling-floor, verify door position:
    # door bottom should be near floor, door top at ~200cm from floor
    if floor_top is not None:
        for det in list(detections):
            if det.type == "door" and det.dimension == "height":
                # Door bottom should be near floor level
                door_bottom_offset = abs(det.y_bottom - floor_top)
                if door_bottom_offset > h * 0.15:
                    # Door bottom is far from floor — reduce confidence
                    det.confidence *= 0.5
                    logger.info(
                        "Door bottom (%d) far from floor (%d), reducing confidence",
                        det.y_bottom, floor_top,
                    )

    # ── 4. Wall-only height heuristic ────────────────────────────────────
    if wall_mask is not None:
        wall_rows = np.where(wall_mask.any(axis=1))[0]
        if len(wall_rows) > 10:
            wall_height_px = int(wall_rows.max() - wall_rows.min())
            # Wall visible height is typically 70-80% of room height
            px_per_cm = float(wall_height_px) / (270.0 * 0.75)
            detections.append(ReferenceDetection(
                type="wall_height_heuristic",
                height_px=wall_height_px, width_px=w,
                x=0, y=int(wall_rows.min()), y_bottom=int(wall_rows.max()),
                area_px=wall_height_px * w,
                expected_cm=round(270.0 * 0.75),
                dimension="height",
                px_per_cm=px_per_cm,
                confidence=0.35,
            ))

    # ── 5. Image-height fallback ─────────────────────────────────────────
    fallback_px_per_cm = float(h) / 270.0
    detections.append(ReferenceDetection(
        type="image_height_fallback",
        height_px=h, width_px=w,
        x=0, y=0, y_bottom=h,
        area_px=h * w,
        expected_cm=270.0,
        dimension="height",
        px_per_cm=fallback_px_per_cm,
        confidence=0.15,
    ))

    # ── 6. Determine best px_per_cm ────────────────────────────────────────
    # PRIORITY ORDER:
    #   1. Door height (200cm) — most reliable, gives exact brick count
    #   2. Ceiling-to-floor span (270cm) — very reliable
    #   3. Multiple medium-confidence refs — cross-validated average
    #   4. Any single reference — use it
    #   5. Image height fallback — last resort

    # Separate detections by reliability tier
    architectural = [d for d in detections
                     if d.type in ("door",) and d.dimension == "height"
                     and d.confidence >= 0.7]
    room_span = [d for d in detections
                 if d.type == "ceiling_to_floor" and d.confidence >= 0.7]
    furniture = [d for d in detections
                 if d.confidence >= 0.45
                 and d.type not in ("wall_height_heuristic", "image_height_fallback",
                                     "ceiling_to_floor", "door")]
    heuristics = [d for d in detections
                  if d.type in ("wall_height_heuristic", "image_height_fallback")]

    if architectural:
        # BEST CASE: door detected — use it directly
        # Door = 200cm. With 8cm bricks → 25 bricks fit in door height.
        # If multiple doors, average them (they should agree).
        total_w = sum(d.confidence for d in architectural)
        weighted_px_per_cm = sum(d.px_per_cm * d.confidence for d in architectural) / total_w
        confidence = "high"
        method = "door"
        logger.info(
            "PRIORITY: Using door reference directly → px/cm=%.3f "
            "(from %d door detection(s))",
            weighted_px_per_cm, len(architectural),
        )

    elif room_span:
        # GOOD: ceiling-to-floor span detected
        best = max(room_span, key=lambda d: d.confidence)
        weighted_px_per_cm = best.px_per_cm

        # Cross-validate with furniture if available
        if furniture:
            furn_avg = sum(d.px_per_cm for d in furniture) / len(furniture)
            deviation = abs(furn_avg - weighted_px_per_cm) / max(weighted_px_per_cm, 0.01)
            if deviation < 0.25:
                # Furniture agrees — blend slightly
                weighted_px_per_cm = weighted_px_per_cm * 0.75 + furn_avg * 0.25
                logger.info("Room span + furniture agree (dev=%.1f%%), blended", deviation * 100)

        confidence = "high"
        method = "ceiling_to_floor"
        logger.info("PRIORITY: Using ceiling-to-floor → px/cm=%.3f", weighted_px_per_cm)

    elif len(furniture) >= 2:
        # OK: multiple furniture refs — cross-validate
        # Reject outliers first
        vals = sorted(d.px_per_cm for d in furniture)
        if len(vals) >= 3:
            q1 = vals[len(vals) // 4]
            q3 = vals[3 * len(vals) // 4]
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            furniture = [d for d in furniture if lower <= d.px_per_cm <= upper]

        if furniture:
            total_w = sum(d.confidence for d in furniture)
            weighted_px_per_cm = sum(d.px_per_cm * d.confidence for d in furniture) / total_w
        else:
            weighted_px_per_cm = fallback_px_per_cm
        confidence = "medium"
        method = "multi_reference"
        logger.info("Using %d furniture references → px/cm=%.3f", len(furniture), weighted_px_per_cm)

    elif furniture:
        # Single furniture ref
        best = max(furniture, key=lambda d: d.confidence)
        weighted_px_per_cm = best.px_per_cm
        confidence = "medium"
        method = best.type
        logger.info("Using single reference '%s' → px/cm=%.3f", best.type, weighted_px_per_cm)

    else:
        # Fallback: use heuristics or image-level estimate
        if heuristics:
            best = max(heuristics, key=lambda d: d.confidence)
            weighted_px_per_cm = best.px_per_cm
            method = best.type
        else:
            weighted_px_per_cm = fallback_px_per_cm
            method = "image_height_fallback"
        confidence = "low"
        logger.info("FALLBACK: Using '%s' → px/cm=%.3f", method, weighted_px_per_cm)

    # ── 7. Sanity check against room proportions ─────────────────────────
    # If ceiling-to-floor is visible, the wall height should be <= 280cm
    if ceil_bottom is not None and floor_top is not None:
        room_height_px = floor_top - ceil_bottom
        if room_height_px > 50:
            implied_room_cm = room_height_px / weighted_px_per_cm
            if implied_room_cm < 200 or implied_room_cm > 400:
                logger.warning(
                    "Sanity check: implied room height %.0fcm is unreasonable "
                    "(expected 220-300cm), adjusting",
                    implied_room_cm,
                )
                # Correct toward 270cm
                corrected = float(room_height_px) / 270.0
                weighted_px_per_cm = weighted_px_per_cm * 0.3 + corrected * 0.7

    # ── 8. Compute wall height from calibration ──────────────────────────
    if box:
        selection_height_px = box[3] - box[1]
    elif wall_mask is not None:
        rows = np.where(wall_mask.any(axis=1))[0]
        selection_height_px = int(rows.max() - rows.min()) if len(rows) > 0 else h
    else:
        selection_height_px = h

    wall_height_cm = float(selection_height_px) / weighted_px_per_cm

    # Build serializable references list
    references = []
    for d in sorted(detections, key=lambda x: -x.confidence):
        if d.confidence < 0.1:
            continue
        references.append({
            "type": d.type,
            "dimension": d.dimension,
            "measured_px": int(d.height_px if d.dimension == "height" else d.width_px),
            "expected_cm": float(d.expected_cm),
            "computed_px_per_cm": round(float(d.px_per_cm), 3),
            "confidence": round(float(d.confidence), 2),
            "bbox": {"x": int(d.x), "y": int(d.y), "w": int(d.width_px), "h": int(d.height_px)},
        })

    logger.info(
        "Scale calibration: px/cm=%.3f from %d references "
        "(confidence=%s, method=%s, wall=%.0fcm)",
        weighted_px_per_cm, len(references),
        confidence, method, wall_height_cm,
    )
    for d in detections:
        if d.confidence >= 0.3:
            logger.info(
                "  ref: %-25s px/cm=%.3f  expected=%dcm  measured=%dpx  conf=%.2f",
                d.type, d.px_per_cm, int(d.expected_cm),
                int(d.height_px if d.dimension == "height" else d.width_px),
                d.confidence,
            )

    return {
        "px_per_cm": round(float(weighted_px_per_cm), 4),
        "wall_height_cm": round(float(wall_height_cm), 1),
        "confidence": confidence,
        "references": references,
        "method": method,
        "n_references": len([d for d in detections if d.confidence >= 0.3]),
    }
