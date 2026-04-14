"""Scale Calibration — estimate real-world px/cm ratio from scene references.

DETERMINISTIC APPROACH:
The most reliable scale reference in any interior photo is the
ceiling-to-floor distance (standard European: 260-270cm).

Priority order:
1. Ceiling-to-floor pixel span → divide by 270cm
2. Door pixel height → divide by 200cm  
3. Wall selection spans full height (ceiling to floor) → divide by 270cm
4. Wall selection partial height → estimate from position relative to
   ceiling/floor edges

This gives consistent, repeatable results because:
- Ceiling and floor are LARGE surfaces that SegFormer detects reliably
- The user's selection box constrains the wall region precisely
- Standard dimensions (270cm ceiling, 200cm door) vary by at most ±10cm
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K 150-class indices for reference elements
ADE20K = {
    "wall": 0, "floor": 3, "ceiling": 5, "door": 14, "window": 8,
    "person": 12, "cabinet": 10, "wardrobe": 35, "refrigerator": 49,
    "sofa": 23, "chair": 19, "table": 15, "bed": 7,
}

# Standard European interior dimensions (cm)
CEILING_HEIGHT_CM = 270.0
DOOR_HEIGHT_CM = 200.0


@dataclass
class ScaleRef:
    """A scale reference measurement."""
    name: str
    height_px: int
    real_cm: float
    px_per_cm: float
    priority: int  # lower = better (1 = best)


def calibrate_scale(
    image: Image.Image,
    segmentation_argmax: np.ndarray,
    wall_mask: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> dict:
    """Estimate px_per_cm using the most reliable available reference.

    Returns dict with px_per_cm, wall_height_cm, confidence, method.
    """
    h, w = segmentation_argmax.shape
    refs: list[ScaleRef] = []

    # ── 1. CEILING-TO-FLOOR span (most reliable) ────────────────────────
    ceiling_mask = (segmentation_argmax == ADE20K["ceiling"]).astype(np.uint8)
    floor_mask = (segmentation_argmax == ADE20K["floor"]).astype(np.uint8)

    ceil_bottom = None
    floor_top = None

    if ceiling_mask.sum() > 100:
        ceil_rows = np.where(ceiling_mask.any(axis=1))[0]
        if len(ceil_rows) > 0:
            ceil_bottom = int(ceil_rows.max())

    if floor_mask.sum() > 100:
        floor_rows = np.where(floor_mask.any(axis=1))[0]
        if len(floor_rows) > 0:
            floor_top = int(floor_rows.min())

    if ceil_bottom is not None and floor_top is not None:
        span_px = floor_top - ceil_bottom
        if span_px > 50:
            ppc = float(span_px) / CEILING_HEIGHT_CM
            refs.append(ScaleRef(
                name="ceiling_to_floor",
                height_px=span_px,
                real_cm=CEILING_HEIGHT_CM,
                px_per_cm=ppc,
                priority=1,
            ))
            logger.info(
                "REF ceiling→floor: %dpx = %.0fcm → px/cm=%.3f",
                span_px, CEILING_HEIGHT_CM, ppc,
            )

    # ── 2. DOOR detection (very reliable, standard 200cm) ───────────────
    door_mask = (segmentation_argmax == ADE20K["door"]).astype(np.uint8)
    if door_mask.sum() > 200:
        contours, _ = cv2.findContours(
            door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.005:  # at least 0.5% of image
                continue
            _, _, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(bw, 1)
            if aspect < 1.3:  # doors are taller than wide
                continue

            # Validate: door bottom should be near floor if floor detected
            _, by, _, _ = cv2.boundingRect(cnt)
            door_bottom = by + bh
            if floor_top is not None and abs(door_bottom - floor_top) > h * 0.15:
                continue  # door bottom too far from floor — skip

            ppc = float(bh) / DOOR_HEIGHT_CM
            refs.append(ScaleRef(
                name="door",
                height_px=bh,
                real_cm=DOOR_HEIGHT_CM,
                px_per_cm=ppc,
                priority=2,
            ))
            logger.info(
                "REF door: %dpx = %.0fcm → px/cm=%.3f (aspect=%.1f)",
                bh, DOOR_HEIGHT_CM, ppc, aspect,
            )

    # ── 3. WALL SELECTION relative to ceiling/floor ─────────────────────
    if box and wall_mask is not None:
        box_y1, box_y2 = box[1], box[3]
        box_h = box_y2 - box_y1

        # Check if selection touches ceiling and floor regions
        touches_ceiling = ceil_bottom is not None and abs(box_y1 - ceil_bottom) < h * 0.08
        touches_floor = floor_top is not None and abs(box_y2 - floor_top) < h * 0.08

        if touches_ceiling and touches_floor:
            # Selection = full wall height = ceiling height
            ppc = float(box_h) / CEILING_HEIGHT_CM
            refs.append(ScaleRef(
                name="wall_full_height",
                height_px=box_h,
                real_cm=CEILING_HEIGHT_CM,
                px_per_cm=ppc,
                priority=1,  # Same reliability as ceiling-to-floor
            ))
            logger.info(
                "REF wall full height: %dpx = %.0fcm → px/cm=%.3f",
                box_h, CEILING_HEIGHT_CM, ppc,
            )
        elif touches_floor and ceil_bottom is not None:
            # Selection starts below ceiling — compute from known ceiling pos
            full_wall_px = floor_top - ceil_bottom if floor_top else box_y2 - ceil_bottom
            if full_wall_px > 50:
                ppc = float(full_wall_px) / CEILING_HEIGHT_CM
                refs.append(ScaleRef(
                    name="wall_from_ceiling",
                    height_px=full_wall_px,
                    real_cm=CEILING_HEIGHT_CM,
                    px_per_cm=ppc,
                    priority=2,
                ))
        elif touches_ceiling and floor_top is not None:
            # Selection ends above floor
            full_wall_px = floor_top - ceil_bottom if ceil_bottom else floor_top - box_y1
            if full_wall_px > 50:
                ppc = float(full_wall_px) / CEILING_HEIGHT_CM
                refs.append(ScaleRef(
                    name="wall_to_floor",
                    height_px=full_wall_px,
                    real_cm=CEILING_HEIGHT_CM,
                    px_per_cm=ppc,
                    priority=2,
                ))

    # ── 4. FALLBACK: image height ≈ room height ────────────────────────
    if not refs:
        # Last resort: assume the full image shows ~270cm
        ppc = float(h) / CEILING_HEIGHT_CM
        refs.append(ScaleRef(
            name="image_height_fallback",
            height_px=h,
            real_cm=CEILING_HEIGHT_CM,
            px_per_cm=ppc,
            priority=10,
        ))
        logger.info("FALLBACK: image height %dpx ≈ %.0fcm → px/cm=%.3f", h, CEILING_HEIGHT_CM, ppc)

    # ── SELECT BEST REFERENCE ───────────────────────────────────────────
    # Sort by priority (lower = better), take the best
    refs.sort(key=lambda r: r.priority)
    best = refs[0]

    # If multiple refs at same priority, average them
    same_priority = [r for r in refs if r.priority == best.priority]
    if len(same_priority) > 1:
        px_per_cm = sum(r.px_per_cm for r in same_priority) / len(same_priority)
        logger.info(
            "Averaged %d refs at priority %d → px/cm=%.3f",
            len(same_priority), best.priority, px_per_cm,
        )
    else:
        px_per_cm = best.px_per_cm

    # ── SANITY CHECK ────────────────────────────────────────────────────
    # With px_per_cm, an 8cm brick should be between 10px and 200px
    brick_8cm_px = px_per_cm * 8.0
    if brick_8cm_px < 5 or brick_8cm_px > 300:
        logger.warning(
            "SANITY FAIL: 8cm brick = %.0fpx (should be 10-200px), "
            "falling back to image/270",
            brick_8cm_px,
        )
        px_per_cm = float(h) / CEILING_HEIGHT_CM

    # ── COMPUTE WALL HEIGHT ─────────────────────────────────────────────
    if box:
        selection_height_px = box[3] - box[1]
    elif wall_mask is not None:
        rows = np.where(wall_mask.any(axis=1))[0]
        selection_height_px = int(rows.max() - rows.min()) if len(rows) > 0 else h
    else:
        selection_height_px = h

    wall_height_cm = float(selection_height_px) / px_per_cm

    # Confidence
    if best.priority <= 1:
        confidence = "high"
    elif best.priority <= 2:
        confidence = "high"
    elif best.priority <= 5:
        confidence = "medium"
    else:
        confidence = "low"

    method = best.name

    # Build references list for logging
    references = []
    for r in refs:
        references.append({
            "type": r.name,
            "dimension": "height",
            "measured_px": r.height_px,
            "expected_cm": r.real_cm,
            "computed_px_per_cm": round(r.px_per_cm, 3),
            "confidence": round(1.0 / r.priority, 2),
            "bbox": {"x": 0, "y": 0, "w": 0, "h": r.height_px},
        })

    logger.info(
        "FINAL SCALE: px/cm=%.3f (method=%s, confidence=%s) "
        "→ 8cm brick=%.0fpx, wall=%.0fcm, door(200cm)=%.0fpx",
        px_per_cm, method, confidence,
        px_per_cm * 8.0, wall_height_cm, px_per_cm * 200.0,
    )

    return {
        "px_per_cm": round(float(px_per_cm), 4),
        "wall_height_cm": round(float(wall_height_cm), 1),
        "confidence": confidence,
        "references": references,
        "method": method,
        "n_references": len(refs),
    }
