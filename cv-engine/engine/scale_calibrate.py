"""Scale Calibration — estimate real-world px/cm ratio from scene references.

Uses SegFormer semantic segmentation to detect architectural reference elements
(doors, windows, ceiling height) and compute a reliable px-per-cm conversion.

Reference dimensions (standard European interior):
- Standard door height: 200 cm
- Standard door width: 80–90 cm
- Standard window height: 120–150 cm
- Standard ceiling height: 270 cm (floor to ceiling)
- Standard baseboard/skirting: 8–10 cm
- Standard light switch height from floor: 120 cm
- Standard outlet height from floor: 30 cm
- Standard brick height: 6.5–8 cm
- Standard lamella/slat gap: 3 cm
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K class indices for reference elements
ADE20K = {
    "wall":      0,
    "floor":     3,
    "ceiling":   5,
    "door":     14,
    "window":    8,
    "column":   41,
    "stairs":   53,
    "person":   12,
    "cabinet":  10,
    "table":    15,
    "curtain":  18,
    "painting": 22,
    "shelf":    24,
    "armchair": 30,
    "sofa":     31,
    "bed":       7,
    "lamp":     36,
    "mirror":   27,
}

# Standard reference dimensions (cm)
REFERENCES = {
    "door_height":     200,   # standard interior door  
    "door_width":       85,   # standard door width
    "window_height":   140,   # average window height
    "ceiling_height":  270,   # standard floor-to-ceiling
    "baseboard_height": 10,   # typical baseboard/skirting
}


def calibrate_scale(
    image: Image.Image,
    segmentation_argmax: np.ndarray,
    wall_mask: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> dict:
    """Estimate px_per_cm from detected architectural references.

    Args:
        image: Room photo
        segmentation_argmax: Per-pixel ADE20K class labels
        wall_mask: Optional wall mask for context
        box: User selection box (x1, y1, x2, y2)

    Returns:
        dict with:
        - px_per_cm: float — pixels per centimeter
        - wall_height_cm: float — estimated wall height in cm
        - confidence: str — "high", "medium", "low"
        - references: list of detected reference elements with measurements
        - method: str — which reference was used
    """
    h, w = segmentation_argmax.shape
    references = []
    px_per_cm_estimates = []

    # ── 1. Door detection (most reliable reference) ───────────────────────
    door_mask = (segmentation_argmax == ADE20K["door"]).astype(np.uint8)
    if door_mask.sum() > 100:
        door_info = _measure_element(door_mask, "door", h, w)
        if door_info:
            # Door height = 200cm standard
            px_per_cm = float(door_info["height_px"]) / REFERENCES["door_height"]
            px_per_cm_estimates.append(("door_height", px_per_cm, 1.0))
            references.append({
                "type": "door",
                "height_px": int(door_info["height_px"]),
                "width_px": int(door_info["width_px"]),
                "expected_height_cm": REFERENCES["door_height"],
                "computed_px_per_cm": round(float(px_per_cm), 3),
            })

            # Cross-validate with door width
            if door_info["width_px"] > 20:
                px_per_cm_w = door_info["width_px"] / REFERENCES["door_width"]
                ratio = px_per_cm / px_per_cm_w if px_per_cm_w > 0 else 1
                # If height/width ratios roughly match (~200/85 ≈ 2.35)
                if 0.5 < ratio < 4.0:
                    px_per_cm_estimates.append(("door_width", px_per_cm_w, 0.6))

            logger.info(
                "Door detected: %dpx tall, px/cm=%.3f (expected %dcm)",
                door_info["height_px"], px_per_cm, REFERENCES["door_height"],
            )

    # ── 2. Window detection ───────────────────────────────────────────────
    window_mask = (segmentation_argmax == ADE20K["window"]).astype(np.uint8)
    if window_mask.sum() > 100:
        window_info = _measure_element(window_mask, "window", h, w)
        if window_info:
            px_per_cm = float(window_info["height_px"]) / REFERENCES["window_height"]
            px_per_cm_estimates.append(("window_height", px_per_cm, 0.7))
            references.append({
                "type": "window",
                "height_px": int(window_info["height_px"]),
                "width_px": int(window_info["width_px"]),
                "expected_height_cm": REFERENCES["window_height"],
                "computed_px_per_cm": round(float(px_per_cm), 3),
            })
            logger.info(
                "Window detected: %dpx tall, px/cm=%.3f",
                window_info["height_px"], px_per_cm,
            )

    # ── 3. Ceiling-to-floor distance (reliable for full-room shots) ──────
    ceiling_mask = (segmentation_argmax == ADE20K["ceiling"]).astype(np.uint8)
    floor_mask = (segmentation_argmax == ADE20K["floor"]).astype(np.uint8)

    if ceiling_mask.sum() > 50 and floor_mask.sum() > 50:
        ceil_rows = np.where(ceiling_mask.any(axis=1))[0]
        floor_rows = np.where(floor_mask.any(axis=1))[0]

        if len(ceil_rows) > 0 and len(floor_rows) > 0:
            ceiling_bottom = int(ceil_rows.max())
            floor_top = int(floor_rows.min())
            room_height_px = floor_top - ceiling_bottom

            if room_height_px > 50:
                px_per_cm = float(room_height_px) / REFERENCES["ceiling_height"]
                px_per_cm_estimates.append(("ceiling_to_floor", px_per_cm, 0.85))
                references.append({
                    "type": "ceiling_to_floor",
                    "height_px": int(room_height_px),
                    "ceiling_bottom_y": ceiling_bottom,
                    "floor_top_y": floor_top,
                    "expected_height_cm": REFERENCES["ceiling_height"],
                    "computed_px_per_cm": round(float(px_per_cm), 3),
                })
                logger.info(
                    "Ceiling-to-floor: %dpx, px/cm=%.3f (expected %dcm)",
                    room_height_px, px_per_cm, REFERENCES["ceiling_height"],
                )

    # ── 4. Wall-only height heuristic (fallback) ─────────────────────────
    if wall_mask is not None:
        wall_rows = np.where(wall_mask.any(axis=1))[0]
        if len(wall_rows) > 10:
            wall_height_px = int(wall_rows.max() - wall_rows.min())
            # Wall usually covers 60-90% of ceiling height
            px_per_cm = float(wall_height_px) / (REFERENCES["ceiling_height"] * 0.75)
            px_per_cm_estimates.append(("wall_height_heuristic", px_per_cm, 0.4))
            references.append({
                "type": "wall_height_heuristic",
                "height_px": wall_height_px,
                "expected_height_cm": round(REFERENCES["ceiling_height"] * 0.75),
                "computed_px_per_cm": round(float(px_per_cm), 3),
            })

    # ── 5. Image-height fallback (lowest confidence) ─────────────────────
    # Typical interior photo shows ~270cm of room height
    fallback_px_per_cm = h / REFERENCES["ceiling_height"]
    px_per_cm_estimates.append(("image_height_fallback", fallback_px_per_cm, 0.2))

    # ── Weighted average of all estimates ─────────────────────────────────
    if not px_per_cm_estimates:
        return {
            "px_per_cm": fallback_px_per_cm,
            "wall_height_cm": REFERENCES["ceiling_height"],
            "confidence": "low",
            "references": [],
            "method": "fallback",
        }

    total_weight = sum(w for _, _, w in px_per_cm_estimates)
    weighted_px_per_cm = sum(
        v * w for _, v, w in px_per_cm_estimates
    ) / total_weight

    # Determine confidence
    high_conf = [e for e in px_per_cm_estimates if e[2] >= 0.85]
    if high_conf:
        confidence = "high"
        method = high_conf[0][0]
    elif any(e[2] >= 0.6 for e in px_per_cm_estimates):
        confidence = "medium"
        method = max(px_per_cm_estimates, key=lambda x: x[2])[0]
    else:
        confidence = "low"
        method = "weighted_average"

    # Compute wall height from the calibration
    if box:
        selection_height_px = box[3] - box[1]
    elif wall_mask is not None:
        rows = np.where(wall_mask.any(axis=1))[0]
        selection_height_px = rows.max() - rows.min() if len(rows) > 0 else h
    else:
        selection_height_px = h

    wall_height_cm = selection_height_px / weighted_px_per_cm

    logger.info(
        "Scale calibration: px/cm=%.3f (confidence=%s, method=%s), "
        "wall height=%.0fcm, %d references",
        weighted_px_per_cm, confidence, method,
        wall_height_cm, len(references),
    )

    return {
        "px_per_cm": round(float(weighted_px_per_cm), 4),
        "wall_height_cm": round(float(wall_height_cm), 1),
        "confidence": confidence,
        "references": references,
        "method": method,
    }


def _measure_element(
    binary_mask: np.ndarray,
    element_type: str,
    img_h: int,
    img_w: int,
) -> dict | None:
    """Measure the bounding box of the largest instance of an element."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Find the largest contour (most likely the actual element)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Filter out very small detections (noise)
    min_area = img_h * img_w * 0.002  # at least 0.2% of image
    if area < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    return {
        "height_px": int(bh),
        "width_px": int(bw),
        "x": int(x), "y": int(y),
        "area_px": int(area),
    }


def compute_calibrated_texture_scale(
    calibration: dict,
    texture_height_px: int,
    meta: dict,
    selection_height_px: float,
) -> float:
    """Compute texture scale from calibration data.

    Uses the calibrated px_per_cm to ensure:
    - Bricks are exactly 8cm tall (moduleHeightMm / 10)
    - Lamella gaps are exactly 3cm (jointMm / 10)
    - etc.

    Args:
        calibration: Output from calibrate_scale()
        texture_height_px: Height of the albedo texture image
        meta: Product metadata with dimensions
        selection_height_px: Height of the wall selection in pixels

    Returns:
        Scale factor for the texture
    """
    px_per_cm = calibration["px_per_cm"]

    # Calculate the real-world height of the albedo tile
    module_h_cm = float(meta.get("moduleHeightMm", 65)) / 10.0
    joint_cm = float(meta.get("jointMm", 10)) / 10.0
    layout = meta.get("layoutType", "running-bond")

    if layout in ("running-bond", "stretcher-bond", "random-stone"):
        courses = int(meta.get("albedoBrickCourses", meta.get("albedoCourses", 8)))
        courses = max(1, min(courses, 32))
        albedo_h_cm = module_h_cm * courses + joint_cm * max(courses - 1, 0)
    else:
        planks = int(meta.get("albedoStackPlanks", meta.get("albedoPlankCount", 2)))
        planks = max(1, min(planks, 12))
        albedo_h_cm = module_h_cm * planks + joint_cm * max(planks - 1, 0)

    # Target height in pixels for the full albedo tile
    target_h_px = albedo_h_cm * px_per_cm

    scale = target_h_px / float(texture_height_px)

    # Verify: one brick should be this many pixels
    brick_px = module_h_cm * px_per_cm
    joint_px = joint_cm * px_per_cm

    logger.info(
        "Calibrated texture scale: %.4f — "
        "brick=%.1fcm (%.1fpx), joint=%.1fcm (%.1fpx), "
        "albedo tile=%.1fcm (%.1fpx), px/cm=%.3f",
        scale, module_h_cm, brick_px, joint_cm, joint_px,
        albedo_h_cm, target_h_px, px_per_cm,
    )

    return scale
