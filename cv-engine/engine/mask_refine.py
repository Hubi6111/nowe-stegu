"""Stage 2: Mask Refinement — SAM2 refines coarse masks into pixel-precise boundaries.

Uses grid-based point sampling, mask-prompt input, and iterative refinement
for maximum accuracy.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "sam2"
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "sam2"

_predictor = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str = "sam2_hiera_small.pt"):
    """Load SAM2 model."""
    global _predictor, _device
    if _predictor is not None:
        return

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _device = _get_device()
    ckpt_path = MODELS_DIR / checkpoint
    model_cfg = "configs/sam2/sam2_hiera_s.yaml"

    logger.info("Loading SAM2 model: %s on %s", checkpoint, _device)
    sam2_model = build_sam2(
        model_cfg,
        str(ckpt_path),
        device=str(_device),
    )
    _predictor = SAM2ImagePredictor(sam2_model)
    logger.info("SAM2 model loaded")


def refine_mask(
    image: Image.Image,
    coarse_mask: np.ndarray,
    polygon: list[dict] | None = None,
    box: tuple[int, int, int, int] | None = None,
    exclude_mask: np.ndarray | None = None,
    n_iterations: int = 2,
) -> np.ndarray:
    """Refine a coarse mask using SAM2 with iterative refinement.

    Uses:
    - Grid-based point sampling for reliable prompts
    - Box prompt from user selection
    - Mask prompt for subsequent iterations
    - Explicit negative points from exclusion zones

    Args:
        image: RGB PIL image
        coarse_mask: Binary mask (uint8, 0/255) to refine
        polygon: Optional polygon points
        box: Optional (x1, y1, x2, y2) bounding box
        exclude_mask: Optional mask of areas to exclude (floor/ceiling)
        n_iterations: Number of SAM2 refinement passes

    Returns:
        Refined binary mask (uint8, 0/255)
    """
    load_model()

    img_array = np.array(image)
    _predictor.set_image(img_array)

    h, w = coarse_mask.shape[:2]

    # Determine bounding box
    if box is not None:
        sam_box = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
    elif polygon and len(polygon) >= 3:
        xs = [p["x"] for p in polygon]
        ys = [p["y"] for p in polygon]
        sam_box = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
    else:
        ys_idx, xs_idx = np.where(coarse_mask > 127)
        if len(xs_idx) == 0:
            return coarse_mask
        sam_box = np.array([
            xs_idx.min(), ys_idx.min(), xs_idx.max(), ys_idx.max()
        ], dtype=np.float32)

    # Generate strategic point prompts
    positive_points = _grid_sample_positive(coarse_mask, n_points=12)
    negative_points = _strategic_negatives(
        coarse_mask, sam_box, exclude_mask, n_points=8
    )

    point_coords = np.concatenate([positive_points, negative_points], axis=0)
    point_labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points)
    )

    # ── Iteration 1: Points + Box ─────────────────────────────
    masks, scores, logits = _predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=sam_box[None, :],
        multimask_output=True,
    )

    # Select the mask with highest IoU with the coarse mask
    best_idx = _select_best_mask(masks, coarse_mask)
    current_mask = masks[best_idx]
    current_logits = logits[best_idx]

    # ── Iteration 2+: Mask prompt refinement ──────────────────
    for iteration in range(1, n_iterations):
        # Use the previous mask's logits as a mask prompt
        mask_input = current_logits[None, :, :]  # (1, 256, 256)

        # Sample new points from the current mask boundaries
        boundary_pos, boundary_neg = _boundary_points(
            (current_mask * 255).astype(np.uint8),
            exclude_mask,
            n_each=4,
        )
        iter_coords = np.concatenate([
            positive_points[:4],  # Keep some core interior points
            boundary_pos,
            boundary_neg,
        ], axis=0)
        iter_labels = np.array(
            [1] * (4 + len(boundary_pos)) + [0] * len(boundary_neg)
        )

        masks2, scores2, logits2 = _predictor.predict(
            point_coords=iter_coords,
            point_labels=iter_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

        # Pick the best refined mask
        best_idx2 = _select_best_mask(
            masks2, (current_mask * 255).astype(np.uint8)
        )
        current_mask = masks2[best_idx2]
        current_logits = logits2[best_idx2]

    refined = (current_mask * 255).astype(np.uint8)

    logger.info(
        "SAM2 refine: %d iterations, %d pos + %d neg points, "
        "mask pixels=%d",
        n_iterations, len(positive_points), len(negative_points),
        refined.sum() // 255,
    )

    return refined


def _grid_sample_positive(
    mask: np.ndarray, n_points: int = 12
) -> np.ndarray:
    """Sample positive points on a regular grid inside the mask.

    Grid sampling is more reliable than random — ensures coverage
    of the entire wall area.
    """
    # Erode mask to ensure points are firmly inside
    kernel = np.ones((15, 15), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=2)

    ys, xs = np.where(eroded > 127)
    if len(xs) < n_points:
        # Fall back to non-eroded
        ys, xs = np.where(mask > 127)

    if len(xs) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2]], dtype=np.float32)

    # Sort by position and sample evenly across the mask
    indices = np.arange(len(xs))

    # Grid approach: divide the mask bbox into a grid
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Create grid cells
    grid_cols = max(1, int(np.sqrt(n_points * (x_max - x_min + 1) / max(y_max - y_min + 1, 1))))
    grid_rows = max(1, n_points // grid_cols)

    points = []
    cell_w = max(1, (x_max - x_min + 1) // grid_cols)
    cell_h = max(1, (y_max - y_min + 1) // grid_rows)

    for r in range(grid_rows):
        for c in range(grid_cols):
            if len(points) >= n_points:
                break
            cx_min = x_min + c * cell_w
            cx_max = min(x_min + (c + 1) * cell_w, x_max)
            cy_min = y_min + r * cell_h
            cy_max = min(y_min + (r + 1) * cell_h, y_max)

            # Find mask pixels in this cell
            cell_mask = (
                (xs >= cx_min) & (xs <= cx_max) &
                (ys >= cy_min) & (ys <= cy_max)
            )
            cell_indices = indices[cell_mask]
            if len(cell_indices) > 0:
                # Pick the center-most pixel in this cell
                mid = len(cell_indices) // 2
                idx = cell_indices[mid]
                points.append([xs[idx], ys[idx]])

    if len(points) == 0:
        # Absolute fallback: sample evenly
        step = max(1, len(xs) // n_points)
        for i in range(0, len(xs), step):
            points.append([xs[i], ys[i]])
            if len(points) >= n_points:
                break

    return np.array(points, dtype=np.float32)


def _strategic_negatives(
    mask: np.ndarray,
    box: np.ndarray,
    exclude_mask: np.ndarray | None,
    n_points: int = 8,
) -> np.ndarray:
    """Sample strategic negative points around the mask.

    Places negatives:
    - Just outside the mask (boundary negatives)
    - In ceiling area (above the mask)
    - In floor area (below the mask)
    - In exclude_mask areas (doors, windows, etc.)
    """
    h, w = mask.shape
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    points = []

    # 1. Points above the mask (ceiling exclusion)
    top_y = max(0, y1 - int((y2 - y1) * 0.05))
    if top_y > 5:
        points.append([int((x1 + x2) / 2), top_y])
        points.append([int(x1 + (x2 - x1) * 0.25), top_y])
        points.append([int(x1 + (x2 - x1) * 0.75), top_y])

    # 2. Points below the mask (floor exclusion)
    bot_y = min(h - 1, y2 + int((y2 - y1) * 0.05))
    if bot_y < h - 5:
        points.append([int((x1 + x2) / 2), bot_y])

    # 3. Points left and right of the mask
    left_x = max(0, x1 - int((x2 - x1) * 0.05))
    if left_x > 5:
        points.append([left_x, int((y1 + y2) / 2)])

    right_x = min(w - 1, x2 + int((x2 - x1) * 0.05))
    if right_x < w - 5:
        points.append([right_x, int((y1 + y2) / 2)])

    # 4. Points from the exclude_mask (floor/ceiling semantic zones)
    if exclude_mask is not None:
        excl_ys, excl_xs = np.where(exclude_mask > 127)
        if len(excl_xs) > 0:
            # Sample points near the box that are in the exclusion zone
            near_box = (
                (excl_xs >= max(0, x1 - 50)) & (excl_xs <= min(w, x2 + 50)) &
                (excl_ys >= max(0, y1 - 50)) & (excl_ys <= min(h, y2 + 50))
            )
            near_indices = np.where(near_box)[0]
            if len(near_indices) > 0:
                n_excl = min(3, len(near_indices))
                chosen = np.random.choice(near_indices, n_excl, replace=False)
                for idx in chosen:
                    points.append([excl_xs[idx], excl_ys[idx]])

    # Ensure we have at least a few negatives
    if len(points) < 2:
        # Corner negatives
        points.append([5, 5])
        points.append([w - 5, h - 5])

    return np.array(points[:n_points], dtype=np.float32)


def _boundary_points(
    mask: np.ndarray,
    exclude_mask: np.ndarray | None,
    n_each: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along the mask boundary for refinement.

    Returns positive points just inside the boundary,
    negative points just outside.
    """
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return (
            np.array([[w // 2, h // 2]], dtype=np.float32),
            np.array([[5, 5]], dtype=np.float32),
        )

    # Use largest contour
    contour = max(contours, key=cv2.contourArea)

    # Sample evenly along contour
    n_contour = len(contour)
    step = max(1, n_contour // (n_each * 2))

    positives = []
    negatives = []

    for i in range(0, n_contour, step):
        pt = contour[i][0]  # (x, y)
        x, y = int(pt[0]), int(pt[1])

        h, w = mask.shape
        # Inward offset (positive): find direction into mask
        # Outward offset (negative): find direction away from mask
        for dx, dy in [(-8, 0), (8, 0), (0, -8), (0, 8)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if mask[ny, nx] > 127 and len(positives) < n_each:
                    positives.append([nx, ny])
                elif mask[ny, nx] < 128 and len(negatives) < n_each:
                    negatives.append([nx, ny])

        if len(positives) >= n_each and len(negatives) >= n_each:
            break

    if not positives:
        ys, xs = np.where(mask > 127)
        if len(xs) > 0:
            positives = [[xs[len(xs) // 2], ys[len(ys) // 2]]]

    if not negatives:
        negatives = [[5, 5]]

    # Add exclude_mask negatives
    if exclude_mask is not None:
        excl_ys, excl_xs = np.where(exclude_mask > 127)
        if len(excl_xs) > 2:
            chosen = np.random.choice(len(excl_xs), min(2, len(excl_xs)), replace=False)
            for idx in chosen:
                negatives.append([excl_xs[idx], excl_ys[idx]])

    return (
        np.array(positives[:n_each], dtype=np.float32),
        np.array(negatives[:n_each + 2], dtype=np.float32),
    )


def _select_best_mask(
    masks: np.ndarray, reference: np.ndarray
) -> int:
    """Select the best mask using a combined recall + coverage score.

    Pure IoU penalizes larger masks (shadows, extended wall areas).
    Instead we score by:
    - recall: how much of the reference is covered (most important)
    - size_bonus: bonus for being larger (captures shadows)
    - penalty: deduction if mask extends far beyond reference
    """
    ref_binary = reference > 127
    ref_count = max(ref_binary.sum(), 1)
    best_score = -1
    best_idx = 0

    for i, m in enumerate(masks):
        m_binary = m > 0.5
        m_count = max(m_binary.sum(), 1)

        intersection = (m_binary & ref_binary).sum()
        recall = intersection / ref_count  # how much reference is captured
        precision = intersection / m_count  # how much of mask is reference

        # Size ratio: mask vs reference (>1 means larger)
        size_ratio = m_count / ref_count

        # Score: prioritize recall, reward moderate expansion, penalize extreme
        if size_ratio > 3.0:
            # Mask is 3x+ the reference — probably wrong
            score = recall * 0.3
        elif size_ratio > 1.5:
            # Moderately larger — mild bonus (shadow recovery)
            score = recall * 1.1
        else:
            # Similar size or smaller — standard IoU-like
            score = recall * (0.7 + 0.3 * precision)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx

