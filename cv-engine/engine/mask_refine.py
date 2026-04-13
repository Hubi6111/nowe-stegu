"""Stage 2: Mask Refinement — SAM2 refines coarse masks into pixel-precise boundaries.

Takes the user polygon + wall prior and uses SAM2 point/box prompts
to produce clean, high-resolution wall masks.
"""

import logging
from pathlib import Path
from typing import Optional

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

    # SAM2 config for hiera_small
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
) -> np.ndarray:
    """Refine a coarse mask using SAM2.

    Uses the coarse mask to generate point prompts (positive inside, negative outside)
    and optionally a bounding box from the user polygon.

    Args:
        image: RGB PIL image
        coarse_mask: Binary mask (uint8, 0/255) to refine
        polygon: Optional list of {"x": float, "y": float} polygon points

    Returns:
        Refined binary mask (uint8, 0/255)
    """
    load_model()

    img_array = np.array(image)
    _predictor.set_image(img_array)

    # Generate point prompts from coarse mask
    positive_points, negative_points = _sample_points_from_mask(coarse_mask)

    point_coords = np.concatenate([positive_points, negative_points], axis=0)
    point_labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points)
    )

    # Bounding box from polygon or mask
    if polygon and len(polygon) >= 3:
        xs = [p["x"] for p in polygon]
        ys = [p["y"] for p in polygon]
        box = np.array([min(xs), min(ys), max(xs), max(ys)])
    else:
        ys, xs = np.where(coarse_mask > 127)
        if len(xs) > 0:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        else:
            return coarse_mask

    masks, scores, _ = _predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=True,
    )

    # Pick the mask with highest score
    best_idx = np.argmax(scores)
    refined = masks[best_idx].astype(np.uint8) * 255

    return refined


def _sample_points_from_mask(
    mask: np.ndarray, n_positive: int = 5, n_negative: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Sample positive (inside) and negative (outside) points from mask."""
    # Erode mask slightly for reliable interior points
    import cv2

    kernel = np.ones((11, 11), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=2)

    # Positive points from eroded interior
    pos_ys, pos_xs = np.where(eroded > 127)
    if len(pos_xs) < n_positive:
        pos_ys, pos_xs = np.where(mask > 127)

    if len(pos_xs) > 0:
        indices = np.random.choice(len(pos_xs), min(n_positive, len(pos_xs)), replace=False)
        positive = np.stack([pos_xs[indices], pos_ys[indices]], axis=1)
    else:
        positive = np.array([[mask.shape[1] // 2, mask.shape[0] // 2]])

    # Negative points from exterior
    neg_ys, neg_xs = np.where(mask < 128)
    if len(neg_xs) > 0:
        indices = np.random.choice(len(neg_xs), min(n_negative, len(neg_xs)), replace=False)
        negative = np.stack([neg_xs[indices], neg_ys[indices]], axis=1)
    else:
        negative = np.array([[0, 0]])

    return positive.astype(np.float32), negative.astype(np.float32)
