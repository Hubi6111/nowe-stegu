"""Stage 1: Wall Prior — semantic segmentation to detect wall regions.

Uses SegFormer-B5 (largest) fine-tuned on ADE20K via HuggingFace.
Returns wall probability + explicit floor/ceiling exclusion maps.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K 150-class index (0-indexed)
ADE20K_WALL = 0        # wall
ADE20K_FLOOR = 3       # floor, flooring
ADE20K_CEILING = 5     # ceiling
ADE20K_DOOR = 14       # door
ADE20K_WINDOW = 8      # windowpane, window
ADE20K_COLUMN = 41     # column, pillar
ADE20K_STAIRS = 53     # stairs, steps

# Classes that should NEVER be part of a wall mask
EXCLUDE_CLASSES = [
    ADE20K_FLOOR,
    ADE20K_CEILING,
    ADE20K_STAIRS,
    # Door and window are wall-adjacent — include them in exclusion
    ADE20K_DOOR,
    ADE20K_WINDOW,
]

_model = None
_processor = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    """Load SegFormer-B5 (largest variant) fine-tuned on ADE20K."""
    global _model, _processor, _device
    if _model is not None:
        return

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    _device = _get_device()
    # Use B5 — largest and most accurate variant
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    logger.info("Loading wall prior model: %s on %s", model_name, _device)

    _processor = SegformerImageProcessor.from_pretrained(model_name)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()

    logger.info("Wall prior model loaded (SegFormer-B5 ADE20K)")


def predict_full(image: Image.Image) -> dict:
    """Run full semantic segmentation, returning wall prob + exclusion maps.

    Returns:
        dict with keys:
        - "wall_prob": float32 [0..1] wall probability
        - "floor_prob": float32 [0..1] floor probability
        - "ceiling_prob": float32 [0..1] ceiling probability
        - "exclude_mask": uint8 [0/255] combined exclusion mask
        - "argmax": int32 per-pixel class labels
    """
    load_model()

    inputs = _processor(images=image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    logits = outputs.logits  # (1, 150, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )

    probs = torch.nn.functional.softmax(upsampled, dim=1)
    probs_np = probs[0].cpu().numpy()  # (150, H, W)
    argmax = probs_np.argmax(axis=0).astype(np.int32)  # (H, W)

    wall_prob = probs_np[ADE20K_WALL]
    floor_prob = probs_np[ADE20K_FLOOR]
    ceiling_prob = probs_np[ADE20K_CEILING]

    # Build exclusion mask: any pixel where excluded class has highest prob
    exclude_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for cls_idx in EXCLUDE_CLASSES:
        exclude_mask[argmax == cls_idx] = 255

    # Also exclude areas where floor/ceiling probability > 0.15
    # even if not the argmax (catches transition zones)
    exclude_mask[(floor_prob > 0.15) | (ceiling_prob > 0.15)] = 255

    return {
        "wall_prob": wall_prob.astype(np.float32),
        "floor_prob": floor_prob.astype(np.float32),
        "ceiling_prob": ceiling_prob.astype(np.float32),
        "exclude_mask": exclude_mask,
        "argmax": argmax,
    }


def predict_wall_mask(image: Image.Image) -> np.ndarray:
    """Predict wall probability mask (backward-compatible).

    Returns:
        Wall probability mask as float32 numpy array [0..1]
    """
    result = predict_full(image)
    return result["wall_prob"]


def predict_wall_binary(
    image: Image.Image, threshold: float = 0.3
) -> np.ndarray:
    """Predict binary wall mask.

    Returns:
        Binary mask as uint8 numpy array (0 or 255)
    """
    prob = predict_wall_mask(image)
    binary = (prob > threshold).astype(np.uint8) * 255
    return binary
