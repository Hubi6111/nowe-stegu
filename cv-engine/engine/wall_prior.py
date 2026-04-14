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

    target_h, target_w = image.height, image.width

    # Multi-scale inference: process at 2 scales and average logits.
    # This dramatically improves accuracy for shadows, small objects,
    # and boundary regions where single-scale often fails.
    scales = [1.0, 1.5]
    accumulated_logits = None

    for scale in scales:
        if scale == 1.0:
            img_input = image
        else:
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            img_input = image.resize((new_w, new_h), Image.LANCZOS)

        inputs = _processor(images=img_input, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

        logits = outputs.logits  # (1, 150, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        if accumulated_logits is None:
            accumulated_logits = upsampled
        else:
            accumulated_logits = accumulated_logits + upsampled

    # Average the multi-scale logits
    accumulated_logits = accumulated_logits / len(scales)

    probs = torch.nn.functional.softmax(accumulated_logits, dim=1)
    probs_np = probs[0].cpu().numpy()  # (150, H, W)
    argmax = probs_np.argmax(axis=0).astype(np.int32)  # (H, W)

    wall_prob = probs_np[ADE20K_WALL]
    floor_prob = probs_np[ADE20K_FLOOR]
    ceiling_prob = probs_np[ADE20K_CEILING]

    # Build exclusion mask — ONLY for very confident non-wall areas.
    # We must be careful NOT to exclude shadowed wall areas that SegFormer
    # might misclassify. Only exclude when:
    # 1. The argmax class is definitely not wall AND
    # 2. The wall probability is very low AND
    # 3. The non-wall class probability is high
    exclude_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    # Hard exclusion: non-wall class is dominant AND wall prob is very low
    for cls_idx in EXCLUDE_CLASSES:
        class_confident = (argmax == cls_idx)
        wall_very_low = (wall_prob < 0.10)
        class_prob_high = (probs_np[cls_idx] > 0.5)
        exclude_mask[class_confident & wall_very_low & class_prob_high] = 255

    # Soft exclusion mask (float 0-1) for gentler suppression in pipeline
    # This gives a continuous "how non-wall is this pixel" signal
    soft_exclude = np.zeros((image.height, image.width), dtype=np.float32)
    soft_exclude = np.maximum(soft_exclude, np.clip(floor_prob - wall_prob, 0, 1))
    soft_exclude = np.maximum(soft_exclude, np.clip(ceiling_prob - wall_prob, 0, 1))

    return {
        "wall_prob": wall_prob.astype(np.float32),
        "floor_prob": floor_prob.astype(np.float32),
        "ceiling_prob": ceiling_prob.astype(np.float32),
        "exclude_mask": exclude_mask,
        "soft_exclude": soft_exclude,
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
