"""Stage 1: Wall Prior — semantic segmentation to detect wall regions.

Uses mmsegmentation with SegFormer (or falls back to HuggingFace SegFormer)
to produce a probability mask of wall pixels from the ADE20K wall class.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K class index for "wall" = 0 (background), actual wall = class 1
# Full list: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
ADE20K_WALL_CLASS = 0  # "wall" in ADE20K  (class index 0 = wall)

_model = None
_processor = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    """Load SegFormer-B3 fine-tuned on ADE20K via HuggingFace transformers."""
    global _model, _processor, _device
    if _model is not None:
        return

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    _device = _get_device()
    model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
    logger.info("Loading wall prior model: %s on %s", model_name, _device)

    _processor = SegformerImageProcessor.from_pretrained(model_name)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()

    logger.info("Wall prior model loaded (SegFormer-B3 ADE20K)")


def predict_wall_mask(image: Image.Image) -> np.ndarray:
    """Predict wall probability mask from a room photo.

    Args:
        image: RGB PIL image

    Returns:
        Wall probability mask as float32 numpy array [0..1], same size as input
    """
    load_model()

    inputs = _processor(images=image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    # outputs.logits shape: (1, num_classes, H/4, W/4)
    logits = outputs.logits
    # Upsample to original size
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )

    # Softmax and extract wall class probability
    probs = torch.nn.functional.softmax(upsampled, dim=1)

    # ADE20K: class 0 = "wall"
    wall_prob = probs[0, ADE20K_WALL_CLASS].cpu().numpy()

    return wall_prob.astype(np.float32)


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
