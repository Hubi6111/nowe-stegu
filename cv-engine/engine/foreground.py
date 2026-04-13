"""Stage 3: Foreground Occlusion — detect objects in front of the wall.

Uses GroundingDINO for open-set object detection + SAM2 for segmentation
to create a mask of all objects that should appear IN FRONT of the texture.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Default prompt listing common interior objects
DEFAULT_PROMPT = (
    "furniture . sofa . chair . table . TV . television . lamp . plant . "
    "frame . picture . shelf . curtain . switch . outlet . door . window . "
    "person . book . cabinet . radiator . mirror . clock . vase"
)

_gdino_model = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_gdino_path():
    """Add GroundingDINO to sys.path."""
    gdino_dir = str(VENDOR_DIR / "GroundingDINO")
    if gdino_dir not in sys.path:
        sys.path.insert(0, gdino_dir)


def load_model():
    """Load GroundingDINO model."""
    global _gdino_model, _device
    if _gdino_model is not None:
        return

    _ensure_gdino_path()
    from groundingdino.util.inference import load_model as load_gdino

    _device = _get_device()
    config_path = str(
        VENDOR_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    )
    ckpt_path = str(MODELS_DIR / "grounding-dino" / "groundingdino_swint_ogc.pth")

    logger.info("Loading GroundingDINO: SwinT on %s", _device)
    _gdino_model = load_gdino(config_path, ckpt_path, device=str(_device))
    logger.info("GroundingDINO loaded")


def detect_foreground_boxes(
    image: Image.Image,
    text_prompt: str = DEFAULT_PROMPT,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> list[dict]:
    """Detect foreground objects and return bounding boxes.

    Returns:
        List of {"box": [x1, y1, x2, y2], "label": str, "score": float}
    """
    load_model()
    _ensure_gdino_path()
    from groundingdino.util.inference import predict as gdino_predict
    import groundingdino.datasets.transforms as T

    # Prepare image for GroundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(image, None)

    boxes, logits, phrases = gdino_predict(
        model=_gdino_model,
        image=img_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=str(_device),
    )

    w, h = image.size
    results = []
    for box, score, label in zip(boxes, logits, phrases):
        # Convert from cxcywh normalized to xyxy pixel
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        results.append({
            "box": [x1, y1, x2, y2],
            "label": label,
            "score": float(score),
        })

    logger.info("Detected %d foreground objects", len(results))
    return results


def detect_foreground_mask(
    image: Image.Image,
    wall_mask: np.ndarray | None = None,
    text_prompt: str = DEFAULT_PROMPT,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> np.ndarray:
    """Detect foreground objects and produce a combined occlusion mask.

    Uses GroundingDINO for detection + SAM2 for precise segmentation.

    Args:
        image: RGB PIL image
        wall_mask: Optional wall mask — only keep foreground objects overlapping with wall
        text_prompt: Object categories to detect
        box_threshold: Detection confidence threshold
        text_threshold: Text matching threshold

    Returns:
        Binary occlusion mask (uint8, 0/255) — white = foreground object
    """
    detections = detect_foreground_boxes(
        image, text_prompt, box_threshold, text_threshold
    )

    if not detections:
        return np.zeros((image.height, image.width), dtype=np.uint8)

    # Use SAM2 to segment each detected box
    from engine.mask_refine import load_model as load_sam2, _predictor as sam2_pred

    # Ensure SAM2 is loaded
    from engine import mask_refine
    mask_refine.load_model()

    img_array = np.array(image)
    mask_refine._predictor.set_image(img_array)

    combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for det in detections:
        box = np.array(det["box"])
        masks, scores, _ = mask_refine._predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        obj_mask = masks[best_idx].astype(np.uint8)

        # If wall mask provided, only keep parts that overlap with wall
        if wall_mask is not None:
            wall_binary = (wall_mask > 127).astype(np.uint8)
            overlap = np.logical_and(obj_mask, wall_binary)
            # Keep the object if it overlaps significantly with wall area
            if overlap.sum() > obj_mask.sum() * 0.05:
                combined_mask = np.maximum(combined_mask, obj_mask * 255)
        else:
            combined_mask = np.maximum(combined_mask, obj_mask * 255)

    logger.info("Foreground mask: %d objects segmented", len(detections))
    return combined_mask
