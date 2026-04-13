"""Stage 4: Depth Estimation — monocular depth via Depth-Anything-V2.

Produces a per-pixel relative depth map for perspective-aware texture projection.
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
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "depth-anything-v2"

_model = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_path():
    da_dir = str(VENDOR_DIR / "Depth-Anything-V2")
    if da_dir not in sys.path:
        sys.path.insert(0, da_dir)


def load_model(checkpoint: str = "depth_anything_v2_vits.pth"):
    """Load Depth-Anything-V2 Small model."""
    global _model, _device
    if _model is not None:
        return

    _ensure_path()
    from depth_anything_v2.dpt import DepthAnythingV2

    _device = _get_device()
    ckpt_path = MODELS_DIR / checkpoint

    logger.info("Loading Depth-Anything-V2 Small on %s", _device)

    # ViT-S config
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    _model = model
    logger.info("Depth-Anything-V2 loaded")


def estimate_depth(image: Image.Image) -> np.ndarray:
    """Estimate relative depth from a single image.

    Args:
        image: RGB PIL image

    Returns:
        Depth map as float32 numpy array, normalized to [0, 1].
        Closer objects have lower values, farther objects have higher values.
    """
    load_model()

    img_np = np.array(image)
    # Depth-Anything-V2 expects BGR
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        depth = _model.infer_image(img_bgr)

    # Normalize to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    return depth_norm.astype(np.float32)


def depth_to_colormap(depth: np.ndarray) -> Image.Image:
    """Convert depth map to a colored visualization.

    Args:
        depth: Float32 depth map [0, 1]

    Returns:
        Colored depth visualization as PIL Image
    """
    depth_uint8 = (depth * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb)
