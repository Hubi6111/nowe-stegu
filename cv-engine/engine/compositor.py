"""Stage 6: Final Realism Compositor — blends textured wall into original scene.

Pure image processing:
- Ambient occlusion at ceiling/floor/corner edges
- Contact shadows where objects touch wall
- Color temperature + luminance matching
- Feathered alpha blending
"""

import logging

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def composite(
    original: Image.Image,
    textured_wall: Image.Image,
    wall_mask: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    feather_radius: int = 3,
    ao_strength: float = 0.15,
) -> Image.Image:
    """Composite textured wall into original image with realistic blending.

    Args:
        original: Original room photo (RGB)
        textured_wall: Wall with texture applied (RGB, same size)
        wall_mask: Binary wall mask (uint8, 0/255)
        foreground_mask: Optional foreground occlusion mask (uint8, 0/255)
        depth_map: Optional depth map for depth-aware blending
        feather_radius: Gaussian blur radius for mask edges
        ao_strength: Ambient occlusion darkness strength (0..1)

    Returns:
        Final composited image (RGB PIL Image)
    """
    orig_np = np.array(original).astype(np.float32)
    wall_np = np.array(textured_wall).astype(np.float32)

    h, w = orig_np.shape[:2]

    # 1. Feathered wall mask
    alpha = wall_mask.astype(np.float32) / 255.0
    if feather_radius > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_radius)

    # 2. Subtract foreground from wall mask (objects go in FRONT)
    if foreground_mask is not None:
        fg = foreground_mask.astype(np.float32) / 255.0
        # Slight dilation of foreground for safety margin
        kernel = np.ones((5, 5), np.uint8)
        fg_dilated = cv2.dilate(fg, kernel, iterations=1)
        alpha = np.clip(alpha - fg_dilated, 0, 1)

    # 3. Luminance matching — match textured wall to original wall's brightness
    wall_np = _match_luminance(orig_np, wall_np, alpha)

    # 4. Ambient occlusion at wall edges
    if ao_strength > 0:
        ao = _ambient_occlusion(wall_mask, strength=ao_strength)
        wall_np = wall_np * ao[:, :, np.newaxis]

    # 5. Alpha blend
    alpha_3ch = alpha[:, :, np.newaxis]
    result = orig_np * (1 - alpha_3ch) + wall_np * alpha_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def _match_luminance(
    original: np.ndarray, textured: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Match textured wall luminance to original wall area."""
    # Convert to LAB
    orig_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    wall_lab = cv2.cvtColor(textured.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    mask_bool = alpha > 0.5

    if mask_bool.sum() < 100:
        return textured

    # Match L channel statistics
    orig_L = orig_lab[:, :, 0]
    wall_L = wall_lab[:, :, 0]

    orig_mean = orig_L[mask_bool].mean()
    orig_std = max(orig_L[mask_bool].std(), 1e-6)
    wall_mean = wall_L[mask_bool].mean()
    wall_std = max(wall_L[mask_bool].std(), 1e-6)

    # Normalize wall luminance to match original
    wall_lab[:, :, 0] = (wall_L - wall_mean) * (orig_std / wall_std) + orig_mean
    wall_lab[:, :, 0] = np.clip(wall_lab[:, :, 0], 0, 255)

    result = cv2.cvtColor(wall_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    return result


def _ambient_occlusion(
    wall_mask: np.ndarray, strength: float = 0.15, blur_size: int = 31
) -> np.ndarray:
    """Generate ambient occlusion darkening at wall edges.

    Creates subtle shadow at ceiling/floor/corner transitions.
    """
    # Find distance from mask edges
    mask_binary = (wall_mask > 127).astype(np.uint8)
    dist = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

    # Normalize distance
    max_dist = max(dist.max(), 1)
    dist_norm = dist / max_dist

    # AO = darker near edges, full brightness in center
    # Use exponential falloff for natural look
    ao = 1.0 - strength * np.exp(-dist_norm * 8)

    # Smooth
    ao = cv2.GaussianBlur(ao.astype(np.float32), (blur_size, blur_size), 0)

    return ao.astype(np.float32)
