"""Deterministic texture projection with perspective, luminance overlay,
color temperature matching, noise matching, and occluder-aware masking.

Pipeline:
  1. Tile texture anchored to the BOTTOM of the wall polygon (realistic
     brick-laying direction — tiles start from the floor, not from y=0)
  2. Perspective warp using the actual polygon corners (not extremal mask
     pixels) for accurate foreshortening on angled walls
  3. Color temperature matching (LAB channel-wise mean/std transfer)
  4. Luminance overlay from original wall (preserves shadows + lighting)
  5. Noise/grain matching
  6. Feathered mask compositing onto original image (wider radius = softer)
"""

import base64
import io
import logging
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

# Default physical floor-to-ceiling wall span (mm) for scale calibration.
# Override with env STEGU_ASSUMED_WALL_HEIGHT_MM (e.g. 3000 for high ceilings).
_DEFAULT_WALL_MM = 2850


def dilate_binary_mask(
    mask: np.ndarray, iterations: int = 2, kernel_size: int = 5
) -> np.ndarray:
    """Expand a binary uint8 mask by a few pixels (cleaner object cut-outs)."""
    if not mask.any() or iterations <= 0:
        return mask
    ks = max(3, kernel_size | 1)  # must be odd and >= 3
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    for _ in range(iterations):
        pil = pil.filter(ImageFilter.MaxFilter(ks))
    return (np.array(pil) > 127).astype(np.uint8)


def erode_binary_mask(
    mask: np.ndarray, iterations: int = 2, kernel_size: int = 5
) -> np.ndarray:
    """Shrink a binary uint8 mask (inverse of dilate)."""
    if not mask.any() or iterations <= 0:
        return mask
    ks = max(3, kernel_size | 1)
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    for _ in range(iterations):
        pil = pil.filter(ImageFilter.MinFilter(ks))
    return (np.array(pil) > 127).astype(np.uint8)


def close_binary_mask(
    mask: np.ndarray, iterations: int = 1, kernel_size: int = 3
) -> np.ndarray:
    """Fill small holes / gaps in a binary mask (dilate then erode)."""
    if not mask.any() or iterations <= 0:
        return mask
    m = dilate_binary_mask(mask, iterations=iterations, kernel_size=kernel_size)
    return erode_binary_mask(m, iterations=iterations, kernel_size=kernel_size)


def estimate_wall_span_mm(
    polygon_height_px: float,
    image_height_px: int,
    image_width_px: int = 0,
) -> float:
    """Estimate real-world height (mm) represented by the polygon's vertical extent.

    Uses the FULL IMAGE dimensions (not just the polygon) to determine room
    proportions. A typical interior photograph captures a room at roughly
    eye-level, with the ceiling near the top and floor near the bottom.

    Strategy:
      - The full image height approximates the visible room height (~2850 mm)
      - The polygon's fraction of the image scales that proportionally
      - Portrait-oriented images are treated differently (phone photos)
    """
    base = float(os.environ.get("STEGU_ASSUMED_WALL_HEIGHT_MM", str(_DEFAULT_WALL_MM)))
    base = max(2000.0, min(base, 3600.0))
    if image_height_px <= 0:
        return base

    # Detect portrait vs landscape orientation for better room estimation
    aspect = (image_width_px / image_height_px) if image_width_px > 0 else 1.5
    if aspect < 1.0:
        # Portrait photo — camera sees less vertical span, so the full image
        # height likely captures a smaller portion of the room (~2200mm)
        room_mm = base * 0.80
    elif aspect > 2.0:
        # Ultra-wide panorama — the full image height captures roughly
        # a full floor-to-ceiling view
        room_mm = base
    else:
        # Standard landscape (1.3–1.8) — most common, captures ~full room
        room_mm = base

    frac = polygon_height_px / float(image_height_px)
    # The polygon selection covers this fraction of the visible room
    wall_mm = room_mm * frac
    # Clamp so a tiny polygon doesn't produce absurdly small values
    return max(800.0, min(wall_mm, base * 1.1))


def refine_mask_with_boundaries(
    mask: np.ndarray,
    wall_boundaries: dict | None,
) -> np.ndarray:
    """Trim mask using wall boundary analysis from Gemini scene measurement.

    Prevents texture from extending onto ceiling, floor, or adjacent walls
    when the user's polygon selection overshoots the actual wall surface.
    The boundary coordinates are normalised 0-1 relative to the full image.
    """
    if not wall_boundaries:
        return mask

    H, W = mask.shape
    refined = mask.copy()

    ceiling_y = wall_boundaries.get("ceilingLineY")
    if wall_boundaries.get("selectionExceedsCeiling") and ceiling_y and 0 < ceiling_y < 1:
        cut = max(0, int(ceiling_y * H))
        refined[:cut, :] = 0

    floor_y = wall_boundaries.get("floorLineY")
    if wall_boundaries.get("selectionExceedsFloor") and floor_y and 0 < floor_y < 1:
        cut = min(H, int(floor_y * H))
        refined[cut:, :] = 0

    left_x = wall_boundaries.get("leftEdgeX")
    if wall_boundaries.get("selectionExceedsLeftWall") and left_x and 0 < left_x < 1:
        cut = max(0, int(left_x * W))
        refined[:, :cut] = 0

    right_x = wall_boundaries.get("rightEdgeX")
    if wall_boundaries.get("selectionExceedsRightWall") and right_x and 0 < right_x < 1:
        cut = min(W, int(right_x * W))
        refined[:, cut:] = 0

    if not refined.any():
        logger.warning("Wall boundary refinement removed all mask pixels — keeping original mask")
        return mask

    return refined


def polygon_to_mask(
    polygon: list[dict], width: int, height: int
) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    pts = [(p["x"], p["y"]) for p in polygon]
    draw.polygon(pts, fill=255)
    return (np.array(img) > 0).astype(np.uint8)


def exclusions_to_mask(
    exclusions: list[dict],
    width: int,
    height: int,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    if not exclusions:
        return np.zeros((height, width), dtype=np.uint8)
    sx = width / canvas_w
    sy = height / canvas_h
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for r in exclusions:
        x0 = r["x"] * sx
        y0 = r["y"] * sy
        x1 = (r["x"] + r["w"]) * sx
        y1 = (r["y"] + r["h"]) * sy
        draw.rectangle([x0, y0, x1, y1], fill=255)
    return (np.array(img) > 0).astype(np.uint8)


def compute_final_mask(
    wall_mask: np.ndarray,
    occluder_mask: np.ndarray | None,
    exclusion_mask: np.ndarray | None,
) -> np.ndarray:
    final = wall_mask.copy()
    if occluder_mask is not None:
        final = final & (~occluder_mask.astype(bool)).astype(np.uint8)
    if exclusion_mask is not None:
        final = final & (~exclusion_mask.astype(bool)).astype(np.uint8)
    return final


def flood_fill_wall_region(
    wall_mask: np.ndarray, seed_mask: np.ndarray
) -> np.ndarray:
    """Find the connected wall region that overlaps with *seed_mask*.

    Uses iterative morphological dilation starting from wall pixels inside
    the seed.  Each round grows by one kernel width but only into pixels
    that are already ``wall_mask == 1``, so the expansion stops at wall
    boundaries detected by OneFormer.  Progressively larger kernels make
    this converge quickly even on large images.
    """
    connected = (wall_mask & seed_mask).astype(np.uint8)
    if not connected.any():
        return seed_mask.copy()

    prev_count = 0
    for kernel in (3, 5, 7, 11, 15, 21, 31, 51):
        for _ in range(5):
            count = int(connected.sum())
            if count == prev_count:
                return connected
            prev_count = count
            pil = Image.fromarray((connected * 255).astype(np.uint8), mode="L")
            expanded = pil.filter(ImageFilter.MaxFilter(kernel))
            connected = (np.array(expanded) > 127).astype(np.uint8) & wall_mask
    return connected


def feather_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    """Gaussian-feather a binary mask for smooth compositing edges."""
    if radius <= 0:
        return mask.astype(np.float32)
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.clip(np.array(blurred).astype(np.float32) / 255.0, 0, 1)


def edge_aware_feather_mask(
    mask: np.ndarray,
    room_image: Image.Image | None = None,
    base_radius: int = 2,
    sharp_radius: int = 1,
) -> np.ndarray:
    """Edge-aware feathering: crisp at detected image edges, soft elsewhere.

    At architectural boundaries (ceiling, wall corners) and object outlines
    (clocks, frames), the room image has strong edges — the alpha transition
    should be crisp there (sharp_radius ≈ 1 px).  Where the mask boundary
    falls on a featureless wall surface (no image edge), a slightly softer
    blend (base_radius ≈ 2–3 px) avoids aliasing without creating halos.
    """
    if base_radius <= 0:
        return mask.astype(np.float32)

    soft = feather_mask(mask, base_radius)

    if room_image is None or sharp_radius >= base_radius:
        return soft

    hard = feather_mask(mask, sharp_radius)

    H, W = mask.shape
    rm = room_image
    if rm.size != (W, H):
        rm = rm.resize((W, H), Image.LANCZOS)
    gray = rm.convert("L")

    edges = np.array(gray.filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0
    edge_pil = Image.fromarray(np.clip(edges * 255, 0, 255).astype(np.uint8), mode="L")
    edge_dilated = np.array(
        edge_pil.filter(ImageFilter.MaxFilter(5))
    ).astype(np.float32) / 255.0
    edge_influence = np.array(
        Image.fromarray(np.clip(edge_dilated * 255, 0, 255).astype(np.uint8), mode="L")
        .filter(ImageFilter.GaussianBlur(radius=2))
    ).astype(np.float32) / 255.0

    edge_weight = np.clip(edge_influence * 4.0, 0.0, 1.0)

    transition = (soft > 0.01) & (soft < 0.99)
    result = soft.copy()
    result[transition] = (
        hard[transition] * edge_weight[transition]
        + soft[transition] * (1.0 - edge_weight[transition])
    )

    return result


def compute_texture_scale(
    polygon_height_px: float,
    image_height_px: int,
    texture_height_px: int,
    meta: dict,
    analysis: dict | None = None,
    image_width_px: int = 0,
) -> float:
    """Compute how much to scale the albedo texture for realistic tiling.

    The key insight: each albedo texture's metadata now contains
    `tileRealHeightMm` — the exact real-world height (in mm) that the
    albedo image represents. This eliminates all course/plank counting.

    Scale formula:
      target_px = tileRealHeightMm × px_per_mm
      scale     = target_px / texture_height_px

    Where px_per_mm comes from CV calibration (ceiling/door detection)
    or falls back to assuming the visible wall ≈ 270cm.
    """
    # ── 1. Get the real-world height this albedo tile covers (mm) ──────
    tile_real_h_mm = float(meta.get("tileRealHeightMm", 0))

    if tile_real_h_mm <= 0:
        # Legacy fallback: compute from courses/joints
        module_h = float(meta.get("moduleHeightMm", 65))
        joint = float(meta.get("jointMm", 10))
        layout = meta.get("layoutType", "running-bond")

        if layout in ("running-bond", "stretcher-bond", "random-stone",
                       "stack-bond", "flemish-bond", "herringbone"):
            courses = int(meta.get("albedoBrickCourses", 8))
            courses = max(1, min(courses, 32))
            tile_real_h_mm = module_h * courses + joint * max(courses - 1, 0)
        else:
            planks = int(meta.get("albedoStackPlanks", 1))
            planks = max(1, min(planks, 12))
            tile_real_h_mm = module_h * planks + joint * max(planks - 1, 0)

        logger.info("Legacy tile height calc: %.0fmm (no tileRealHeightMm in metadata)", tile_real_h_mm)

    # ── 2. Determine px_per_mm ────────────────────────────────────────
    if analysis and analysis.get("px_per_cm"):
        # Best: CV-calibrated from ceiling/floor/door detection
        px_per_cm = float(analysis["px_per_cm"])
        px_per_mm = px_per_cm / 10.0
        source = f"cv-calibration (px/cm={px_per_cm:.3f})"
    else:
        # Fallback: assume visible wall selection ≈ 270cm
        wall_mm = 2700.0  # 270cm standard ceiling
        if analysis and analysis.get("wallHeightCm"):
            wall_mm = float(analysis["wallHeightCm"]) * 10.0
            wall_mm = max(1500.0, min(wall_mm, 5000.0))
        px_per_mm = polygon_height_px / wall_mm
        source = f"heuristic (wall={wall_mm:.0f}mm)"

    # ── 3. Compute scale ──────────────────────────────────────────────
    target_h_px = max(tile_real_h_mm * px_per_mm, 1.0)
    scale = target_h_px / float(texture_height_px)

    # ── 4. Apply per-product multiplier if set ────────────────────────
    mult = float(meta.get("textureScaleMultiplier", 1.0))
    env_m = os.environ.get("STEGU_TEXTURE_SCALE_MULTIPLIER")
    if env_m:
        try:
            mult *= float(env_m)
        except ValueError:
            pass
    mult = max(0.35, min(mult, 2.5))
    final_scale = scale * mult

    # ── 5. Log for debugging ──────────────────────────────────────────
    module_h = float(meta.get("moduleHeightMm", 80))
    brick_px = module_h * px_per_mm
    logger.info(
        "TEXTURE SCALE: %.4f (×%.2f mult) — tile=%.0fmm→%.0fpx, "
        "1 module=%dmm→%.0fpx, source=%s, product=%s",
        final_scale, mult, tile_real_h_mm, target_h_px,
        int(module_h), brick_px, source, meta.get("name", "?"),
    )

    return final_scale


# ── Perspective helpers ───────────────────────────────────────────────────────


def _polygon_to_quad(
    polygon: list[dict],
) -> list[tuple[float, float]] | None:
    """Extract 4 representative corner points from the user polygon.

    Uses the 4 extremal-point heuristic (top-left = min x+y, etc.) applied
    to the actual polygon vertices rather than to noisy mask pixels, which
    gives a much more accurate perspective estimation for user-drawn walls.
    """
    if not polygon or len(polygon) < 4:
        return None
    pts = np.array([[p["x"], p["y"]] for p in polygon])
    s = pts.sum(axis=1)       # x+y → TL=min, BR=max
    d = pts[:, 0] - pts[:, 1]  # x-y → TR=max, BL=min

    tl = tuple(pts[s.argmin()])
    br = tuple(pts[s.argmax()])
    tr = tuple(pts[d.argmax()])
    bl = tuple(pts[d.argmin()])

    # Degenerate: all 4 are the same point
    if len({tl, tr, br, bl}) < 3:
        return None
    return [tl, tr, br, bl]


def _find_quad_corners(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Fallback: extract 4 corner points from binary mask extremal pixels."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)

    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]

    tl = tuple(pts[s.argmin()])
    br = tuple(pts[s.argmax()])
    tr = tuple(pts[d.argmax()])
    bl = tuple(pts[d.argmin()])

    return [tl, tr, br, bl]


def _perspective_coeffs(
    src: list[tuple[float, float]], dst: list[tuple[float, float]]
) -> list[float]:
    """Compute 8 perspective transform coefficients for PIL.

    Maps src points (in output image) to dst points (in source image).
    """
    A = []
    B = []
    for (xo, yo), (xi, yi) in zip(src, dst):
        A.append([xo, yo, 1, 0, 0, 0, -xi * xo, -xi * yo])
        A.append([0, 0, 0, xo, yo, 1, -yi * xo, -yi * yo])
        B.append(xi)
        B.append(yi)
    return np.linalg.solve(np.array(A), np.array(B)).tolist()


def _quad_is_rectangular(
    quad: list[tuple[float, float]], threshold: float = 0.04
) -> bool:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bw < 10 or bh < 10:
        return True
    diag = (bw**2 + bh**2) ** 0.5
    rect = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]
    for q, r in zip(quad, rect):
        if ((q[0] - r[0]) ** 2 + (q[1] - r[1]) ** 2) ** 0.5 / diag > threshold:
            return False
    return True


def _perspective_warp(
    tiled: np.ndarray,
    mask: np.ndarray,
    polygon: list[dict] | None = None,
) -> np.ndarray:
    """Apply perspective correction to the tiled texture.

    Prefers actual polygon corners over extremal mask pixels for accuracy.
    """
    H, W = mask.shape

    # Prefer polygon-derived quad; fall back to mask-derived quad
    quad = None
    if polygon:
        quad = _polygon_to_quad(polygon)
    if quad is None:
        quad = _find_quad_corners(mask)

    if quad is None or _quad_is_rectangular(quad):
        return tiled

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    rect = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]

    try:
        coeffs = _perspective_coeffs(quad, rect)
    except np.linalg.LinAlgError:
        return tiled

    tiled_pil = Image.fromarray(
        np.clip(tiled * 255, 0, 255).astype(np.uint8)
    )
    warped_pil = tiled_pil.transform(
        (W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )
    return np.array(warped_pil).astype(np.float32) / 255.0


# ── Color / lighting helpers ──────────────────────────────────────────────────


def _srgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB float [0,1] array (H,W,3) to CIE-Lab."""
    linear = np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    x = linear[:, :, 0] * 0.4124564 + linear[:, :, 1] * 0.3575761 + linear[:, :, 2] * 0.1804375
    y = linear[:, :, 0] * 0.2126729 + linear[:, :, 1] * 0.7151522 + linear[:, :, 2] * 0.0721750
    z = linear[:, :, 0] * 0.0193339 + linear[:, :, 1] * 0.1191920 + linear[:, :, 2] * 0.9503041
    x /= 0.95047; z /= 1.08883

    def _f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE-Lab float back to sRGB float [0,1]."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    fx = lab[:, :, 1] / 500.0 + fy
    fz = fy - lab[:, :, 2] / 200.0
    eps = 0.008856

    def _finv(t):
        return np.where(t ** 3 > eps, t ** 3, (t - 16.0 / 116.0) / 7.787)

    x = _finv(fx) * 0.95047
    y = _finv(fy)
    z = _finv(fz) * 1.08883
    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
    linear = np.stack([r, g, b], axis=-1)
    linear = np.clip(linear, 0, None)
    srgb = np.where(linear > 0.0031308,
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
                    12.92 * linear)
    return np.clip(srgb, 0, 1)


def _color_temperature_match(
    tiled: np.ndarray, room_arr: np.ndarray, mask: np.ndarray, strength: float = 0.42
) -> np.ndarray:
    """Transfer color statistics from wall to texture in LAB color space."""
    wall_px = room_arr[mask > 0]
    tex_px = tiled[mask > 0]
    if len(wall_px) < 100 or len(tex_px) < 100:
        return tiled

    tex_lab = _srgb_to_lab(tiled)
    room_lab = _srgb_to_lab(room_arr)

    wall_lab = room_lab[mask > 0]
    tex_lab_px = tex_lab[mask > 0]

    adjusted = tex_lab.copy()
    for ch in range(3):
        tex_mean = tex_lab_px[:, ch].mean()
        tex_std = max(tex_lab_px[:, ch].std(), 0.01)
        wall_mean = wall_lab[:, ch].mean()
        wall_std = max(wall_lab[:, ch].std(), 0.01)
        shifted = (tex_lab[:, :, ch] - tex_mean) * (wall_std / tex_std) + wall_mean
        adjusted[:, :, ch] = tex_lab[:, :, ch] * (1 - strength) + shifted * strength

    return _lab_to_srgb(adjusted)


def _luminance_overlay(
    tiled: np.ndarray, room_arr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Blend global room lighting with local contact-shadow detail.

    Three-scale decomposition:
      • Coarse blur (radius 35) — directional light gradient from windows,
        overall ambient exposure across the wall.
      • Fine blur (radius 4) — contact shadows where furniture meets the
        wall, shelf undersides, lamp cones.
      • Micro blur (radius 2) — surface-level modulation from the original
        wall that preserves subtle texture / paint irregularity.
    """
    gray = 0.299 * room_arr[:, :, 0] + 0.587 * room_arr[:, :, 1] + 0.114 * room_arr[:, :, 2]

    wall_vals = gray[mask > 0]
    if len(wall_vals) == 0:
        return tiled
    wall_mean = wall_vals.mean()
    if wall_mean < 0.01:
        return tiled

    gray_pil = Image.fromarray((gray * 255).astype(np.uint8), mode="L")

    coarse = np.array(
        gray_pil.filter(ImageFilter.GaussianBlur(radius=35))
    ).astype(np.float32) / 255.0
    fine = np.array(
        gray_pil.filter(ImageFilter.GaussianBlur(radius=4))
    ).astype(np.float32) / 255.0
    micro = np.array(
        gray_pil.filter(ImageFilter.GaussianBlur(radius=2))
    ).astype(np.float32) / 255.0

    light_mean = coarse[mask > 0].mean()
    lum_global = coarse / (light_mean + 1e-6)
    lum_global = np.clip(lum_global, 0.22, 2.2)

    local = fine / (coarse + 1e-4)
    local = np.clip(local, 0.60, 1.40)
    lm = float(local[mask > 0].mean())
    if lm > 1e-6:
        local = 0.75 + 0.25 * (local / lm)

    micro_detail = micro / (fine + 1e-4)
    micro_detail = np.clip(micro_detail, 0.85, 1.15)
    mm = float(micro_detail[mask > 0].mean())
    if mm > 1e-6:
        micro_detail = 0.92 + 0.08 * (micro_detail / mm)

    lum_factor = lum_global * local * micro_detail
    lum_factor = np.clip(lum_factor, 0.20, 2.3)

    return np.clip(tiled * lum_factor[:, :, np.newaxis], 0, 1)


def _noise_match(
    blended: np.ndarray, room_arr: np.ndarray, strength: float = 0.35
) -> np.ndarray:
    """Transfer grain/noise from the original photo onto the result."""
    H, W = room_arr.shape[:2]
    if H < 80 or W < 80:
        return blended

    gray = 0.299 * room_arr[:, :, 0] + 0.587 * room_arr[:, :, 1] + 0.114 * room_arr[:, :, 2]
    gray_pil = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    gray_smooth = np.array(
        gray_pil.filter(ImageFilter.GaussianBlur(radius=2))
    ).astype(np.float32) / 255.0
    noise_level = float(np.std(gray - gray_smooth))

    room_blur = np.zeros_like(room_arr)
    for c in range(3):
        ch = Image.fromarray((room_arr[:, :, c] * 255).astype(np.uint8))
        ch = ch.filter(ImageFilter.GaussianBlur(radius=1.5))
        room_blur[:, :, c] = np.array(ch).astype(np.float32) / 255.0

    photo_noise = room_arr - room_blur

    rng = np.random.default_rng(42)
    synth_noise = rng.normal(0, noise_level * 0.5, blended.shape).astype(np.float32)

    combined = photo_noise * 0.6 + synth_noise * 0.4
    return np.clip(blended + combined * strength, 0, 1)


# ── Main projection ───────────────────────────────────────────────────────────


def project_texture(
    room: Image.Image,
    mask: np.ndarray,
    texture: Image.Image,
    meta: dict | None = None,
    alpha: float = 0.995,
    feather_radius: int = 2,
    polygon: list[dict] | None = None,
    analysis: dict | None = None,
) -> Image.Image:
    """Project a tiled texture onto the masked wall region.

    The tiling is anchored to the BOTTOM of the wall polygon (y_max of mask)
    so that bricks appear to start from the floor — just as they would in a
    real installation.  The left anchor is the leftmost mask pixel (x_min).

    When polygon is provided the perspective warp uses the actual polygon
    corners rather than estimating them from the binary mask, which gives
    accurate foreshortening for angled walls.
    """
    H, W = mask.shape
    room_resized = room.resize((W, H), Image.LANCZOS)
    room_arr = np.array(room_resized).astype(np.float32) / 255.0

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return room_resized

    poly_h = float(ys.max() - ys.min())

    if meta:
        tex_scale = compute_texture_scale(poly_h, H, texture.height, meta, analysis=analysis, image_width_px=W)
    else:
        wall_mm = estimate_wall_span_mm(poly_h, H, W)
        px_per_mm = poly_h / wall_mm
        target_tile_h = max(400.0 * px_per_mm, 1.0)
        tex_scale = target_tile_h / float(texture.height)

    scaled_w = max(int(texture.width * tex_scale), 1)
    scaled_h = max(int(texture.height * tex_scale), 1)

    tex_resized = texture.resize((scaled_w, scaled_h), Image.LANCZOS)
    tex_arr = np.array(tex_resized).astype(np.float32) / 255.0

    # ── 1. Tile anchored to polygon bottom-left ───────────────────────────
    # This places brick courses so the BOTTOM seam aligns with the floor line,
    # matching how bricks are actually laid in construction.
    y_ref = int(ys.max())   # bottom of the wall (floor level)
    x_ref = int(xs.min())   # left edge of the wall

    # First tile row / col that covers y=0 / x=0 while keeping y_ref on seam
    k_y = math.ceil(y_ref / scaled_h) if scaled_h > 0 else 0
    k_x = math.ceil(x_ref / scaled_w) if scaled_w > 0 else 0
    y_start = y_ref - k_y * scaled_h   # ≤ 0
    x_start = x_ref - k_x * scaled_w   # ≤ 0

    tiled = np.zeros_like(room_arr)
    y = y_start
    while y < H:
        x = x_start
        while x < W:
            src_y0 = max(y, 0)
            src_x0 = max(x, 0)
            src_y1 = min(y + scaled_h, H)
            src_x1 = min(x + scaled_w, W)
            if src_y1 > src_y0 and src_x1 > src_x0:
                ty0 = src_y0 - y
                tx0 = src_x0 - x
                tiled[src_y0:src_y1, src_x0:src_x1] = tex_arr[
                    ty0: ty0 + (src_y1 - src_y0),
                    tx0: tx0 + (src_x1 - src_x0),
                ]
            x += scaled_w
        y += scaled_h

    # ── 2. Perspective warp (polygon-based) ──────────────────────────────
    try:
        tiled = _perspective_warp(tiled, mask, polygon=polygon)
    except Exception as exc:
        logger.debug("Perspective warp skipped: %s", exc)

    # ── 3. Color temperature matching ─────────────────────────────────────
    tiled = _color_temperature_match(tiled, room_arr, mask)

    # ── 4. Luminance overlay ──────────────────────────────────────────────
    tiled = _luminance_overlay(tiled, room_arr, mask)

    # ── 5. Almost full texture (low alpha was causing ghosting over furniture)
    blended = tiled * alpha + room_arr * (1.0 - alpha)

    # ── 6. Noise matching ─────────────────────────────────────────────────
    blended = _noise_match(blended, room_arr)

    # ── 7. Edge-aware feathered mask compositing ─────────────────────────
    mask_f = edge_aware_feather_mask(
        mask, room_image=room_resized, base_radius=feather_radius, sharp_radius=1,
    )
    mask_3d = mask_f[:, :, np.newaxis]

    result = room_arr * (1 - mask_3d) + blended * mask_3d
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ── Geometry guide + wall mask generators (for AI pipeline v2) ────────────


def generate_geometry_guide(
    room: Image.Image,
    mask: np.ndarray,
    texture: Image.Image,
    meta: dict | None = None,
    polygon: list[dict] | None = None,
    analysis: dict | None = None,
) -> Image.Image:
    """Generate a clean geometry guide image for the AI renderer.

    The guide shows the exact tiled texture (correct scale, spacing,
    perspective) on the original room photo, masked to the wall area only.
    NO color matching, NO luminance overlay, NO noise — just the raw
    texture laid out deterministically on the wall.

    This is the "locked blueprint" that the AI must not alter.
    """
    H, W = mask.shape
    room_resized = room.resize((W, H), Image.LANCZOS)
    room_arr = np.array(room_resized).astype(np.float32) / 255.0

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return room_resized

    poly_h = float(ys.max() - ys.min())

    if meta:
        tex_scale = compute_texture_scale(
            poly_h, H, texture.height, meta, analysis=analysis, image_width_px=W
        )
    else:
        wall_mm = estimate_wall_span_mm(poly_h, H, W)
        px_per_mm = poly_h / wall_mm
        target_tile_h = max(400.0 * px_per_mm, 1.0)
        tex_scale = target_tile_h / float(texture.height)

    scaled_w = max(int(texture.width * tex_scale), 1)
    scaled_h = max(int(texture.height * tex_scale), 1)

    tex_resized = texture.resize((scaled_w, scaled_h), Image.LANCZOS)
    tex_arr = np.array(tex_resized).astype(np.float32) / 255.0

    # Tile anchored to bottom-left (same as project_texture)
    y_ref = int(ys.max())
    x_ref = int(xs.min())
    k_y = math.ceil(y_ref / scaled_h) if scaled_h > 0 else 0
    k_x = math.ceil(x_ref / scaled_w) if scaled_w > 0 else 0
    y_start = y_ref - k_y * scaled_h
    x_start = x_ref - k_x * scaled_w

    tiled = np.zeros_like(room_arr)
    y = y_start
    while y < H:
        x = x_start
        while x < W:
            src_y0 = max(y, 0)
            src_x0 = max(x, 0)
            src_y1 = min(y + scaled_h, H)
            src_x1 = min(x + scaled_w, W)
            if src_y1 > src_y0 and src_x1 > src_x0:
                ty0 = src_y0 - y
                tx0 = src_x0 - x
                tiled[src_y0:src_y1, src_x0:src_x1] = tex_arr[
                    ty0: ty0 + (src_y1 - src_y0),
                    tx0: tx0 + (src_x1 - src_x0),
                ]
            x += scaled_w
        y += scaled_h

    # Perspective warp (same as project_texture)
    try:
        tiled = _perspective_warp(tiled, mask, polygon=polygon)
    except Exception:
        pass

    # Composite: texture on wall, original everywhere else
    # Hard mask — no feathering, no blending
    mask_3d = mask[:, :, np.newaxis].astype(np.float32)
    result = room_arr * (1.0 - mask_3d) + tiled * mask_3d
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def generate_wall_mask_image(
    mask: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Convert a binary numpy mask to a clean white-on-black PIL Image.

    White = wall (editable area), Black = everything else (immutable).
    If target_size is (W, H), resize to match the original image dimensions.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.NEAREST)
    return img


# ── Post-Gemini masked composite ─────────────────────────────────────────────


def masked_composite(
    original: Image.Image,
    refined: Image.Image,
    mask: np.ndarray,
    feather_radius: int = 3,
) -> Image.Image:
    W, H = original.size

    if refined.size != (W, H):
        refined = refined.resize((W, H), Image.LANCZOS)

    if mask.shape != (H, W):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask = (np.array(mask_pil) > 127).astype(np.uint8)

    orig_arr = np.array(original).astype(np.float32) / 255.0
    ref_arr = np.array(refined).astype(np.float32) / 255.0

    blend = edge_aware_feather_mask(
        mask, room_image=original, base_radius=feather_radius, sharp_radius=1,
    )
    blend_3d = blend[:, :, np.newaxis]

    result = orig_arr * (1.0 - blend_3d) + ref_arr * blend_3d
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ── Mask visualisation for AI analysis ───────────────────────────────────────


def render_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 140, 0),
    alpha: float = 0.45,
    exclusion_mask: np.ndarray | None = None,
    max_long_side: int = 1280,
) -> Image.Image:
    """Draw the wall mask as a coloured overlay on the original image.

    The result is sent to Gemini so it can visually inspect where the
    texture will be placed.  A secondary exclusion mask (already-excluded
    objects) is drawn in red so Gemini can see what was already cut out.

    Parameters
    ----------
    color:          RGB fill for the wall region (default amber/orange)
    alpha:          overlay opacity 0–1 (default 0.45)
    exclusion_mask: already-excluded objects drawn in red
    max_long_side:  resize large images before sending to Gemini
    """
    W, H = image.size

    if mask.shape != (H, W):
        mpil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask = (np.array(mpil.resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)

    img_arr = np.array(image.convert("RGB")).astype(np.float32)

    overlay = img_arr.copy()
    wall_pixels = mask > 0
    for i, c in enumerate(color):
        overlay[:, :, i][wall_pixels] = c

    if exclusion_mask is not None:
        if exclusion_mask.shape != (H, W):
            epil = Image.fromarray((exclusion_mask * 255).astype(np.uint8), mode="L")
            exclusion_mask = (
                np.array(epil.resize((W, H), Image.NEAREST)) > 127
            ).astype(np.uint8)
        exc_pixels = exclusion_mask > 0
        exc_color = (210, 40, 40)
        for i, c in enumerate(exc_color):
            overlay[:, :, i][exc_pixels] = c

    blended = np.clip(
        img_arr * (1.0 - alpha) + overlay * alpha, 0, 255
    ).astype(np.uint8)

    result = Image.fromarray(blended)

    long_side = max(W, H)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        result = result.resize(
            (int(W * scale), int(H * scale)), Image.LANCZOS
        )

    return result


# ── Utilities ─────────────────────────────────────────────────────────────────


def mask_to_b64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_mask_b64(data: str, width: int, height: int) -> np.ndarray:
    raw = data.split(",", 1)[-1] if "," in data else data
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def image_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()
