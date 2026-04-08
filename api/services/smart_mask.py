"""Smart wall surface detection using color analysis and edge detection.

When OneFormer/SAM2 inference is unavailable, this provides intelligent wall
masking based on color similarity and edge boundaries within the user-selected
polygon region.

Key improvements over the naive single-sample approach:
  - Samples color from MULTIPLE points in the lower portion of the polygon
    (ceiling is at the top, so we avoid sampling there)
  - Excludes high-edge pixels from the sample set (only smooth wall areas)
  - Uses slightly more lenient color distance (2.5 σ vs 2.2 σ)
  - Wider morphological operations for cleaner wall/occluder separation
"""

import numpy as np
from PIL import Image, ImageFilter


def _edge_mask(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Compute edge magnitude from the image (0-1 float)."""
    img = image.resize((width, height), Image.LANCZOS).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(np.float32) / 255.0
    blurred = Image.fromarray((edge_arr * 255).astype(np.uint8), mode="L")
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(blurred).astype(np.float32) / 255.0


def detect_wall_surface(
    image: Image.Image,
    poly_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect wall surface and occluders within user polygon.

    Strategy:
    1. Compute edge magnitude — smooth (low-edge) areas are candidate wall.
    2. Sample color only from the LOWER 75 % of the polygon bounding box
       and only from low-edge pixels to avoid objects and potential ceiling
       at the top of the polygon.
    3. Mark all polygon pixels whose color is within 2.5 σ of the sample
       mean as wall, further constrained to low-edge areas.
    4. Clean up with morphological open/close and derive occluder mask.

    Returns (wall_mask, occluder_mask) as binary uint8 arrays.
    """
    H, W = poly_mask.shape
    img = image.resize((W, H), Image.LANCZOS) if image.size != (W, H) else image
    img_arr = np.array(img).astype(np.float32) / 255.0

    ys, xs = np.where(poly_mask > 0)
    if len(ys) < 100:
        return poly_mask.copy(), np.zeros_like(poly_mask)

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # ── edge magnitude ────────────────────────────────────────────────────
    edges = _edge_mask(img, W, H)
    low_edge = (edges < 0.10).astype(np.uint8)

    # ── sample only from lower 75 % of polygon (avoid ceiling junction) ──
    y_sample_start = y_min + int((y_max - y_min) * 0.25)
    sample_region = np.zeros_like(poly_mask)
    sample_region[y_sample_start:y_max, x_min:x_max] = 1
    sample_region = sample_region & poly_mask & low_edge

    # If too few pixels, expand search
    if sample_region.sum() < 20:
        # Try full lower half without edge constraint
        sample_region = np.zeros_like(poly_mask)
        sample_region[y_sample_start:y_max, x_min:x_max] = 1
        sample_region = sample_region & poly_mask

    if sample_region.sum() < 10:
        # Last resort: centred patch
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        h_span = max(int((y_max - y_min) * 0.15), 10)
        w_span = max(int((x_max - x_min) * 0.15), 10)
        sample_region = np.zeros_like(poly_mask)
        sample_region[
            max(cy - h_span, 0): min(cy + h_span, H),
            max(cx - w_span, 0): min(cx + w_span, W),
        ] = 1
        sample_region = sample_region & poly_mask

    sample_pixels = img_arr[sample_region > 0]
    if len(sample_pixels) < 5:
        return poly_mask.copy(), np.zeros_like(poly_mask)

    mean_c = sample_pixels.mean(axis=0)
    std_c = np.maximum(sample_pixels.std(axis=0), 6.0)

    # ── color distance (slightly more lenient: 2.5 σ) ────────────────────
    diff = (img_arr - mean_c) / std_c
    dist = np.sqrt((diff ** 2).sum(axis=2))
    wall_color = (dist < 2.5).astype(np.uint8) & poly_mask

    # ── remove strong edges ───────────────────────────────────────────────
    strong_edges = (edges > 0.12).astype(np.uint8) & poly_mask
    wall_color = wall_color & (~strong_edges.astype(bool)).astype(np.uint8)

    # ── morphological clean-up (fill holes, then trim noise) ─────────────
    wall_pil = Image.fromarray(wall_color * 255, mode="L")
    wall_pil = wall_pil.filter(ImageFilter.MaxFilter(7))   # dilate — fill small gaps
    wall_pil = wall_pil.filter(ImageFilter.MinFilter(7))   # erode  — remove noise
    wall_pil = wall_pil.filter(ImageFilter.MinFilter(5))   # erode  — shrink slightly
    wall_pil = wall_pil.filter(ImageFilter.MaxFilter(5))   # dilate — restore size
    wall_clean = (np.array(wall_pil) > 127).astype(np.uint8) & poly_mask

    # ── occluder mask (polygon minus wall) ───────────────────────────────
    occ_raw = poly_mask.astype(bool) & ~wall_clean.astype(bool)
    occ_pil = Image.fromarray(occ_raw.astype(np.uint8) * 255, mode="L")
    occ_pil = occ_pil.filter(ImageFilter.MinFilter(3))
    occ_pil = occ_pil.filter(ImageFilter.MaxFilter(7))
    occluder_mask = (np.array(occ_pil) > 127).astype(np.uint8) & poly_mask

    # Discard very tiny occluder blobs (< 0.3 % of polygon area)
    min_occ_pixels = int(poly_mask.sum() * 0.003)
    if occluder_mask.sum() < min_occ_pixels:
        occluder_mask = np.zeros_like(poly_mask)

    wall_mask = poly_mask & (~occluder_mask.astype(bool)).astype(np.uint8)

    return wall_mask, occluder_mask


def trim_architectural_boundaries(
    image: Image.Image,
    mask: np.ndarray,
    poly_mask: np.ndarray,
) -> np.ndarray:
    """Tighten mask at ceiling line and where the wall meets windows / open sides.

    Combines luminance analysis with vertical-gradient edge detection so the
    ceiling-wall junction is found even when the wall itself is light-coloured.
    The wall's own colour (sampled from the lower half of the polygon) serves
    as a reference: anything significantly brighter + uniform above is ceiling.
    """
    H, W = mask.shape
    if not mask.any():
        return mask

    img = image.resize((W, H), Image.LANCZOS) if image.size != (W, H) else image
    arr = np.array(img).astype(np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    ys, xs = np.where(poly_mask > 0)
    if len(ys) < 16:
        return mask

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    poly_h = y_max - y_min + 1
    poly_w = x_max - x_min + 1

    out = mask.copy().astype(np.uint8)

    # Vertical gradient magnitude — strong values indicate a horizontal edge
    gy = np.zeros_like(gray)
    if H > 1:
        gy[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])

    # Reference: wall colour from the lower half of the polygon
    y_wall_start = y_min + int(poly_h * 0.50)
    wall_sample = (poly_mask > 0).copy()
    wall_sample[:y_wall_start, :] = 0
    wall_pixels = gray[wall_sample > 0]
    wall_lum = float(wall_pixels.mean()) if len(wall_pixels) > 20 else 0.5

    # ── Top: ceiling band detection ────────────────────────────────────────
    y_scan_end = min(int(y_min + poly_h * 0.40), H - 1)
    non_ceiling_run = 0
    for y in range(y_min, y_scan_end + 1):
        row_hit = (poly_mask[y, :] > 0) & (out[y, :] > 0)
        n = int(row_hit.sum())
        if n < 5:
            non_ceiling_run += 1
            if non_ceiling_run >= 4:
                break
            continue
        vals = gray[y, :][row_hit]
        mu, sig = float(vals.mean()), float(vals.std())

        is_bright_uniform = mu > 0.78 and sig < 0.08
        is_brighter_than_wall = mu > wall_lum + 0.08 and sig < 0.10

        if is_bright_uniform or is_brighter_than_wall:
            out[y, row_hit] = 0
            non_ceiling_run = 0
        else:
            non_ceiling_run += 1
            if non_ceiling_run >= 3:
                break

    # ── Sides: bright, uniform columns (window / adjacent wall) ───────────
    side_w = max(int(poly_w * 0.06), 6)
    for xa, xb, reverse in (
        (x_min, min(x_min + side_w, W), False),
        (max(x_max - side_w + 1, 0), min(x_max + 1, W), True),
    ):
        non_wall_run = 0
        rng = range(xb - 1, xa - 1, -1) if reverse else range(xa, xb)
        for x in rng:
            col_hit = (poly_mask[:, x] > 0) & (out[:, x] > 0)
            n = int(col_hit.sum())
            if n < 6:
                non_wall_run += 1
                if non_wall_run >= 3:
                    break
                continue
            vals = gray[:, x][col_hit]
            mu, sig = float(vals.mean()), float(vals.std())

            is_bright_uniform = mu > 0.85 and sig < 0.06
            is_much_brighter = mu > wall_lum + 0.15 and sig < 0.08

            if is_bright_uniform or is_much_brighter:
                out[:, x] = np.where(col_hit, 0, out[:, x])
                non_wall_run = 0
            else:
                non_wall_run += 1
                if non_wall_run >= 3:
                    break

    return out
