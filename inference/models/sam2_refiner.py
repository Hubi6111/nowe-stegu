"""SAM 2 mask edge refinement with polygon-based point prompts.

Key improvement over the original mask-only approach:
  - Erodes the coarse mask to find deep interior points (safe wall areas,
    away from objects like clocks and paintings that sit on the wall).
  - Uses those eroded interior points as POSITIVE prompts.
  - Uses image-corner points as NEGATIVE prompts.
  - Requests multimask_output=True and picks the best mask by a combined
    IoU + SAM2-score metric, with a penalty for masks that expand outside
    the original polygon.
This makes SAM2 behave as a proper segmenter (click-on-wall → segment wall)
rather than a pure edge-smoother, so foreground objects are naturally excluded.
"""

import logging
import os
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────


def _sample_interior_points(
    mask: np.ndarray, n_points: int = 7
) -> list[list[int]]:
    """Return n_points from the deeply-eroded interior of the mask.

    Iteratively applies MinFilter until ≥5 % of the original area remains,
    then samples uniformly.  Returns an empty list if the mask is tiny.
    """
    if mask.sum() < 100:
        return []

    threshold = max(1, int(mask.sum() * 0.05))
    current_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    for _ in range(8):
        candidate = current_pil.filter(ImageFilter.MinFilter(7))
        if (np.array(candidate) > 127).sum() < threshold:
            break
        current_pil = candidate

    interior = (np.array(current_pil) > 127).astype(np.uint8)
    ys, xs = np.where(interior > 0)
    if len(ys) == 0:
        return []

    n = min(n_points, len(ys))
    indices = np.round(np.linspace(0, len(ys) - 1, n)).astype(int)
    return [[int(xs[i]), int(ys[i])] for i in indices]


def _pick_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    poly_mask: np.ndarray,
) -> np.ndarray:
    """Select the SAM2 output mask that best covers the polygon without over-expanding."""
    poly_bool = poly_mask.astype(bool)
    poly_area = float(poly_bool.sum())

    best_idx = int(scores.argmax())
    best_quality = -1.0

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_bool = mask.astype(bool)
        mask_area = float(mask_bool.sum())

        if poly_area > 0:
            iou = float((mask_bool & poly_bool).sum()) / max(
                float((mask_bool | poly_bool).sum()), 1.0
            )
            # Penalise masks that spill far outside the user polygon
            outside_ratio = float((mask_bool & ~poly_bool).sum()) / max(
                mask_area, 1.0
            )
        else:
            iou = 0.0
            outside_ratio = 0.0

        quality = float(score) * 0.35 + iou * 0.50 - outside_ratio * 0.15
        if quality > best_quality:
            best_quality = quality
            best_idx = i

    return masks[best_idx].astype(np.uint8)


def _polygon_boundary_negative_points(
    polygon: list[dict], W: int, H: int
) -> list[list[int]]:
    """Negative prompts: just under ceiling line + outside left/right of polygon.

    Pushes SAM2 away from the ceiling strip and from spill past vertical wall ends
    (windows, corners) while keeping positives on the actual wall.
    """
    if not polygon or len(polygon) < 3:
        return []

    xs = [float(p["x"]) for p in polygon]
    ys = [float(p["y"]) for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_y = max(y_max - y_min, 1.0)
    span_x = max(x_max - x_min, 1.0)

    pts: list[list[int]] = []

    # Along top edge of ROI, slightly inside the image (below y_min)
    y_top = int(np.clip(y_min + max(4.0, span_y * 0.03), 0, H - 1))
    n_top = 11
    for i in range(n_top):
        t = (i + 0.5) / n_top
        x = int(np.clip(x_min + t * span_x, 0, W - 1))
        pts.append([x, y_top])

    # Outside left / right of bbox (typical window or return wall)
    inset = max(10.0, span_x * 0.04)
    y_mid = (y_min + y_max) / 2.0
    n_side = 5
    for j in range(n_side):
        dy = (j - (n_side - 1) / 2.0) * (span_y * 0.22 / max(n_side - 1, 1))
        yy = int(np.clip(y_mid + dy, 0, H - 1))
        xl = int(np.clip(x_min - inset, 0, W - 1))
        xr = int(np.clip(x_max + inset, 0, W - 1))
        pts.append([xl, yy])
        pts.append([xr, yy])

    return pts


# ── refiner ──────────────────────────────────────────────────────────────────


class SAM2Refiner:
    def __init__(self, cache_dir: str):
        self.predictor = None

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            checkpoint = self._resolve_checkpoint(cache_dir)
            if checkpoint is None:
                logger.warning(
                    "SAM 2 checkpoint not found in %s. "
                    "Download sam2.1_hiera_large.pt into that directory.",
                    cache_dir,
                )
                return

            device_order: list[str] = []
            if torch.backends.mps.is_available():
                device_order.append("mps")
            if torch.cuda.is_available():
                device_order.append("cuda")
            device_order.append("cpu")

            last_err: Exception | None = None
            self.device = "cpu"
            for dev in device_order:
                try:
                    logger.info("Loading SAM 2 on %s …", dev)
                    sam2_model = build_sam2(
                        "sam2.1_hiera_l", checkpoint, device=dev
                    )
                    self.predictor = SAM2ImagePredictor(sam2_model)
                    self.device = dev
                    logger.info("SAM 2 ready on %s", dev)
                    break
                except Exception as exc:
                    last_err = exc
                    logger.warning("SAM 2 on %s failed: %s", dev, exc)
                    self.predictor = None
            else:
                if last_err is not None:
                    raise last_err

        except ImportError:
            logger.warning(
                "SAM 2 not installed. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
        except Exception as exc:
            logger.error("Failed to load SAM 2: %s", exc)

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def _resolve_checkpoint(self, cache_dir: str) -> str | None:
        candidates = [
            os.path.join(cache_dir, "sam2.1_hiera_large.pt"),
            os.path.join(cache_dir, "sam2_hiera_large.pt"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id="facebook/sam2.1-hiera-large",
                filename="sam2.1_hiera_large.pt",
                cache_dir=cache_dir,
            )
        except Exception:
            pass

        return None

    def refine(
        self, image: Image.Image, coarse_mask: np.ndarray
    ) -> np.ndarray:
        """Backward-compatible wrapper — uses coarse mask without polygon."""
        return self.refine_with_polygon(image, coarse_mask, polygon=None)

    def refine_with_polygon(
        self,
        image: Image.Image,
        coarse_mask: np.ndarray,
        polygon: list[dict] | None = None,
    ) -> np.ndarray:
        """Refine using SAM2 with point prompts derived from polygon interior.

        Positive prompts come from the deeply-eroded interior of the coarse
        mask — regions that are almost certainly pure wall surface, far from
        any objects or edges.  Negative prompts at image corners tell SAM2
        not to expand into non-wall areas.

        Using multimask_output=True and selecting the best mask by IoU with
        the polygon further prevents SAM2 from expanding into objects on the
        wall (clocks, paintings, etc.) that were correctly excluded by the
        upstream OneFormer step.
        """
        if self.predictor is None:
            logger.warning("SAM 2 unavailable — returning coarse mask")
            return coarse_mask

        H, W = image.height, image.width
        image_array = np.array(image)
        self.predictor.set_image(image_array)

        # --- positive prompts: deep interior of coarse mask ----------------
        pos_points = _sample_interior_points(coarse_mask, n_points=7)

        if not pos_points:
            ys, xs = np.where(coarse_mask > 0)
            if len(ys) == 0:
                return coarse_mask
            pos_points = [[int(xs.mean()), int(ys.mean())]]

        # --- negative prompts: image corners + polygon ceiling / sides -------
        neg_points = [
            [5, 5],
            [W - 5, 5],
            [5, H - 5],
            [W - 5, H - 5],
            [W // 2, 5],
            [W // 2, H - 5],
        ]
        if polygon:
            neg_points.extend(_polygon_boundary_negative_points(polygon, W, H))

        point_coords = np.array(pos_points + neg_points, dtype=np.float32)
        point_labels = np.array(
            [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32
        )

        # --- coarse mask as prior ----------------------------------------
        mask_pil = Image.fromarray((coarse_mask * 255).astype(np.uint8))
        mask_256 = (
            np.array(mask_pil.resize((256, 256), Image.NEAREST)).astype(
                np.float32
            )
            / 255.0
        )
        mask_logits = np.where(mask_256 > 0.5, 10.0, -10.0).astype(
            np.float32
        )
        mask_input = mask_logits[np.newaxis, :, :]

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

        return _pick_best_mask(masks, scores, coarse_mask)
