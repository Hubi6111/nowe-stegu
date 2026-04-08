"""OneFormer ADE20K semantic segmentation for wall detection."""

import logging
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

WALL_LABELS = {"wall"}

# Surfaces that must be EXCLUDED from wall detection
CEILING_LABELS = {"ceiling"}
FLOOR_LABELS = {"floor", "flooring", "carpet", "rug", "mat"}

# Objects that appear ON/IN FRONT of walls — texture must not cover them
OCCLUDER_LABELS = {
    # Seating & beds
    "bed", "sofa", "couch", "armchair", "chair", "bench", "ottoman", "stool",
    "pillow", "headboard", "mattress", "blanket", "cushion", "swivel chair",
    # Storage & case goods
    "cabinet", "wardrobe", "closet", "bookcase", "bookshelf", "shelf", "shelves",
    "sideboard", "dresser", "chest", "drawer", "buffet", "counter",
    # Tables
    "table", "desk", "coffee table", "nightstand", "end table", "countertop",
    "pool table",
    # Openings & coverings
    "curtain", "curtains", "drape", "blind", "shutter",
    "window", "windowpane", "door", "doorway", "gate",
    # Lighting
    "lamp", "chandelier", "sconce", "light", "light fixture",
    "pendant", "lantern", "candelabrum",
    # Wall decorations — the most common source of bad cuts
    "clock", "mirror", "picture", "painting", "frame", "artwork",
    "poster", "photo", "photograph", "canvas",
    "tapestry", "banner", "sign", "plaque",
    # Plants & organics
    "plant", "potted plant", "flower", "flowers", "vase", "pot",
    "basket", "wreath",
    # Candles & accessories
    "candle", "candlestick", "holder", "decoration", "ornament",
    "figurine", "statue", "sculpture",
    # Electronics
    "television", "tv", "monitor", "screen", "speaker", "loudspeaker",
    "computer", "keyboard", "laptop",
    # Small items on shelves / desks
    "book", "bottle", "glass", "cup", "box", "crate",
    # Appliances & utilities
    "radiator", "fireplace", "air conditioner", "fan",
    "pipe", "column", "partition", "pillar",
    "refrigerator", "oven", "stove", "dishwasher", "microwave",
    "washing machine", "sink", "bathtub", "toilet",
    # People & animals (must not be covered by texture)
    "person", "animal", "dog", "cat",
    # Railings & structural
    "railing", "handrail", "banister", "stairway", "stairs",
    # Other interior elements
    "switch", "outlet", "socket", "vent", "grille",
    "towel", "coat", "clothes",
}

MODEL_ID = "shi-labs/oneformer_ade20k_swin_large"


class OneFormerDetector:
    def __init__(self, cache_dir: str):
        import torch
        from transformers import (
            OneFormerProcessor,
            OneFormerForUniversalSegmentation,
        )

        # MPS does not support float64; OneFormer / HF paths may default to
        # float64 on macOS. Prefer float32 everywhere; fall back to CPU if MPS
        # still fails during load or first forward.
        self._torch = torch

        self.processor = OneFormerProcessor.from_pretrained(
            MODEL_ID, cache_dir=cache_dir
        )
        try:
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                MODEL_ID,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
            )
        except TypeError:
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                MODEL_ID, cache_dir=cache_dir,
            )
        self.model.float().eval()

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
                logger.info(
                    "Trying OneFormer (%s) on %s (float32) …", MODEL_ID, dev
                )
                self.model = self.model.to(dev).float().eval()
                self.device = dev
                self._smoke_forward()
                break
            except Exception as exc:
                last_err = exc
                logger.warning("OneFormer on %s failed: %s", dev, exc)
                self.model = self.model.to("cpu").float().eval()
                self.device = "cpu"
        else:
            if last_err is not None:
                raise last_err
            raise RuntimeError("OneFormer failed to initialize on any device")

        id2label = self.model.config.id2label
        self.wall_ids = {
            int(k)
            for k, v in id2label.items()
            if v.lower() in WALL_LABELS
        }
        self.ceiling_ids = {
            int(k)
            for k, v in id2label.items()
            if v.lower() in CEILING_LABELS
        }
        self.floor_ids = {
            int(k)
            for k, v in id2label.items()
            if v.lower() in FLOOR_LABELS
        }
        self.occluder_ids = {
            int(k)
            for k, v in id2label.items()
            if any(occ in v.lower() for occ in OCCLUDER_LABELS)
        }
        logger.info(
            "OneFormer ready — wall IDs: %s, ceiling: %s, floor: %s, occluder (%d classes)",
            self.wall_ids,
            self.ceiling_ids,
            self.floor_ids,
            len(self.occluder_ids),
        )

    def _inputs_to_device_fp32(self, inputs: dict):
        torch = self._torch
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and v.dtype != torch.float32:
                    v = v.to(dtype=torch.float32)
                v = v.to(self.device)
                out[k] = v
            else:
                out[k] = v
        return out

    def _smoke_forward(self) -> None:
        """Catch MPS float64 issues that surface only during forward."""
        torch = self._torch
        tiny = Image.new("RGB", (64, 64), color=(128, 128, 128))
        inputs = self.processor(
            images=tiny, task_inputs=["semantic"], return_tensors="pt"
        )
        inputs = self._inputs_to_device_fp32(inputs)
        with torch.no_grad():
            _ = self.model(**inputs)

    def _run_forward(self, image: Image.Image):
        """Run model forward pass, retrying on CPU if device fails."""
        torch = self._torch

        def _forward_on(dev: str):
            if self.device != dev:
                logger.warning("Switching OneFormer to %s for this forward pass", dev)
                self.model = self.model.to(dev).float().eval()
                self.device = dev
            inputs = self.processor(
                images=image, task_inputs=["semantic"], return_tensors="pt"
            )
            inputs = self._inputs_to_device_fp32(inputs)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sem_seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(image.height, image.width)]
            )[0]
            # Move to CPU before converting to numpy to avoid MPS→numpy error
            if hasattr(sem_seg, "cpu"):
                sem_seg = sem_seg.cpu()
            return sem_seg.numpy()

        try:
            return _forward_on(self.device)
        except Exception as exc:
            if self.device == "cpu":
                raise
            logger.warning(
                "OneFormer forward on %s failed (%s) — retrying on CPU",
                self.device, exc,
            )
            return _forward_on("cpu")

    def detect(
        self,
        image: Image.Image,
        roi_polygon: list[dict] | None = None,
    ) -> dict:
        """Run semantic segmentation and return wall + occluder masks."""
        sem_seg = self._run_forward(image)

        wall_mask = np.isin(sem_seg, list(self.wall_ids)).astype(np.uint8)
        ceiling_mask = np.isin(sem_seg, list(self.ceiling_ids)).astype(np.uint8)
        floor_mask = np.isin(sem_seg, list(self.floor_ids)).astype(np.uint8)
        occluder_mask = np.isin(sem_seg, list(self.occluder_ids)).astype(np.uint8)

        # Explicitly remove ceiling and floor from wall areas
        wall_mask = wall_mask & (~ceiling_mask.astype(bool)).astype(np.uint8)
        wall_mask = wall_mask & (~floor_mask.astype(bool)).astype(np.uint8)

        # Full-frame ceiling/floor for downstream subtraction (ceiling bleed, floor edge).
        ceiling_full = ceiling_mask.copy()
        floor_full = floor_mask.copy()

        # Full-image masks BEFORE polygon clipping — the pipeline uses these
        # with the polygon as a soft seed (not a hard boundary).
        wall_mask_full = wall_mask.copy()
        occluder_mask_full = occluder_mask.copy()

        if roi_polygon and len(roi_polygon) >= 3:
            roi = polygon_to_mask(roi_polygon, image.width, image.height)
            wall_mask = wall_mask & roi
            occluder_mask = occluder_mask & roi

        # Wall cannot overlap with occluders
        wall_mask = wall_mask & (~occluder_mask.astype(bool)).astype(np.uint8)

        return {
            "wall_mask": wall_mask,
            "wall_mask_full": wall_mask_full,
            "occluder_mask": occluder_mask,
            "occluder_mask_full": occluder_mask_full,
            "ceiling_mask": ceiling_full,
            "floor_mask": floor_full,
        }


def polygon_to_mask(
    polygon: list[dict], width: int, height: int
) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    pts = [(p["x"], p["y"]) for p in polygon]
    draw.polygon(pts, fill=255)
    return (np.array(img) > 0).astype(np.uint8)
