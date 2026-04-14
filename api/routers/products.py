"""Product catalog — reads texture folders from assets/textures/stegu/."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

TEXTURES_DIR = Path(
    os.environ.get(
        "TEXTURES_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "assets" / "textures" / "stegu"),
    )
)

REQUIRED_FIELDS = [
    "name", "moduleWidthMm", "moduleHeightMm",
    "jointMm", "layoutType", "offsetRatio",
]


class ProductOut(BaseModel):
    productId: str
    name: str
    textureImage: str
    moduleWidthMm: int
    moduleHeightMm: int
    jointMm: int
    layoutType: str
    offsetRatio: float
    category: str


LAYOUT_TO_CATEGORY = {
    "running-bond": "cegły",
    "random-stone": "kamień",
    "vertical-stack": "lamele",
    "stack-bond": "cegły",
    "flemish-bond": "cegły",
    "herringbone": "cegły",
}


def _load_product(folder: Path) -> ProductOut:
    meta_path = folder / "metadata.json"
    albedo_path = folder / "albedo.jpg"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json missing in {folder.name}")
    if not albedo_path.exists():
        raise FileNotFoundError(f"albedo.jpg missing in {folder.name}")

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"{folder.name} metadata missing: {', '.join(missing)}")

    return ProductOut(
        productId=folder.name,
        name=raw["name"],
        textureImage="albedo.jpg",
        moduleWidthMm=raw["moduleWidthMm"],
        moduleHeightMm=raw["moduleHeightMm"],
        jointMm=raw["jointMm"],
        layoutType=raw["layoutType"],
        offsetRatio=raw["offsetRatio"],
        category=LAYOUT_TO_CATEGORY.get(raw["layoutType"], "inne"),
    )


@router.get("/api/products", response_model=list[ProductOut])
def list_products():
    if not TEXTURES_DIR.exists():
        raise HTTPException(500, detail=f"Textures directory not found: {TEXTURES_DIR}")

    folders = sorted(
        [d for d in TEXTURES_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    products: list[ProductOut] = []
    errors: list[str] = []

    for folder in folders:
        try:
            products.append(_load_product(folder))
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        raise HTTPException(500, detail={"message": "Product errors", "errors": errors})

    return products


@router.get("/api/textures/{product_id}")
def get_texture(product_id: str):
    safe = product_id.replace("..", "").replace("/", "")
    path = TEXTURES_DIR / safe / "albedo.jpg"

    if not path.is_file():
        raise HTTPException(404, detail=f"Texture not found: {safe}")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
