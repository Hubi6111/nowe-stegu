"""Admin API — texture management, watermark, rate limits, generation gallery."""

import base64
import io
import json
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.admin_store import (
    verify_password,
    change_password,
    load_config,
    get_daily_limit,
    set_daily_limit,
    get_watermark_config,
    set_watermark_config,
    save_watermark,
    delete_watermark,
    get_usage_stats,
    list_generations,
    get_generation_image_path,
    get_generation_stats,
    WATERMARK_PATH,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

TEXTURES_DIR = Path(
    os.environ.get(
        "TEXTURES_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "assets" / "textures" / "stegu"),
    )
)


def _require_admin(x_admin_password: str = Header(...)):
    if not verify_password(x_admin_password):
        raise HTTPException(401, detail="Nieprawidłowe hasło")


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str

@router.post("/login")
def admin_login(req: LoginRequest):
    if not verify_password(req.password):
        raise HTTPException(401, detail="Nieprawidłowe hasło")
    return {"ok": True}


class ChangePasswordRequest(BaseModel):
    new_password: str

@router.post("/change-password", dependencies=[Depends(_require_admin)])
def admin_change_password(req: ChangePasswordRequest):
    if len(req.new_password) < 4:
        raise HTTPException(400, detail="Hasło musi mieć min. 4 znaki")
    change_password(req.new_password)
    return {"ok": True}


# ── Textures CRUD ─────────────────────────────────────────────────────────────

@router.get("/textures", dependencies=[Depends(_require_admin)])
def list_textures_admin():
    if not TEXTURES_DIR.is_dir():
        return []
    result = []
    for folder in sorted(TEXTURES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        meta_path = folder / "metadata.json"
        has_albedo = (folder / "albedo.jpg").is_file()
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except Exception:
                pass
        result.append({
            "id": folder.name,
            "name": meta.get("name", folder.name),
            "has_albedo": has_albedo,
            "meta": meta,
        })
    return result


class TextureUpdateRequest(BaseModel):
    name: str | None = None
    moduleWidthMm: float | None = None
    moduleHeightMm: float | None = None
    jointMm: float | None = None
    layoutType: str | None = None
    offsetRatio: float | None = None
    textureScaleMultiplier: float | None = None
    albedoBrickCourses: int | None = None
    tags: list[str] | None = None

@router.put("/textures/{texture_id}", dependencies=[Depends(_require_admin)])
def update_texture(texture_id: str, req: TextureUpdateRequest):
    safe = texture_id.replace("..", "").replace("/", "")
    folder = TEXTURES_DIR / safe
    if not folder.is_dir():
        raise HTTPException(404, detail="Tekstura nie znaleziona")
    meta_path = folder / "metadata.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text("utf-8"))

    update_data = req.model_dump(exclude_none=True)
    meta.update(update_data)

    meta_path.write_text(json.dumps(meta, indent=2), "utf-8")
    return {"ok": True, "meta": meta}


class TextureCreateRequest(BaseModel):
    id: str
    name: str
    albedo_base64: str
    moduleWidthMm: float = 245
    moduleHeightMm: float = 80
    jointMm: float = 10
    layoutType: str = "running-bond"
    offsetRatio: float = 0.5
    tags: list[str] = []

@router.post("/textures", dependencies=[Depends(_require_admin)])
def create_texture(req: TextureCreateRequest):
    safe = req.id.replace("..", "").replace("/", "").replace(" ", "-").lower()
    if not safe:
        raise HTTPException(400, detail="Nieprawidłowe ID")
    folder = TEXTURES_DIR / safe
    if folder.exists():
        raise HTTPException(409, detail="Tekstura o tym ID już istnieje")

    folder.mkdir(parents=True, exist_ok=True)

    raw = req.albedo_base64.split(",", 1)[-1] if "," in req.albedo_base64 else req.albedo_base64
    try:
        img_bytes = base64.b64decode(raw)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(folder / "albedo.jpg", "JPEG", quality=92)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, detail=f"Nieprawidłowy obraz: {exc}")

    meta = {
        "name": req.name,
        "moduleWidthMm": req.moduleWidthMm,
        "moduleHeightMm": req.moduleHeightMm,
        "jointMm": req.jointMm,
        "layoutType": req.layoutType,
        "offsetRatio": req.offsetRatio,
        "tags": req.tags,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    return {"ok": True, "id": safe}


@router.delete("/textures/{texture_id}", dependencies=[Depends(_require_admin)])
def delete_texture(texture_id: str):
    safe = texture_id.replace("..", "").replace("/", "")
    folder = TEXTURES_DIR / safe
    if not folder.is_dir():
        raise HTTPException(404, detail="Tekstura nie znaleziona")
    shutil.rmtree(folder)
    return {"ok": True}


# ── Watermark ─────────────────────────────────────────────────────────────────

@router.get("/watermark", dependencies=[Depends(_require_admin)])
def watermark_status():
    return get_watermark_config()


class WatermarkConfigRequest(BaseModel):
    enabled: bool | None = None
    opacity: float | None = None

@router.put("/watermark", dependencies=[Depends(_require_admin)])
def update_watermark_config(req: WatermarkConfigRequest):
    set_watermark_config(enabled=req.enabled, opacity=req.opacity)
    return get_watermark_config()


class WatermarkUploadRequest(BaseModel):
    image_base64: str

@router.post("/watermark/upload", dependencies=[Depends(_require_admin)])
def upload_watermark(req: WatermarkUploadRequest):
    raw = req.image_base64.split(",", 1)[-1] if "," in req.image_base64 else req.image_base64
    try:
        data = base64.b64decode(raw)
        from PIL import Image
        Image.open(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(400, detail=f"Nieprawidłowy obraz: {exc}")
    save_watermark(data)
    return get_watermark_config()


@router.delete("/watermark", dependencies=[Depends(_require_admin)])
def remove_watermark():
    delete_watermark()
    return {"ok": True}


@router.get("/watermark/preview")
def watermark_preview():
    if not WATERMARK_PATH.is_file():
        raise HTTPException(404, detail="Brak watermarku")
    return FileResponse(WATERMARK_PATH, media_type="image/png")


# ── Rate limits ───────────────────────────────────────────────────────────────

@router.get("/limits", dependencies=[Depends(_require_admin)])
def limits_status():
    return {
        "daily_limit": get_daily_limit(),
        "usage": get_usage_stats(),
    }


class LimitUpdateRequest(BaseModel):
    daily_limit: int

@router.put("/limits", dependencies=[Depends(_require_admin)])
def update_limits(req: LimitUpdateRequest):
    set_daily_limit(req.daily_limit)
    return {"daily_limit": get_daily_limit()}


# ── Generations gallery ───────────────────────────────────────────────────────

@router.get("/generations", dependencies=[Depends(_require_admin)])
def admin_list_generations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    client_ip: str | None = Query(None),
):
    return list_generations(limit=limit, offset=offset, client_ip=client_ip)


@router.get("/generations/{gen_id}/image")
def admin_generation_image(gen_id: str):
    p = get_generation_image_path(gen_id)
    if not p:
        raise HTTPException(404, detail="Obraz nie znaleziony")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/stats", dependencies=[Depends(_require_admin)])
def admin_stats():
    return {
        **get_generation_stats(),
        "daily_limit": get_daily_limit(),
        "usage_today": get_usage_stats(),
    }
