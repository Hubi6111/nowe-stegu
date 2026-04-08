"""Persistent admin configuration — JSON file backed store.

Manages: admin password, watermark settings, daily generation limits,
and per-IP rate tracking.
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT / "data")))
CONFIG_PATH = DATA_DIR / "admin-config.json"
WATERMARK_PATH = DATA_DIR / "watermark.png"
RATE_LIMIT_PATH = DATA_DIR / "rate-limits.json"

_lock = threading.Lock()

DEFAULT_CONFIG = {
    "admin_password_hash": hashlib.sha256(b"stegu").hexdigest(),
    "daily_generation_limit": 50,
    "watermark_enabled": False,
    "watermark_opacity": 0.3,
}


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    _ensure_data_dir()
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text("utf-8"))
        except Exception:
            logger.warning("Corrupt admin config, using defaults")
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    _ensure_data_dir()
    with _lock:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), "utf-8")


def verify_password(password: str) -> bool:
    cfg = load_config()
    h = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return h == cfg.get("admin_password_hash", "")


def change_password(new_password: str):
    cfg = load_config()
    cfg["admin_password_hash"] = hashlib.sha256(new_password.encode("utf-8")).hexdigest()
    save_config(cfg)


def get_daily_limit() -> int:
    return int(load_config().get("daily_generation_limit", 50))


def set_daily_limit(limit: int):
    cfg = load_config()
    cfg["daily_generation_limit"] = max(0, limit)
    save_config(cfg)


def get_watermark_config() -> dict:
    cfg = load_config()
    return {
        "enabled": cfg.get("watermark_enabled", False),
        "opacity": cfg.get("watermark_opacity", 0.3),
        "has_file": WATERMARK_PATH.is_file(),
    }


def set_watermark_config(enabled: bool | None = None, opacity: float | None = None):
    cfg = load_config()
    if enabled is not None:
        cfg["watermark_enabled"] = enabled
    if opacity is not None:
        cfg["watermark_opacity"] = max(0.05, min(1.0, opacity))
    save_config(cfg)


def save_watermark(data: bytes):
    _ensure_data_dir()
    WATERMARK_PATH.write_bytes(data)
    cfg = load_config()
    cfg["watermark_enabled"] = True
    save_config(cfg)


def delete_watermark():
    if WATERMARK_PATH.is_file():
        WATERMARK_PATH.unlink()
    cfg = load_config()
    cfg["watermark_enabled"] = False
    save_config(cfg)


def load_watermark_image():
    """Load watermark as PIL Image or None."""
    if not WATERMARK_PATH.is_file():
        return None
    from PIL import Image
    try:
        return Image.open(WATERMARK_PATH).convert("RGBA")
    except Exception:
        return None


# ── Rate limiting ─────────────────────────────────────────────────────────────

def _today_key() -> str:
    return time.strftime("%Y-%m-%d")


def _load_rates() -> dict:
    if RATE_LIMIT_PATH.is_file():
        try:
            return json.loads(RATE_LIMIT_PATH.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _save_rates(rates: dict):
    _ensure_data_dir()
    with _lock:
        RATE_LIMIT_PATH.write_text(json.dumps(rates, indent=2), "utf-8")


def check_rate_limit(client_ip: str) -> tuple[bool, int, int]:
    """Check if client_ip is within daily limit.

    Returns (allowed, used_count, daily_limit).
    """
    limit = get_daily_limit()
    if limit <= 0:
        return True, 0, 0

    today = _today_key()
    rates = _load_rates()

    if rates.get("_date") != today:
        rates = {"_date": today}

    used = int(rates.get(client_ip, 0))
    return used < limit, used, limit


def increment_usage(client_ip: str):
    today = _today_key()
    rates = _load_rates()

    if rates.get("_date") != today:
        rates = {"_date": today}

    rates[client_ip] = int(rates.get(client_ip, 0)) + 1
    _save_rates(rates)


def get_usage_stats() -> dict:
    rates = _load_rates()
    date = rates.pop("_date", _today_key())
    return {"date": date, "users": {k: v for k, v in rates.items()}, "total": sum(rates.values())}


# ── Generation history ────────────────────────────────────────────────────────

GENERATIONS_DIR = DATA_DIR / "generations"
GENERATIONS_INDEX = DATA_DIR / "generations-index.json"


def _load_gen_index() -> list[dict]:
    if GENERATIONS_INDEX.is_file():
        try:
            return json.loads(GENERATIONS_INDEX.read_text("utf-8"))
        except Exception:
            pass
    return []


def _save_gen_index(index: list[dict]):
    _ensure_data_dir()
    with _lock:
        GENERATIONS_INDEX.write_text(json.dumps(index, indent=2), "utf-8")


def save_generation(
    client_ip: str,
    product_id: str,
    product_name: str,
    gemini_model: str,
    timings: dict,
    refined_b64: str | None,
    thumbnail_b64: str | None = None,
) -> str:
    """Save a generation record. Returns the generation ID."""
    import uuid

    _ensure_data_dir()
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)

    gen_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    if refined_b64:
        raw = refined_b64.split(",", 1)[-1] if "," in refined_b64 else refined_b64
        import base64 as b64mod
        (GENERATIONS_DIR / f"{gen_id}.jpg").write_bytes(b64mod.b64decode(raw))

    if thumbnail_b64:
        raw_t = thumbnail_b64.split(",", 1)[-1] if "," in thumbnail_b64 else thumbnail_b64
        import base64 as b64mod
        (GENERATIONS_DIR / f"{gen_id}_thumb.jpg").write_bytes(b64mod.b64decode(raw_t))

    record = {
        "id": gen_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_ip": client_ip,
        "product_id": product_id,
        "product_name": product_name,
        "gemini_model": gemini_model,
        "timings": timings,
        "has_image": refined_b64 is not None,
    }

    index = _load_gen_index()
    index.insert(0, record)
    if len(index) > 500:
        index = index[:500]
    _save_gen_index(index)

    return gen_id


def list_generations(limit: int = 50, offset: int = 0, client_ip: str | None = None) -> dict:
    index = _load_gen_index()
    if client_ip:
        index = [g for g in index if g.get("client_ip") == client_ip]
    total = len(index)
    items = index[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items}


def get_generation_image_path(gen_id: str) -> Path | None:
    safe = gen_id.replace("..", "").replace("/", "")
    p = GENERATIONS_DIR / f"{safe}.jpg"
    return p if p.is_file() else None


def get_generation_stats() -> dict:
    """Aggregate stats across all generations."""
    index = _load_gen_index()
    if not index:
        return {"total": 0, "today": 0, "unique_ips": 0, "by_product": {}, "by_day": {}}

    today = _today_key()
    today_count = sum(1 for g in index if g.get("timestamp", "").startswith(today))
    unique_ips = len(set(g.get("client_ip", "") for g in index))

    by_product: dict[str, int] = {}
    by_day: dict[str, int] = {}
    for g in index:
        pid = g.get("product_name") or g.get("product_id", "unknown")
        by_product[pid] = by_product.get(pid, 0) + 1
        day = g.get("timestamp", "")[:10]
        if day:
            by_day[day] = by_day.get(day, 0) + 1

    return {
        "total": len(index),
        "today": today_count,
        "unique_ips": unique_ips,
        "by_product": dict(sorted(by_product.items(), key=lambda x: -x[1])),
        "by_day": dict(sorted(by_day.items(), reverse=True)[:30]),
    }
