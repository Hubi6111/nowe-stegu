"""HTTP client for the inference service (OneFormer + SAM2).

The inference service preloads models at startup.  This client checks
readiness before forwarding heavy requests and returns structured errors
rather than hanging until a socket timeout.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

INFERENCE_URL = (
    os.environ.get("INFERENCE_BASE_URL")
    or os.environ.get("INFERENCE_URL")
    or "http://localhost:8001"
)

HEALTH_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


async def inference_health() -> dict | None:
    """Return /health payload or None if the service is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            r = await client.get(f"{INFERENCE_URL}/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


async def inference_ready() -> bool:
    """True only if models are loaded and the service can handle requests."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            r = await client.get(f"{INFERENCE_URL}/ready")
            return r.status_code == 200
    except Exception:
        return False


def _raise_for_inference_error(r: httpx.Response, endpoint: str) -> None:
    """Convert any non-2xx inference response to a descriptive RuntimeError."""
    if r.is_success:
        return
    try:
        detail = r.json().get("detail", r.text[:200])
    except Exception:
        detail = r.text[:200] if r.text else f"HTTP {r.status_code}"
    raise RuntimeError(f"Inference {endpoint} returned {r.status_code}: {detail}")


async def detect_wall(
    image_b64: str,
    polygon: list[dict],
    canvas_width: int | float,
    canvas_height: int | float,
) -> dict:
    """Call inference /wall-detect and return parsed response."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{INFERENCE_URL}/wall-detect",
            json={
                "image": image_b64,
                "polygon": polygon,
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
            },
        )
        _raise_for_inference_error(r, "/wall-detect")
        return r.json()


async def refine_wall(
    image_b64: str,
    coarse_mask_b64: str,
    canvas_width: int | float,
    canvas_height: int | float,
    polygon: list[dict] | None = None,
) -> dict:
    """Call inference /wall-refine and return parsed response."""
    payload: dict = {
        "image": image_b64,
        "coarse_mask": coarse_mask_b64,
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
    }
    if polygon:
        payload["polygon"] = polygon
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{INFERENCE_URL}/wall-refine", json=payload)
        _raise_for_inference_error(r, "/wall-refine")
        return r.json()
