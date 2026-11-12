"""Image loading / saving.

Images are processed internally as float32 BGR in the range 0..1 so that
chained corrections never accumulate 8-bit rounding errors.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

SUPPORTED = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")


def load_image(path: str) -> np.ndarray:
    """Load an image as float32 BGR (0..1). Handles unicode paths and 16-bit."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def read_dpi(path: str):
    """Scan resolution stored in a PNG (pHYs chunk) or TIFF/JPEG, or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 16)
    except OSError:
        return None
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        i = head.find(b"pHYs")
        if i > 0 and i + 13 <= len(head):
            x = int.from_bytes(head[i + 4:i + 8], "big")
            unit = head[i + 12]
            if unit == 1 and x > 0:                   # pixels per metre
                return x * 0.0254
        return None
    if head[:2] == b"\xff\xd8":                         # JPEG JFIF
        i = head.find(b"JFIF\x00")
        if i > 0 and head[i + 7] == 1:
            return float(int.from_bytes(head[i + 8:i + 10], "big")) or None
    return None


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def save_image(path: str, img: np.ndarray, bit_depth: int = 8, jpeg_quality: int = 97) -> None:
    """Save a float32 BGR image. 16-bit is honoured for PNG and TIFF."""
    ext = os.path.splitext(path)[1].lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        out = to_uint8(img)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif ext in (".png", ".tif", ".tiff") and bit_depth == 16:
        out = to_uint16(img)
    else:
        out = to_uint8(img)
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode(ext, out, params)
    if not ok:
        raise IOError(f"Could not encode image as {ext}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a BGR float image."""
    return img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
