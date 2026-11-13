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


def _tiff_dpi(buf: bytes, base: int = 0):
    """XResolution/ResolutionUnit from a TIFF structure at `base` in `buf`
    (a TIFF file, or the Exif block of a JPEG)."""
    if buf[base:base + 2] == b"II":
        order = "little"
    elif buf[base:base + 2] == b"MM":
        order = "big"
    else:
        return None
    u16 = lambda o: int.from_bytes(buf[base + o:base + o + 2], order)
    u32 = lambda o: int.from_bytes(buf[base + o:base + o + 4], order)
    if u16(2) != 42:
        return None
    ifd = u32(4)
    if ifd + 2 > len(buf) - base:
        return None
    xres, unit = None, 2                                  # default unit: inch
    for k in range(u16(ifd)):
        e = ifd + 2 + 12 * k
        if e + 12 > len(buf) - base:
            break
        tag, typ = u16(e), u16(e + 2)
        if tag == 282 and typ == 5:                       # XResolution, RATIONAL
            off = u32(e + 8)
            num, den = u32(off), u32(off + 4)
            xres = num / den if den else None
        elif tag == 296:                                  # ResolutionUnit, SHORT
            unit = u16(e + 8)
    if not xres or xres <= 1:
        return None
    if unit == 3:                                         # per centimetre
        return xres * 2.54
    return xres if unit == 2 else None


def read_dpi(path: str):
    """Scan resolution stored in the file (PNG pHYs, TIFF, JPEG JFIF/Exif),
    or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 18)
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
    if head[:4] in (b"II*\x00", b"MM\x00*"):          # TIFF
        return _tiff_dpi(head)
    if head[:2] == b"\xff\xd8":                         # JPEG
        i = head.find(b"Exif\x00\x00")
        if i > 0:
            dpi = _tiff_dpi(head, i + 6)
            if dpi:
                return dpi
        i = head.find(b"JFIF\x00")
        if i > 0 and i + 12 <= len(head):
            unit = head[i + 7]
            x = int.from_bytes(head[i + 8:i + 10], "big")
            if unit == 1 and x > 1:
                return float(x)
            if unit == 2 and x > 1:                     # dots per centimetre
                return x * 2.54
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
