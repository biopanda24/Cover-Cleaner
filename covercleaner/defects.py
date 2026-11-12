"""Dust / speck removal and glare suppression.

Dust: small isolated blobs that differ strongly from a median-filtered
version of their surroundings.  The threshold rises with local texture so
busy picture areas are not mistaken for dirt, and blobs that sit in a crowd
of similar blobs (small print, patterns) are rejected.  Accepted specks are
inpainted from their surroundings.

Glare: light reflected by glossy or creased paper adds a neutral veil on
top of the picture.  It shows up as a soft bump in the darkest colour
channel.  Instead of painting over it, the veil is subtracted, which keeps
the picture detail underneath.
"""
from __future__ import annotations

import cv2
import numpy as np

from .imageio import luminance, to_uint8


def _odd(x: float) -> int:
    return max(3, int(x) | 1)


def _candidates(img, sensitivity, max_px, dark, bright):
    """Loose first pass: blobs that stand out from a median-filtered copy."""
    u8 = to_uint8(img)
    med = cv2.medianBlur(u8, _odd(min(2 * max_px + 1, 99)))
    diff = (u8.astype(np.float32) - med.astype(np.float32)) / 255.0
    lum = luminance(diff)
    mag = np.abs(diff).max(axis=2)
    # local texture level: typical |difference| in the neighbourhood
    k = _odd(6 * max_px)
    activity = cv2.boxFilter(np.minimum(mag, 0.2), -1, (k, k))
    base = 0.20 - 0.15 * float(np.clip(sensitivity, 0, 1))
    thr = base + 6.0 * activity
    cand = np.zeros(img.shape[:2], np.uint8)
    if dark:
        cand |= ((lum < 0) & (mag > thr)).astype(np.uint8)
    if bright:
        cand |= ((lum > 0) & (mag > thr)).astype(np.uint8)
    return cand, base


def dust_components(img: np.ndarray, sensitivity: float = 0.5, max_size_pct: float = 0.6,
                    dark: bool = True, bright: bool = True, limit: int = 4000):
    """Find specks. Returns (labels, list of dicts with stats per accepted blob)."""
    h, w = img.shape[:2]
    max_px = max(4.0, min(h, w) * max_size_pct / 100.0)
    cand, base = _candidates(img, sensitivity, max_px, dark, bright)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    found = []
    if n <= 1:
        return labels, found
    bw = stats[:, cv2.CC_STAT_WIDTH]; bh = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    ok = (np.maximum(bw, bh) <= max_px) & (area >= 3)
    ok[0] = False
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-area[idx])][:limit]
    lum = luminance(img)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = cv2.magnitude(gx, gy)
    k3 = np.ones((3, 3), np.uint8)
    for i in idx:
        x, y, cw, ch = stats[i, :4]
        size = max(cw, ch)
        g = int(max(4, 0.6 * size)) + 3
        x0, y0 = max(0, x - g), max(0, y - g)
        x1, y1 = min(w, x + cw + g), min(h, y + ch + g)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        near = cv2.dilate(comp, k3, iterations=2)
        far = cv2.dilate(near, k3, iterations=g - 3)
        ring = (far > 0) & (near == 0)
        if ring.sum() < 12:
            continue
        patch = img[y0:y1, x0:x1]
        ring_px = patch[ring]
        ring_med = np.median(ring_px, axis=0)
        contrast = float(np.abs(patch[comp > 0].mean(axis=0) - ring_med).max())
        # 1. A speck sits on a uniform patch.  Corners of shapes, pieces of
        #    thin lines and textured areas have a mixed surrounding.
        dev = np.abs(ring_px - ring_med).max(axis=1)
        uniform = float((dev < max(0.05, 0.35 * contrast)).mean())
        if uniform < 0.92:
            continue
        # 2. Is the wider neighbourhood flat too?  A speck on a solid colour
        #    is certainly dirt; a bright dot inside a photo may be a catch
        #    light or a reflection that belongs to the picture.
        gw = int(3 * size) + 12
        wx0, wy0 = max(0, x - gw), max(0, y - gw)
        wx1, wy1 = min(w, x + cw + gw), min(h, y + ch + gw)
        wide = img[wy0:wy1:2, wx0:wx1:2].reshape(-1, 3)
        wdev = np.abs(wide - ring_med).max(axis=1)
        tol = max(0.05, 0.15 * contrast)
        win_area = max(1, (wx1 - wx0) * (wy1 - wy0))
        own = 4.0 * cw * ch / win_area          # the speck (and halo) itself
        flat = float((wdev >= tol).mean()) <= 0.06 + own
        # ...and the speck must be alone there, not one of a group of marks
        others = int(cand[wy0:wy1, wx0:wx1].sum()) - int(area[i])
        flat = flat and others <= 0.5 * area[i] + 4
        slope = float(np.percentile(grad[y0:y1, x0:x1][near > 0], 90))
        sharp = slope / max(contrast, 1e-3)
        found.append(dict(label=int(i), x=int(x), y=int(y), w=int(cw), h=int(ch),
                          area=int(area[i]), contrast=contrast, sharp=sharp,
                          uniform=uniform, flat=flat))
    return labels, found


def detect_dust(img: np.ndarray, sensitivity: float = 0.5, max_size_pct: float = 0.6,
                dark: bool = True, bright: bool = True) -> np.ndarray:
    """Return a uint8 mask (255 = defect) of specks in a float BGR image."""
    h, w = img.shape[:2]
    max_px = max(4.0, min(h, w) * max_size_pct / 100.0)
    labels, found = dust_components(img, sensitivity, max_size_pct, dark, bright)
    # Up to sensitivity 0.75 only specks on flat colour are trusted; above
    # that, isolated dots inside pictures are removed as well.
    anywhere = sensitivity >= 0.75
    keep = np.zeros(int(labels.max()) + 1, bool)
    for f in found:
        if f["sharp"] >= 0.15 and (f["flat"] or anywhere):
            keep[f["label"]] = True
    mask = keep[labels].astype(np.uint8) * 255
    grow = _odd(max(9, max_px * 0.3))      # cover the speck's blurred halo
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))


def inpaint(img: np.ndarray, mask: np.ndarray, radius: float = 4.0) -> np.ndarray:
    """Fill masked pixels of a float image from their surroundings."""
    if mask is None or not mask.any():
        return img
    out = img.copy()
    pad = int(radius * 4) + 8
    # work on the bounding boxes of mask clusters to keep it fast
    blobs = cv2.dilate(mask, np.ones((pad, pad), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        sl = (slice(y, y + bh), slice(x, x + bw))
        m = np.ascontiguousarray(mask[sl])
        if not m.any():
            continue
        # 16-bit keeps precision; OpenCV's float inpainting is unreliable.
        patch16 = np.clip(out[sl] * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        sel = m > 0
        for c in range(3):
            filled = cv2.inpaint(np.ascontiguousarray(patch16[:, :, c]), m, radius,
                                 cv2.INPAINT_TELEA)
            out[sl][:, :, c][sel] = filled[sel].astype(np.float32) / 65535.0
    return out


def glare_map(img: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Estimate the additive neutral glare veil (float32, same size as img)."""
    h, w = img.shape[:2]
    scale = min(1.0, 1000.0 / min(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    minc = small.min(axis=2)
    maxc = small.max(axis=2)
    k = _odd(0.035 * min(small.shape[:2]))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(cv2.GaussianBlur(minc, (0, 0), 1.5), cv2.MORPH_OPEN, se)
    opened = cv2.GaussianBlur(opened, (0, 0), k * 0.25)
    hat = np.clip(cv2.GaussianBlur(minc, (0, 0), 1.0) - opened, 0, 1)
    thr = 0.22 - 0.16 * float(np.clip(sensitivity, 0, 1))
    cand = (hat > thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    gx = cv2.Sobel(minc, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(minc, cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = cv2.magnitude(gx, gy)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 12 or area > 0.05 * cand.size:
            continue
        sl = (slice(y, y + bh), slice(x, x + bw))
        comp = labels[sl] == i
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~comp
        if not ring.any():
            continue
        contrast = float(hat[sl][comp].mean())
        sharp = float(grad[sl][ring].mean()) / max(contrast, 1e-3)
        neutral = float((maxc[sl][comp] - minc[sl][comp]).mean()) < 0.25
        # printed highlights and lettering have crisp outlines; glare is soft
        keep[i] = sharp < 0.30 and neutral
    sel = keep[labels].astype(np.float32)
    sel = cv2.GaussianBlur(cv2.dilate(sel, np.ones((5, 5), np.uint8)), (0, 0), 2.0)
    veil = hat * sel
    return cv2.resize(veil, (w, h), interpolation=cv2.INTER_LINEAR)


def remove_glare(img: np.ndarray, sensitivity: float = 0.5, strength: float = 1.0) -> np.ndarray:
    veil = glare_map(img, sensitivity)
    if not (veil > 0.01).any():
        return img
    return np.clip(img - float(strength) * veil[:, :, None], 0.0, 1.0)
