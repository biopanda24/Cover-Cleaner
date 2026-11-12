"""Halftone (print screen) detection and removal.

Offset-printed covers are made of regular dot grids, one per ink, each at
its own angle.  In the frequency domain such a grid is a set of very sharp
peaks, while the picture itself lives at lower frequencies.  We therefore:

* average the power spectrum of many tiles (Welch) and compare it with its
  own radial median to find peaks that cannot be picture content;
* remove exactly those peaks with narrow notch filters (this already kills
  the dot pattern and the rosette/moire beats while leaving everything else
  untouched);
* optionally add a smooth low-pass just below the screen frequency to remove
  what is left of the dot structure;
* near strong edges (solid line art, lettering) fall back to the notch-only
  result so text stays crisp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import fft as sfft

TILE = 256


@dataclass
class ScreenAnalysis:
    found: bool = False
    period: float = 0.0            # px, of the lowest strong screen frequency
    frequency: float = 0.0         # cycles / px
    strength: float = 0.0          # peak-to-background power ratio
    peaks: list = field(default_factory=list)   # [fx, fy, ratio]
    lpi_at_600dpi: float = 0.0


def _welch(chan: np.ndarray) -> tuple[np.ndarray, int]:
    win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)
    acc = np.zeros((TILE, TILE), np.float64)
    cnt = 0
    h, w = chan.shape
    step = TILE // 2 if h * w < 4000 * 4000 else TILE
    for y in range(0, h - TILE + 1, step):
        for x in range(0, w - TILE + 1, step):
            p = chan[y:y + TILE, x:x + TILE]
            m = float(p.mean())
            if m < 0.06 or m > 0.97:
                continue                       # solid ink / bare paper: no dots
            acc += np.abs(sfft.fft2((p - m) * win)) ** 2
            cnt += 1
    return acc / max(cnt, 1), cnt


def analyze_screen(img: np.ndarray, min_ratio: float = 60.0) -> ScreenAnalysis:
    """Look for a halftone screen in a float BGR image."""
    res = ScreenAnalysis()
    if min(img.shape[:2]) < TILE * 2:
        return res
    fy, fx = np.meshgrid(sfft.fftfreq(TILE), sfft.fftfreq(TILE), indexing="ij")
    r = np.hypot(fx, fy)
    ri = np.minimum((r * TILE).astype(int), TILE - 1)
    ratio = np.zeros((TILE, TILE))
    for c in range(3):
        acc, cnt = _welch(img[:, :, c])
        if cnt < 4:
            continue
        rad = np.ones(TILE)
        for k in range(TILE):
            sel = acc[ri == k]
            if sel.size:
                rad[k] = np.median(sel)
        # Floor = spectrum of faint white noise (sigma 0.5/255), so a clean,
        # noise-free image cannot produce huge ratios out of nothing.
        floor = (0.002 ** 2) * (0.375 * TILE) ** 2
        ratio = np.maximum(ratio, acc / np.maximum(rad[ri], floor))
    # Ignore low frequencies (picture content) and the axes, where sharp
    # horizontal/vertical picture edges and scanner streaks concentrate.
    ratio[r < 0.07] = 0
    axis = (np.abs(fx) < 1.5 / TILE) | (np.abs(fy) < 1.5 / TILE)
    cand = ratio.copy()
    cand[axis & (r < 0.2)] = 0
    shifted = sfft.fftshift(cand).astype(np.float32)
    local_max = cv2.dilate(shifted, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((shifted >= local_max) & (shifted > 25.0))
    fxs = sfft.fftshift(fx)[ys, xs]
    fys = sfft.fftshift(fy)[ys, xs]
    vals = shifted[ys, xs]
    # Presence test uses off-axis peaks only: sharp horizontal / vertical
    # picture edges put energy on the axes at every frequency.
    rr = np.hypot(fxs, fys)
    off_axis = (np.abs(fxs) > 2.5 / TILE) & (np.abs(fys) > 2.5 / TILE)
    if not off_axis.any() or vals[off_axis].max() < min_ratio:
        return res
    top = float(vals[off_axis].max())
    # A dot grid is periodic in two perpendicular directions, so the main
    # peak must have a partner rotated by 90 degrees.
    j = int(np.flatnonzero(off_axis)[np.argmax(vals[off_axis])])
    py = int(round(fxs[j] * TILE)) % TILE
    px = int(round(-fys[j] * TILE)) % TILE
    win = ratio[np.ix_(np.arange(py - 2, py + 3) % TILE, np.arange(px - 2, px + 3) % TILE)]
    if win.max() < 0.03 * top:
        return res
    # Fundamental = lowest frequency among the really strong peaks.
    strong = off_axis & (vals >= max(min_ratio, 0.2 * top))
    f0 = float(rr[strong].min())
    if not 0.10 <= f0 <= 0.49:
        return res
    order = np.argsort(-vals)[:400]
    peaks = []
    for i in order:
        if rr[i] < 0.42 * f0:
            continue                           # below the rosette beats: keep
        if fys[i] < 0 or (fys[i] == 0 and fxs[i] < 0):
            continue                           # keep one of each +/- pair
        peaks.append([float(fxs[i]), float(fys[i]), float(vals[i])])
    res.found = True
    res.frequency = f0
    res.period = 1.0 / f0
    res.strength = top
    res.peaks = peaks
    res.lpi_at_600dpi = 600.0 * f0
    return res


def _filters(shape, peaks, f0, smoothing):
    """Frequency responses: (notch only, notch + low-pass)."""
    h, w = shape
    notch = np.ones((h, w), np.float32)
    for fx, fy, val in peaks:
        sig = 0.05 * max(f0, 0.15) * (1.0 + 0.3 * np.log10(max(val, 25.0) / 25.0))  # cycles / px
        ry = int(np.ceil(3.5 * sig * h)); rx = int(np.ceil(3.5 * sig * w))
        for sgn in (1, -1):
            cy = sgn * fy * h; cx = sgn * fx * w
            iy = np.arange(int(np.floor(cy)) - ry, int(np.ceil(cy)) + ry + 1)
            ix = np.arange(int(np.floor(cx)) - rx, int(np.ceil(cx)) + rx + 1)
            gy = np.exp(-0.5 * ((iy - cy) / (sig * h)) ** 2)
            gx = np.exp(-0.5 * ((ix - cx) / (sig * w)) ** 2)
            depth = 1.0 - np.outer(gy, gx).astype(np.float32)
            notch[np.ix_(iy % h, ix % w)] *= depth
    if smoothing <= 0 or f0 <= 0:
        return notch, notch
    fy = sfft.fftfreq(h).astype(np.float32)[:, None]
    fx = sfft.fftfreq(w).astype(np.float32)[None, :]
    fc = f0 * (0.95 - 0.45 * float(np.clip(smoothing, 0, 1)))
    r2 = (fx * fx + fy * fy) / (fc * fc)
    low = np.exp(-0.6931 * r2 ** 3)            # 6th-order super-Gaussian
    return notch, notch * low


def descreen(img: np.ndarray, peaks, period: float, strength: float = 1.0,
             smoothing: float = 0.5, protect_edges: float = 0.7) -> np.ndarray:
    """Remove the halftone pattern described by `peaks` from a float image."""
    if not peaks or strength <= 0:
        return img
    f0 = 1.0 / period if period > 0 else 0.0
    pad = 48
    work = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    # A fast FFT size avoids pathological prime dimensions.
    H = sfft.next_fast_len(work.shape[0]); W = sfft.next_fast_len(work.shape[1])
    work = cv2.copyMakeBorder(work, 0, H - work.shape[0], 0, W - work.shape[1],
                              cv2.BORDER_REFLECT_101)
    notch, full = _filters((H, W), peaks, f0, smoothing)
    same = notch is full
    soft = np.empty_like(work)
    crisp = soft if same else np.empty_like(work)
    for c in range(3):
        spec = sfft.fft2(work[:, :, c], workers=-1)
        soft[:, :, c] = sfft.ifft2(spec * full, workers=-1).real
        if not same:
            crisp[:, :, c] = sfft.ifft2(spec * notch, workers=-1).real
    out = soft
    if not same and protect_edges > 0:
        # Strong, large-scale edges mark solid line art: keep it sharp.
        lum = cv2.cvtColor(soft, cv2.COLOR_BGR2GRAY)
        sig = max(1.0, 0.6 * period)
        lum = cv2.GaussianBlur(lum, (0, 0), sig)
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy) * 0.125                  # per-pixel slope
        lo = 0.06
        wgt = np.clip((g - lo) / 0.05, 0.0, 1.0)
        wgt = wgt * wgt * (3 - 2 * wgt)
        wgt = cv2.GaussianBlur(wgt, (0, 0), 1.0)[:, :, None] * float(protect_edges)
        out = soft + wgt * (crisp - soft)
    out = out[pad:pad + img.shape[0], pad:pad + img.shape[1]]
    if strength < 1.0:
        out = img + float(strength) * (out - img)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
