"""Tone / colour restoration, denoising and sharpening."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageio import luminance


@dataclass
class ToneAnalysis:
    black: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    white: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    has_black: bool = False
    has_white: bool = False
    noise: float = 0.0


def analyze_tone(img: np.ndarray) -> ToneAnalysis:
    """Find the ink-black and paper-white reference levels of a print scan."""
    res = ToneAnalysis()
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    lum = luminance(small)
    flat = small.reshape(-1, 3)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    dark = flat[lum.ravel() <= lo]
    light = flat[lum.ravel() >= hi]
    if len(dark):
        b = np.median(dark, axis=0)
        # Only when the cover really contains neutral (near) black ink;
        # dark coloured ink must not be pushed to black.
        if b.max() < 0.22 and float(b.max() - b.min()) < 0.045:
            res.black = [float(x * 0.9) for x in b]
            res.has_black = True
    if len(light):
        wv = np.median(light, axis=0)
        # Only when the lightest area is plausibly bare paper: bright and
        # nearly neutral.  A coloured cover without white keeps its levels.
        if wv.min() > 0.86 and float(wv.max() - wv.min()) < 0.045:
            res.white = [float(min(1.0, x * 1.005)) for x in wv]
            res.has_white = True
    res.noise = estimate_noise(img)
    return res


def estimate_noise(img: np.ndarray) -> float:
    """Robust noise sigma (0..1 scale) from the Laplacian in smooth areas."""
    lum = luminance(img)
    h, w = lum.shape
    lum = lum[h // 8: h - h // 8: 2, w // 8: w - w // 8: 2]
    lap = cv2.Laplacian(lum, cv2.CV_32F, ksize=1)
    return float(1.4826 * np.median(np.abs(lap)) / np.sqrt(20.0))


def apply_tone(img: np.ndarray, black, white, gamma: float = 1.0, saturation: float = 1.0,
               contrast: float = 0.0, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    b = np.asarray(black, np.float32).reshape(1, 1, 3)
    w = np.asarray(white, np.float32).reshape(1, 1, 3)
    out = (img - b) / np.maximum(w - b, 0.05)
    if temperature or tint:
        # B, G, R gains; positive temperature = warmer, positive tint = magenta
        g = np.array([1.0 - 0.25 * temperature + 0.10 * tint,
                      1.0 - 0.20 * tint,
                      1.0 + 0.25 * temperature + 0.10 * tint], np.float32)
        out = out * g.reshape(1, 1, 3)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, 1.0 / max(gamma, 0.05))
    if contrast:
        # smooth S-curve (or inverse) around mid grey
        c = float(np.clip(contrast, -1, 1))
        s = out * out * (3 - 2 * out)
        out = out + c * (s - out) if c > 0 else out + (-c) * (0.5 + (out - 0.5) * 0.5 - out)
    if saturation != 1.0:
        lum = luminance(out)[:, :, None]
        out = lum + (out - lum) * float(saturation)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def denoise(img: np.ndarray, strength: float = 0.3, chroma: float = 0.5) -> np.ndarray:
    """Edge-preserving smoothing, stronger on colour than on brightness."""
    if strength <= 0 and chroma <= 0:
        return img
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    if strength > 0:
        sc = 0.015 + 0.085 * float(strength)
        y = cv2.bilateralFilter(y, 7, sc, 2.0 + 2.0 * float(strength))
    if chroma > 0:
        sc = 0.03 + 0.12 * float(chroma)
        crcb = cv2.merge([cr, cb, np.zeros_like(cr)])
        crcb = cv2.bilateralFilter(crcb, 9, sc, 3.0 + 3.0 * float(chroma))
        cr, cb = crcb[:, :, 0], crcb[:, :, 1]
    out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return np.clip(out, 0.0, 1.0)


def sharpen(img: np.ndarray, amount: float = 0.5, radius: float = 1.2) -> np.ndarray:
    """Unsharp mask on brightness only, with halo limiting."""
    if amount <= 0:
        return img
    lum = luminance(img)
    blur = cv2.GaussianBlur(lum, (0, 0), max(0.3, float(radius)))
    detail = lum - blur
    # soft limiter keeps strong edges from ringing
    lim = 0.12
    detail = lim * np.tanh(detail / lim)
    # ignore the faintest differences so flat areas stay smooth
    detail = np.sign(detail) * np.maximum(np.abs(detail) - 0.004, 0.0)
    return np.clip(img + (float(amount) * 1.6 * detail)[:, :, None], 0.0, 1.0)


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return np.clip(cv2.resize(img, size, interpolation=interp), 0.0, 1.0)
