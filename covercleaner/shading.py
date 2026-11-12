"""Edge shading correction.

Paper that does not lie flat (a booklet near its spine, a curled insert)
gets darker towards the lifted edge.  The picture content is unknown, but a
shadow has a recognisable signature: moving one pixel towards the edge, the
brightness of *most* pixels changes by the same small factor, equally in
all three colour channels.  The median of pixel-to-pixel log differences
along the edge therefore measures the shadow while ignoring picture detail,
and integrating it gives the falloff profile to divide out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SIDES = ("left", "top", "right", "bottom")
_EPS = 0.02


@dataclass
class ShadingAnalysis:
    # per side (l, t, r, b): brightness loss at the very edge, 0..1
    falloff: list = field(default_factory=lambda: [0.0] * 4)
    is_shadow: list = field(default_factory=lambda: [False] * 4)
    profiles: list = field(default_factory=list)


def _oriented(img: np.ndarray, side: int) -> np.ndarray:
    """View of the image with the given side as column 0."""
    if side == 0:
        return img
    if side == 1:
        return img.transpose(1, 0, 2)
    if side == 2:
        return img[:, ::-1]
    return img.transpose(1, 0, 2)[:, ::-1]


def _profile(img: np.ndarray, side: int, band: int) -> np.ndarray:
    """Log-gain profile, shape (band, 3); 0 = same brightness as interior."""
    v = _oriented(img, side)[:, : band + 1]
    v = cv2.GaussianBlur(np.ascontiguousarray(v), (0, 0), 2.0)
    lg = np.log(v + _EPS)
    d = lg[:, :-1] - lg[:, 1:]                       # towards the edge
    step = max(1, d.shape[0] // 1500)
    med = np.median(d[::step], axis=0)               # (band, 3)
    # A jump shared by a whole column is a printed frame line, not shadow.
    med = np.where(np.abs(med) > 0.02, 0.0, med)
    prof = np.cumsum(med[::-1], axis=0)[::-1]
    k = max(3, (band // 12) | 1)
    return cv2.GaussianBlur(prof, (1, k), 0)


def analyze_shading(img: np.ndarray, width_pct: float = 4.0) -> ShadingAnalysis:
    res = ShadingAnalysis()
    band = max(8, int(min(img.shape[:2]) * width_pct / 100.0))
    for side in range(4):
        prof = _profile(img, side, band)
        res.profiles.append(prof)
        edge = prof[: max(2, band // 10)].mean(axis=0)   # log gain at the edge
        mean = float(edge.mean())
        res.falloff[side] = float(1.0 - np.exp(mean))
        neutral = float(np.ptp(edge)) < 0.3 * abs(mean) + 0.01
        # monotonic: mostly darkening all the way from interior to the edge
        lum = prof.mean(axis=1)
        monotonic = float((np.diff(lum) > -1e-4).mean()) > 0.75
        res.is_shadow[side] = bool(-0.45 < mean < -0.035 and neutral and monotonic)
    return res


def correct_shading(img: np.ndarray, width_pct: float = 4.0, strength: float = 1.0,
                    sides=(True, True, True, True)) -> np.ndarray:
    if strength <= 0 or not any(sides):
        return img
    band = max(8, int(min(img.shape[:2]) * width_pct / 100.0))
    out = img.copy()
    for side in range(4):
        if not sides[side]:
            continue
        prof = _profile(img, side, band).mean(axis=1)    # neutral correction
        prof = np.clip(prof, -0.7, 0.7)
        gain = np.exp(-prof * float(strength)).astype(np.float32)
        view = _oriented(out, side)
        region = view[:, :band]
        region[:] = (region + _EPS) * gain[None, :, None] - _EPS
    return np.clip(out, 0.0, 1.0)
