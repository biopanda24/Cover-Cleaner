"""The restoration pipeline: staged, cached, driven entirely by Params.

Stage order (each works on the output of the previous one):

    geometry -> descreen -> shading -> defects -> flat -> denoise
             -> tone -> sharpen -> output

Every stage result is cached together with a key built from the parameters
of that stage and all stages before it, so changing e.g. the saturation only
recomputes the last stages.
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from . import defects as _defects
from . import descreen as _descreen
from . import flat as _flat
from . import geometry as _geometry
from . import shading as _shading
from . import tone as _tone
from .params import Params, stage_key

STAGES = ("geometry", "descreen", "shading", "defects", "flat", "denoise", "tone", "sharpen",
          "output")
Progress = Optional[Callable[[str, float], None]]


class Pipeline:
    def __init__(self, source: np.ndarray):
        self.source = source
        self._cache: dict[str, tuple[str, np.ndarray]] = {}
        # Hand-painted masks, in the coordinates of the geometry stage output.
        self.heal_mask: Optional[np.ndarray] = None      # extra areas to repair
        self.protect_mask: Optional[np.ndarray] = None   # never touch these
        self._mask_version = 0
        self.defect_mask: Optional[np.ndarray] = None    # last mask actually used
        self.flat_mask: Optional[np.ndarray] = None      # pixels evened out by 'flat'

    # ---- user masks ------------------------------------------------------
    def masks_changed(self) -> None:
        self._mask_version += 1

    def ensure_masks(self, shape) -> None:
        h, w = shape[:2]
        if self.heal_mask is None or self.heal_mask.shape != (h, w):
            self.heal_mask = np.zeros((h, w), np.uint8)
            self.protect_mask = np.zeros((h, w), np.uint8)
            self._mask_version += 1

    def clear_masks(self) -> None:
        if self.heal_mask is not None:
            self.heal_mask[:] = 0
            self.protect_mask[:] = 0
            self._mask_version += 1

    # ---- stages ----------------------------------------------------------
    def _geometry(self, img, p: Params):
        g = p.geometry
        if not g.enabled:
            return img
        return _geometry.apply_geometry(img, g.corners, g.bow if g.use_bow else None,
                                        g.trim, g.rotate90)

    def _descreen(self, img, p: Params):
        d = p.descreen
        if not d.enabled or not d.peaks:
            return img
        return _descreen.descreen(img, d.peaks, d.period, d.strength, d.smoothing,
                                  d.protect_edges)

    def _shading(self, img, p: Params):
        s = p.shading
        if not s.enabled:
            return img
        return _shading.correct_shading(img, s.width_pct, s.strength, s.sides)

    def _defects(self, img, p: Params):
        d = p.defects
        self.ensure_masks(img.shape)
        mask = np.zeros(img.shape[:2], np.uint8)
        out = img
        if d.enabled:
            if d.glare:
                out = _defects.remove_glare(out, d.glare_sensitivity, d.glare_strength)
            if d.dust_dark or d.dust_bright:
                mask = _defects.detect_dust(out, d.sensitivity, d.max_size,
                                            d.dust_dark, d.dust_bright)
        # Hand-painted repairs apply even when automatic detection is off.
        mask = cv2.bitwise_or(mask, self.heal_mask)
        mask[self.protect_mask > 0] = 0
        self.defect_mask = mask
        return _defects.inpaint(out, mask) if mask.any() else out

    def _flat(self, img, p: Params):
        f = p.flat
        if not f.enabled:
            self.flat_mask = None
            return img
        out, self.flat_mask = _flat.flatten(
            img, f.tolerance, f.min_area, f.strength, f.unify, f.crisp_only,
            f.fill_marks, f.pure_bw, f.clean_edges, protect=self.protect_mask)
        return out

    def _denoise(self, img, p: Params):
        d = p.denoise
        return _tone.denoise(img, d.strength, d.chroma) if d.enabled else img

    def _tone(self, img, p: Params):
        t = p.tone
        if not t.enabled:
            return img
        return _tone.apply_tone(img, t.black, t.white, t.gamma, t.saturation,
                                t.contrast, t.temperature, t.tint)

    def _sharpen(self, img, p: Params):
        s = p.sharpen
        return _tone.sharpen(img, s.amount, s.radius) if s.enabled else img

    def _output(self, img, p: Params):
        return _tone.resize(img, p.output.scale)

    # ---- driver ----------------------------------------------------------
    def _key(self, name: str, p: Params) -> str:
        if name == "output":
            return str(p.output.scale)
        key = stage_key(getattr(p, name))
        if name in ("defects", "flat"):
            key += f"|masks{self._mask_version}"
        return key

    def run(self, params: Params, upto: str = "output", progress: Progress = None) -> np.ndarray:
        """Run (or reuse) all stages up to and including `upto`."""
        img = self.source
        key = ""
        last = STAGES.index(upto)
        for i, name in enumerate(STAGES[: last + 1]):
            if name == "defects":
                # geometry may have changed the image size since masks were made
                self.ensure_masks(img.shape)
            key += "/" + name + ":" + self._key(name, params)
            hit = self._cache.get(name)
            if hit is not None and hit[0] == key:
                img = hit[1]
                continue
            if progress:
                progress(name, i / (last + 1))
            img = getattr(self, "_" + name)(img, params)
            self._cache[name] = (key, img)
        if progress:
            progress("done", 1.0)
        return img

    def stage_output(self, name: str) -> Optional[np.ndarray]:
        hit = self._cache.get(name)
        return None if hit is None else hit[1]


def restore(source: np.ndarray, params: Params, progress: Progress = None) -> np.ndarray:
    """One-shot convenience wrapper."""
    return Pipeline(source).run(params, progress=progress)
