"""Parameter set describing every correction applied to one scan.

The whole restoration is a pure function of (source image, Params), so a
Params object can be saved next to an image, edited by hand or by the GUI,
and re-applied at any time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional


@dataclass
class GeometryParams:
    """Crop, deskew and flatten: map the cover outline to a rectangle."""
    enabled: bool = True
    # Four corners in source pixels: TL, TR, BR, BL.  None = whole image.
    corners: Optional[list] = None
    # Per side (top, right, bottom, left) outward bulge of the paper edge
    # relative to the straight chord, sampled at evenly spaced points.
    bow: Optional[list] = None
    use_bow: bool = True
    # Extra pixels removed after rectification: left, top, right, bottom.
    trim: list = field(default_factory=lambda: [0, 0, 0, 0])
    rotate90: int = 0  # number of clockwise quarter turns
    disc: bool = False  # round disc: white outside the inscribed circle
    angle: float = 0.0  # fine rotation, degrees counter-clockwise


@dataclass
class DescreenParams:
    """Remove the halftone (print screen) pattern."""
    enabled: bool = False
    # Halftone peak frequencies in cycles/pixel [[fx, fy, strength], ...].
    peaks: list = field(default_factory=list)
    period: float = 0.0       # dominant dot pitch in pixels (0 = unknown)
    strength: float = 1.0     # 0..1 blend of the descreened result
    smoothing: float = 0.5    # 0 = notch filter only, 1 = full low-pass
    protect_edges: float = 0.7  # keep solid line art / text crisp


@dataclass
class ShadingParams:
    """Even out brightness falloff near the paper edges (lifted paper, spine)."""
    enabled: bool = False
    width_pct: float = 4.0    # band width as % of the short side
    strength: float = 1.0
    # Which sides to correct: left, top, right, bottom.
    sides: list = field(default_factory=lambda: [True, True, True, True])


@dataclass
class DefectParams:
    """Dust, specks, scratches (inpainted) and glare (subtracted)."""
    enabled: bool = False
    dust_dark: bool = True
    dust_bright: bool = True
    sensitivity: float = 0.5  # 0..1, higher finds fainter specks
    max_size: float = 0.6     # largest speck, % of the short side
    glare: bool = False
    glare_strength: float = 1.0
    glare_sensitivity: float = 0.5


@dataclass
class FlatParams:
    """Even out areas that were one solid colour in the original artwork."""
    enabled: bool = False
    tolerance: float = 0.06   # how much a region may vary and still be "flat"
    min_area: float = 0.05    # smallest region, % of the image area
    strength: float = 1.0
    unify: bool = True        # same colour everywhere -> one palette colour
    crisp_only: bool = True   # only regions with crisp, artwork-like outlines
    pure_bw: bool = True      # near-black -> black, near-white -> white
    clean_edges: bool = True  # rebuild anti-aliased edges between flat areas
    fill_marks: float = 0.0   # absorb enclosed marks up to this size (% short side)


@dataclass
class DenoiseParams:
    enabled: bool = False
    strength: float = 0.3     # 0..1
    chroma: float = 0.5       # extra smoothing of colour noise, 0..1


@dataclass
class ToneParams:
    """Undo paper/ink/scanner tone compression."""
    enabled: bool = False
    black: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # B, G, R
    white: list = field(default_factory=lambda: [1.0, 1.0, 1.0])  # B, G, R
    gamma: float = 1.0
    saturation: float = 1.0
    contrast: float = 0.0     # -1..1 S-curve
    temperature: float = 0.0  # -1..1 blue <-> yellow
    tint: float = 0.0         # -1..1 green <-> magenta


@dataclass
class SharpenParams:
    enabled: bool = False
    amount: float = 0.5
    radius: float = 1.2


@dataclass
class OutputParams:
    scale: float = 1.0        # resample factor applied last
    bit_depth: int = 8


@dataclass
class Params:
    # "traditional": classical image processing only.
    # "ml": additionally uses neural networks where they do better
    #        (currently LaMa inpainting for defects inside textured areas).
    mode: str = "traditional"
    geometry: GeometryParams = field(default_factory=GeometryParams)
    descreen: DescreenParams = field(default_factory=DescreenParams)
    shading: ShadingParams = field(default_factory=ShadingParams)
    defects: DefectParams = field(default_factory=DefectParams)
    flat: FlatParams = field(default_factory=FlatParams)
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    tone: ToneParams = field(default_factory=ToneParams)
    sharpen: SharpenParams = field(default_factory=SharpenParams)
    output: OutputParams = field(default_factory=OutputParams)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        p = cls()
        for f in fields(cls):
            sub = getattr(p, f.name)
            if not is_dataclass(sub):
                if f.name in d:
                    setattr(p, f.name, d[f.name])
                continue
            for k, v in (d.get(f.name) or {}).items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
        return p

    @classmethod
    def from_json(cls, s: str) -> "Params":
        return cls.from_dict(json.loads(s))

    def copy(self) -> "Params":
        return Params.from_dict(json.loads(json.dumps(self.to_dict())))


def stage_key(obj: Any) -> str:
    """Stable string for change detection of one parameter group."""
    return json.dumps(asdict(obj) if is_dataclass(obj) else obj, sort_keys=True)
