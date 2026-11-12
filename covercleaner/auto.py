"""Auto mode: analyse a scan and choose the corrections to apply.

Each analysis step looks at the image as it will be *after* the previous
corrections (e.g. dust is searched for in the descreened image), and only
enables a fix when there is positive evidence that it is needed.  The
report explains every decision so the user can review and override it.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import defects, descreen, flat, geometry, ml, shading, tone
from .params import Params
from .pipeline import Pipeline

Progress = Optional[Callable[[str, float], None]]


def auto_params(source: np.ndarray, progress: Progress = None,
                pipeline: Optional[Pipeline] = None,
                mode: str = "traditional") -> tuple[Params, list[str]]:
    """Return (Params, human readable report) for a scan."""
    p = Params()
    p.mode = mode
    report: list[str] = []
    if mode == "ml" and not ml.available():
        report.append("ML pipeline selected but the model file is missing "
                      f"({ml.model_path()}); using traditional repairs.")
    elif mode == "ml":
        report.append("Pipeline: ML (neural inpainting for defects inside textured areas).")
    else:
        report.append("Pipeline: Traditional.")
    pipe = pipeline or Pipeline(source)

    def step(name, frac):
        if progress:
            progress(name, frac)

    # ---- 1. geometry -----------------------------------------------------
    step("Finding cover outline", 0.0)
    det = geometry.detect_cover(source)
    g = p.geometry
    if det.has_background:
        g.enabled = True
        g.corners = [[float(x), float(y)] for x, y in det.corners]
        g.bow = [[float(v) for v in row] for row in det.bow]
        g.use_bow = det.max_bow >= 1.5
        g.trim = [int(t) for t in det.trim]
        w, h = geometry.output_size(det.corners)
        report.append(f"Crop/deskew: cover {w} x {h} px found, rotated {det.angle_deg:+.2f} deg; "
                      f"edge trim L{g.trim[0]} T{g.trim[1]} R{g.trim[2]} B{g.trim[3]} px.")
        if g.use_bow:
            report.append(f"Flatten: paper edges bow by up to {det.max_bow:.1f} px; straightening.")
        weak = [geometry.SIDE_NAMES[i] for i, c in enumerate(det.side_confidence) if c < 0.3]
        if weak:
            report.append("  Low confidence on the " + ", ".join(weak)
                          + " edge - check the outline and drag the corners if needed.")
    else:
        g.enabled = True
        g.corners = None
        report.append("Crop/deskew: no scanner background found; keeping the full frame.")
    for note in det.notes:
        report.append("  " + note)
    img = pipe.run(p, upto="geometry")

    # ---- 2. halftone screen ----------------------------------------------
    step("Analysing print screen", 0.2)
    scr = descreen.analyze_screen(img)
    d = p.descreen
    if scr.found:
        d.enabled = True
        d.peaks = scr.peaks
        d.period = scr.period
        d.smoothing = 0.75
        d.protect_edges = 0.7
        report.append(f"Descreen: halftone dots every {scr.period:.2f} px "
                      f"(~{scr.lpi_at_600dpi:.0f} lpi if scanned at 600 dpi), "
                      f"{len(scr.peaks)} screen frequencies removed.")
    else:
        report.append("Descreen: no halftone pattern detected; skipped.")
    step("Removing print screen", 0.3)
    img = pipe.run(p, upto="descreen")

    # ---- 3. edge shading -------------------------------------------------
    step("Checking edge shading", 0.5)
    sh = shading.analyze_shading(img, p.shading.width_pct)
    if any(sh.is_shadow):
        p.shading.enabled = True
        p.shading.sides = [bool(x) for x in sh.is_shadow]
        txt = ", ".join(f"{shading.SIDES[i]} -{sh.falloff[i] * 100:.0f}%"
                        for i in range(4) if sh.is_shadow[i])
        report.append(f"Edge shading: shadow falloff at {txt}; evening out.")
    else:
        report.append("Edge shading: edges are evenly lit; skipped.")
    img = pipe.run(p, upto="shading")

    # ---- 4. dust and glare -----------------------------------------------
    step("Looking for dust and glare", 0.6)
    df = p.defects
    mask = defects.detect_dust(img, df.sensitivity, df.max_size)
    import cv2
    n_specks = max(0, cv2.connectedComponents(mask, connectivity=8)[0] - 1)
    if n_specks:
        df.enabled = True
        report.append(f"Dust: {n_specks} specks found ({mask.mean() / 2.55:.3f}% of the area); "
                      "inpainting.")
    else:
        report.append("Dust: none found.")
    veil = defects.glare_map(img, df.glare_sensitivity)
    glare_area = float((veil > 0.05).mean())
    if glare_area > 0.0005:
        # Soft highlights in a photo look just like glare, so this is never
        # switched on automatically.
        report.append(f"Glare: {glare_area * 100:.2f}% of the area looks like possible glare. "
                      "Not applied automatically - enable 'Glare' if these are reflections.")
    else:
        report.append("Glare: none detected.")
    img = pipe.run(p, upto="defects")

    # ---- 5. flat colour areas ----------------------------------------------
    step("Looking for flat colour areas", 0.7)
    fp = p.flat
    fr = flat.find_flat_regions(img, fp.tolerance, fp.min_area, fp.unify, fp.crisp_only,
                                fp.pure_bw)
    main = [c for c in fr.palette if c[3] >= 0.01]
    cols = ", ".join("%d,%d,%d" % (round(c[2] * 255), round(c[1] * 255), round(c[0] * 255))
                     for c in main[:5])
    # Only for covers that are mostly solid-colour artwork; a photo with a
    # few even patches would be posterised.
    if fr.coverage >= 0.5 and 1 <= len(main) <= 8:
        fp.enabled = True
        report.append(f"Flat colours: {fr.coverage * 100:.0f}% of the cover is solid colour "
                      f"({len(main)} main colours, R,G,B: {cols}); evening out.")
    elif fr.coverage >= 0.05:
        report.append(f"Flat colours: {fr.coverage * 100:.0f}% looks like solid colour - not "
                      "enough to be sure this is flat artwork. Enable 'Flat colour cleanup' "
                      "if it is.")
    else:
        report.append("Flat colours: none found (photographic cover); skipped.")
    img = pipe.run(p, upto="flat")

    # ---- 6. noise, tone, sharpening ----------------------------------------
    step("Measuring tone and noise", 0.8)
    ta = tone.analyze_tone(img)
    if ta.noise > 0.006:
        p.denoise.enabled = True
        p.denoise.strength = float(np.clip(ta.noise * 25.0, 0.15, 0.6))
        p.denoise.chroma = 0.5
        report.append(f"Denoise: residual noise sigma {ta.noise * 255:.1f}/255; "
                      f"strength {p.denoise.strength:.2f}.")
    else:
        report.append(f"Denoise: image is clean (sigma {ta.noise * 255:.1f}/255); skipped.")
    t = p.tone
    if ta.has_black or ta.has_white:
        t.enabled = True
        parts = []
        if ta.has_black:
            t.black = ta.black
            parts.append("ink black %d,%d,%d -> 0" % tuple(round(v * 255) for v in ta.black[::-1]))
        if ta.has_white:
            t.white = ta.white
            parts.append("paper white %d,%d,%d -> 255" % tuple(round(v * 255) for v in ta.white[::-1]))
        report.append("Levels (R,G,B): " + "; ".join(parts) + ".")
        if not ta.has_white:
            report.append("  No bare paper white found, highlights left unchanged.")
    else:
        report.append("Levels: no clear black or white reference; left unchanged.")
    if scr.found:
        p.sharpen.enabled = True
        p.sharpen.amount = 0.5
        p.sharpen.radius = float(np.clip(0.4 * scr.period, 0.8, 2.5))
        report.append(f"Sharpen: restoring edge contrast lost to the print screen "
                      f"(radius {p.sharpen.radius:.1f} px).")
    step("done", 1.0)
    return p, report
