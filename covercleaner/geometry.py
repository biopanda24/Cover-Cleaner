"""Find the cover inside the scan and map it onto a clean rectangle.

Detection works in two passes:

1. Coarse: the scanner background (lid) colour is estimated from the image
   border, everything that differs from it is collected, and a rotated
   bounding rectangle is fitted.
2. Fine: for each side of that rectangle a band of the full-resolution image
   is resampled so the expected edge is a straight horizontal line.  The
   paper edge is located in every column of the band (outermost sharp
   transition, stepping over the soft lid shadow), and a robust line plus a
   smooth "bow" curve are fitted through those points.

The result is four corners plus a per-side bow, which a Coons patch turns
into a remap.  With zero bow this is a plain deskew + crop; with bow it also
straightens edges that are curved because the paper was not lying flat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .imageio import luminance

BOW_SAMPLES = 9
SIDE_NAMES = ("top", "right", "bottom", "left")


@dataclass
class CoverDetection:
    corners: np.ndarray                      # (4, 2) TL, TR, BR, BL
    bow: np.ndarray                          # (4, BOW_SAMPLES) outward px
    has_background: bool = True
    confidence: float = 0.0                  # 0..1, weakest side
    side_confidence: list = field(default_factory=lambda: [0.0] * 4)
    angle_deg: float = 0.0
    trim: list = field(default_factory=lambda: [0, 0, 0, 0])  # l, t, r, b
    max_bow: float = 0.0
    bg_color: tuple = (1.0, 1.0, 1.0)
    notes: list = field(default_factory=list)
    off_sides: list = field(default_factory=list)   # image borders the cover runs past
    disc: list = None                            # [cx, cy, r] for a round disc


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def full_corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL for a rectangle rotated by < 45 deg."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # clockwise in image coordinates
    # the corner pair forming the "top" side has the smallest mean y
    best = min(range(4), key=lambda i: (pts[i, 1] + pts[(i + 1) % 4, 1]) / 2.0)
    return np.roll(pts, -best, axis=0)


def _intersect(p1, d1, p2, d2):
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]], np.float64)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, b)[0]
    return np.array([p1[0] + d1[0] * t, p1[1] + d1[1] * t])


def _side_frame(a, b, center):
    """Unit tangent, outward unit normal and length of side a->b."""
    d = b - a
    length = float(np.hypot(*d))
    t = d / max(length, 1e-9)
    n = np.array([t[1], -t[0]])
    if np.dot((a + b) / 2 - center, n) < 0:
        n = -n
    return t, n, length


# --------------------------------------------------------------------------
# coarse pass
# --------------------------------------------------------------------------

IMAGE_SIDES = ("top", "bottom", "left", "right")


def _border_strips(a: np.ndarray, m: int) -> dict:
    return {"top": a[:m], "bottom": a[-m:], "left": a[:, :m], "right": a[:, -m:]}


def _background_model(small: np.ndarray, m: int, off=()):
    """Per-pixel estimate of the scanner background (lid) colour.

    The lid is not always evenly lit: where it does not lie flat, e.g. lifted
    by a booklet's spine, a whole side of the scan can be several levels
    darker than the rest.  Each image border gives a colour profile along
    that side (median filtered, so dust and small objects drop out); inside
    the image the four profiles are blended by inverse squared distance, so
    every pixel is compared with the lid colour of its nearest side.
    Borders listed in `off` are covered by the cover itself (it runs past
    the edge of the scan) and only contribute the overall lid colour.

    Returns (global background colour, model image).
    """
    sh, sw = small.shape[:2]
    strips = _border_strips(small, m)
    lid_strips = [v for k, v in strips.items() if k not in off] or list(strips.values())
    glob = np.median(np.concatenate([s.reshape(-1, 3) for s in lid_strips]), axis=0)

    def profile(side: str, values: np.ndarray, length: int) -> np.ndarray:
        if side in off:
            return np.tile(glob.astype(np.float32), (length, 1))
        k = max(3, int(0.08 * length) | 1)
        prof = median_filter(values, size=(k, 1), mode="nearest")
        # Far from the overall lid colour = the cover itself reaches the
        # image edge here; fall back to the overall colour.
        far = np.abs(prof - glob).max(axis=1) > 0.10
        prof[far] = glob
        return prof.astype(np.float32)

    top = profile("top", np.median(small[:m], axis=0), sw)          # (sw, 3)
    bottom = profile("bottom", np.median(small[-m:], axis=0), sw)
    left = profile("left", np.median(small[:, :m], axis=1), sh)      # (sh, 3)
    right = profile("right", np.median(small[:, -m:], axis=1), sh)
    y = np.arange(sh, dtype=np.float32)[:, None] + 1.0
    x = np.arange(sw, dtype=np.float32)[None, :] + 1.0
    wt, wb = 1.0 / y ** 2, 1.0 / (sh + 1.0 - y) ** 2           # (sh, 1)
    wl, wr = 1.0 / x ** 2, 1.0 / (sw + 1.0 - x) ** 2           # (1, sw)
    model = (wt[..., None] * top[None] + wb[..., None] * bottom[None]
             + wl[..., None] * left[:, None] + wr[..., None] * right[:, None])
    model /= (wt + wb + wl + wr)[..., None]
    return glob, model


def _coarse_rect(img: np.ndarray, det: CoverDetection):
    h, w = img.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    sh, sw = small.shape[:2]
    m = max(3, int(round(0.012 * max(sh, sw))))
    glob, model = _background_model(small, m)
    dist = np.abs(small - model).max(axis=2)
    # A border mostly unlike the lid is covered by the cover itself: it runs
    # past that edge of the scan.  Up to two such borders are accepted.
    def covered_run(strip: np.ndarray, along: int) -> float:
        """Longest stretch of the border (fraction) where the cover lies."""
        hit = np.median(strip > 0.06, axis=along) > 0.5
        best = cur = 0
        for v in hit:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best / max(1, len(hit))

    # only the outermost pixels: a cover lying close to the edge of the scan
    # must not count as running past it
    strips = _border_strips(dist, 2)
    off = tuple(k for k, v in strips.items()
                if covered_run(v, 0 if k in ("top", "bottom") else 1) > 0.2)
    if 0 < len(off) <= 2 and set(off) not in ({"top", "bottom"}, {"left", "right"}):
        glob, model = _background_model(small, m, off)
        dist = np.abs(small - model).max(axis=2)
        det.off_sides = list(off)
        det.notes.append("Cover runs past the " + " and ".join(off) + " edge of the scan; "
                         "that edge is kept as it is.")
    det.bg_color = tuple(float(x) for x in glob)
    bdist = np.concatenate([v.ravel() for k, v in _border_strips(dist, m).items()
                            if k not in det.off_sides])
    if len(det.off_sides) != len(off) or np.percentile(bdist, 85) > 0.10:
        det.notes.append("Image border is not a uniform scanner background; "
                         "assuming the scan is already cropped.")
        return None
    if float(np.abs(model - glob).max()) > 0.035:
        det.notes.append("Scanner lid is unevenly lit (darker along one side); "
                         "compensated.")
    mad = float(np.median(bdist))
    thr = max(0.045, 8.0 * mad)
    mask = (dist > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(40, int(0.0004 * sh * sw))
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # dust is small; the cover (or its shadow line) is large or long
        keep[i] = area >= min_area or max(bw, bh) >= 0.25 * min(sh, sw)
    pts = np.column_stack(np.nonzero(keep[labels])[::-1]).astype(np.float32)
    if len(pts) < 100:
        det.notes.append("No cover found against the background.")
        return None
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float64)
    area = rect[1][0] * rect[1][1]
    if area < 0.08 * sh * sw:
        det.notes.append("Detected object is too small to be the cover.")
        return None
    # A disc (CD / DVD) fills its enclosing circle but only pi/4 of its box.
    (cx, cy), cr = cv2.minEnclosingCircle(hull)
    hull_area = float(cv2.contourArea(hull))
    if not det.off_sides and hull_area > 0.92 * np.pi * cr * cr and hull_area < 0.85 * area:
        det.disc = [cx / scale, cy / scale, cr / scale]
    return _order_corners(box / scale)


def _refine_disc(img_blur: np.ndarray, cx: float, cy: float, r: float):
    """Fit the disc's rim at full resolution: edge picks along 720 rays, then a
    robust least-squares circle."""
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    reach = max(30.0, 0.03 * r)
    o = np.arange(reach, -reach - 1, -1.0)                  # outside -> inside
    rad = r + o[None, :]
    px = (cx + np.cos(ang)[:, None] * rad).astype(np.float32)
    py = (cy + np.sin(ang)[:, None] * rad).astype(np.float32)
    band = cv2.remap(img_blur, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    e = _pick_edges(band, o)
    ok = np.isfinite(e)
    if ok.sum() < 100:
        return cx, cy, r, 0.0
    x = cx + np.cos(ang[ok]) * (r + e[ok])
    y = cy + np.sin(ang[ok]) * (r + e[ok])
    keep = np.ones(len(x), bool)
    for _ in range(4):
        # algebraic circle fit: x^2 + y^2 + D x + E y + F = 0
        A = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
        rhs = -(x[keep] ** 2 + y[keep] ** 2)
        D, E, F = np.linalg.lstsq(A, rhs, rcond=None)[0]
        ccx, ccy = -D / 2, -E / 2
        rr = float(np.sqrt(max(ccx ** 2 + ccy ** 2 - F, 1.0)))
        res = np.abs(np.hypot(x - ccx, y - ccy) - rr)
        keep = res < max(2.0, 2.5 * float(np.median(res[keep])))
    conf = float((res < 3.0).sum()) / len(ang)
    return float(ccx), float(ccy), rr, conf


# --------------------------------------------------------------------------
# fine pass
# --------------------------------------------------------------------------

def _sample_band(img_blur, a, t, n, length, radius, step=2.0, margin=0.03, inner=None):
    """Resample a band around side a->(a+t*length).

    Rows = positions along the side, columns = offsets from +radius
    (outside) to -inner (inside, default -radius) in 1 px steps.
    """
    s = np.arange(length * margin, length * (1 - margin), step)
    o = np.arange(radius, -(radius if inner is None else inner) - 1, -1.0)
    px = a[0] + np.outer(s, np.full_like(o, t[0])) + np.outer(np.ones_like(s), o * n[0])
    py = a[1] + np.outer(s, np.full_like(o, t[1])) + np.outer(np.ones_like(s), o * n[1])
    band = cv2.remap(img_blur, px.astype(np.float32), py.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return s, o, band


SHARP_NEAR = 4      # px either side over which a paper edge must show its step


def _pick_edges(band: np.ndarray, o: np.ndarray, sens: float = 1.0):
    """Locate the paper edge in every row of a band. Returns offsets (nan = none).

    The paper edge is the outermost *sharp* transition.  Lid shadows and a
    lid lifted by a booklet spine change the brightness gradually over tens
    to hundreds of pixels.  Each monotone transition is taken as a whole
    (ramp plus the step at its end) and located at its steepest point; it
    counts only if that point changes the colour visibly within +-4 px.
    """
    # Texture along the edge, measured before averaging along it: halftone
    # print varies from dot to dot, a shadow on the lid is uniform.
    tex = np.abs(band - cv2.GaussianBlur(band, (1, 5), 1.0)).max(axis=2)
    tex = cv2.blur(tex, (1, 15))
    band = cv2.blur(band, (1, 15))                       # average along the edge
    lum = luminance(band)
    dl = np.gradient(lum, axis=1)
    dc = np.gradient(band, axis=1)
    mag = np.sqrt((dc ** 2).sum(axis=2))
    g = np.where(dl >= 0, mag, -mag)                     # signed colour gradient
    outer = g[:, : max(4, g.shape[1] // 4)]
    noise = 1.4826 * np.median(np.abs(outer - np.median(outer)))
    thr = max(6.0 * noise, 0.010) / sens
    min_step = max(0.02 / sens, 2.0 * thr)
    out = np.full(g.shape[0], np.nan)
    ncol = g.shape[1]
    cs = np.concatenate([np.zeros((band.shape[0], 1, 3), np.float64),
                         np.cumsum(band, axis=1, dtype=np.float64)], axis=1)
    gs = cv2.blur(g.astype(np.float32), (9, 1))         # smoothed across the edge

    def step(r, p, k):
        """Colour difference between the k px after and the k px before p."""
        a0, a1 = max(0, p - k), p
        b0, b1 = p + 1, min(ncol, p + 1 + k)
        if a1 <= a0 or b1 <= b0:
            return 0.0
        before = (cs[r, a1] - cs[r, a0]) / (a1 - a0)
        after = (cs[r, b1] - cs[r, b0]) / (b1 - b0)
        return float(np.abs(after - before).max())

    for r in range(g.shape[0]):
        row = g[r]
        cand = np.flatnonzero(np.abs(row) > thr)
        p = -1
        sgn = 0.0
        j = -1
        ci = 0
        rs = gs[r]
        while ci < cand.size:
            i0 = int(cand[ci])
            s0 = np.sign(row[i0])
            # The whole monotone transition (a shadow ramp running into the
            # paper edge included) is one run; small dips in the gradient
            # do not split it.  Its steepest point is the edge.
            j0 = i0
            while j0 + 1 < ncol and rs[j0 + 1] * s0 > thr * 0.1:
                j0 += 1
            seg = row[i0:j0 + 1] * s0
            pk = i0 + int(np.argmax(seg))
            if step(r, pk, SHARP_NEAR) >= min_step:
                p, sgn, j = pk, s0, j0
                break
            ci = int(np.searchsorted(cand, j0 + 1))        # too soft: look further in
        if p < 2:                                         # none, or beyond the band
            continue
        if sgn < 0:
            # A darkening edge followed shortly by a brightening one, with a
            # smooth (unprinted) stretch between them, is a shadow line on
            # the lid next to the paper: the paper edge is the rise.  Dark
            # print right inside the edge has halftone texture and is kept.
            k = p + 2
            lim = min(ncol - 1, p + 22)
            while k <= lim and row[k] <= thr:
                k += 1
            if k <= lim:
                k2 = k
                while k2 + 1 < ncol and row[k2 + 1] > thr * 0.3:
                    k2 += 1
                rise = k + int(np.argmax(row[k:k2 + 1]))
                a0, a1 = min(p + 3, ncol - 1), min(ncol, max(p + 4, rise - 3))
                seg_c = band[r, a0:a1]
                neutral = float((seg_c.max(axis=1) - seg_c.min(axis=1)).mean()) < 0.15
                smooth = neutral and float(tex[r, a0:a1].mean()) < 0.006
                up = step(r, rise, SHARP_NEAR)
                if smooth and up >= min_step and up >= 0.2 * step(r, p, SHARP_NEAR):
                    p = rise
        # sub-pixel parabola refinement
        if 0 < p < ncol - 1:
            y0, y1, y2 = abs(row[p - 1]), abs(row[p]), abs(row[p + 1])
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            frac = float(np.clip(frac, -0.5, 0.5))
        else:
            frac = 0.0
        out[r] = o[p] - frac                              # o decreases with index
    return out


def _ransac_poly(s, e, deg=1, tol=1.5, iters=400, max_sag=0.0):
    """Robust polynomial e(s) of degree 1 or 2.

    Returns (coefficients [a, b, c] of a*s^2 + b*s + c, inlier mask) or None.
    """
    rng = np.random.default_rng(0)
    n = len(s)
    if n < deg + 1:
        return None
    span = float(np.ptp(s)) + 1e-9
    mid = float(s.min()) + span / 2
    best, best_cnt = None, -1
    for _ in range(iters):
        idx = rng.integers(0, n, deg + 1)
        ss = np.sort(s[idx])
        if np.diff(ss).min() < 0.15 * span / deg:
            continue
        coef = np.polyfit(s[idx], e[idx], deg)
        if deg == 2:
            if abs(coef[0]) * span * span / 4 > max_sag:
                continue
            slope = 2 * coef[0] * mid + coef[1]
        else:
            slope = coef[0]
        if abs(slope) > 0.08:
            continue
        inl = np.abs(e - np.polyval(coef, s)) < tol
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, inl
    if best is None or best_cnt < deg + 2:
        return None
    inl = best
    for _ in range(2):
        coef = np.polyfit(s[inl], e[inl], deg)
        new = np.abs(e - np.polyval(coef, s)) < tol
        if new.sum() < deg + 2:
            break
        inl = new
    full = np.zeros(3)
    full[3 - len(coef):] = coef
    return full, inl


def _fit_side(s, e):
    """Fit the paper edge e(s): a line, or a gentle curve if the paper bows.

    Prefers the outermost consistent line.  Returns (coef, confidence,
    inlier mask over all samples) or None.
    """
    ok = np.isfinite(e)
    if ok.sum() < max(12, 0.10 * len(e)):
        return None
    s1, e1 = s[ok], e[ok]
    fit = _ransac_poly(s1, e1, 1)
    if fit is None:
        return None
    coef, inl = fit
    # Points clearly outside the dominant line may be the real paper edge
    # (the dominant line being a printed frame just inside it).
    outside = (e1 - np.polyval(coef, s1)) > 3.0
    if outside.sum() >= max(10, 0.18 * len(e1)):
        fit2 = _ransac_poly(s1[outside], e1[outside], 1)
        if fit2 is not None and fit2[1].sum() >= 0.6 * outside.sum() \
                and np.ptp(s1[outside][fit2[1]]) > 0.4 * np.ptp(s1):
            coef = fit2[0]
            inl = np.abs(e1 - np.polyval(coef, s1)) < 1.5
    # Paper that is not flat has gently curved edges: accept a parabola
    # when it explains clearly more of the edge than the straight line.
    span = float(np.ptp(s1))
    band = np.abs(e1 - np.polyval(coef, s1)) < max(8.0, 0.012 * span)
    if band.sum() >= 30:
        quad = _ransac_poly(s1[band], e1[band], 2, max_sag=0.012 * span)
        if quad is not None and quad[1].sum() > 1.15 * inl.sum() + 5:
            coef = quad[0]
            inl = np.abs(e1 - np.polyval(coef, s1)) < 1.5
    mask = np.zeros(len(e), bool)
    mask[np.flatnonzero(ok)[inl]] = True
    # Confidence: share of the side whose edge lies within 3 px of the fit.
    # (Edges of paper lifted off the glass are slightly out of focus, so
    # the 1.5 px inlier tolerance used for fitting would understate it.)
    conf = float((np.abs(e1 - np.polyval(coef, s1)) < 3.0).sum()) / len(e)
    return coef, conf, mask


LID_TEXTURE = 0.004     # high-frequency level: bare lid < ~0.0025, print > ~0.005


def _tint_edge(band: np.ndarray, o: np.ndarray, s: np.ndarray, chunks: int = 12):
    """Find a paper edge by colour, ignoring brightness.

    The lid is neutral grey however it is shaded (a lid lifted by a booklet
    spine fades from white to grey over hundreds of pixels), while paper -
    even "white" cover stock - has a tint.  The band is averaged along the
    side in chunks; in each, brightness is removed (each colour minus the
    pixel's mean) and the outermost point from which the colour stays away
    from the lid's for 15 px is taken.  The chunks must agree on a straight
    line.  Returns (coef, confidence) like _fit_side, or None.
    """
    rows = band.shape[0]
    if rows < chunks * 4:
        return None
    ps, pe = [], []
    for k in range(chunks):
        r0, r1 = k * rows // chunks, (k + 1) * rows // chunks
        seg = band[r0:r1].mean(axis=0)                       # (cols, 3)
        tint = seg - seg.mean(axis=1, keepdims=True)
        tint = cv2.GaussianBlur(tint[None].astype(np.float32), (0, 0), 1.5)[0]
        lid = np.median(tint[:24], axis=0)
        dev = np.abs(tint - lid).max(axis=1)
        far = dev > 0.02                                      # ~5 levels of tint
        run = np.convolve(far.astype(np.int32), np.ones(15, np.int32), "valid") == 15
        idx = np.flatnonzero(run)
        if idx.size == 0 or idx[0] < 4:
            continue
        i = int(idx[0])
        plateau = float(np.median(dev[i + 5:i + 15]))
        j = i
        while j > 0 and dev[j - 1] > 0.5 * plateau:           # back to the half-way point
            j -= 1
        ps.append(float(s[(r0 + r1) // 2]))
        pe.append(float(o[j]))
    if len(ps) < 0.7 * chunks:
        return None
    ps, pe = np.array(ps), np.array(pe)
    best = None
    for i in range(len(ps)):                                  # robust: try each pair
        for j in range(i + 1, len(ps)):
            if ps[j] == ps[i]:
                continue
            k_ = (pe[j] - pe[i]) / (ps[j] - ps[i])
            if abs(k_) > 0.03:
                continue
            inl = np.abs(pe - (pe[i] + k_ * (ps - ps[i]))) < 6.0
            if best is None or inl.sum() > best.sum():
                best = inl
    if best is None or best.sum() < 0.55 * chunks:
        return None
    coef = np.polyfit(ps[best], pe[best], 1)
    full = np.zeros(3)
    full[1:] = coef
    # confidence scaled so a clear majority of chunks (55 %) passes the 0.7 bar
    return full, min(1.0, float(best.sum()) / chunks / 0.55 * 0.7)


def _faint_edge(band: np.ndarray, o: np.ndarray, s: np.ndarray, chunks: int = 10):
    """Find a faint but straight paper edge (white paper on a white lid).

    The band is averaged along the side in `chunks` pieces, which lowers the
    noise enough to see steps of 2-3 levels.  In each piece the outermost
    small but sharp step is taken; the pieces must agree on a straight line.
    Returns (coef, confidence) like _fit_side, or None.
    """
    rows = band.shape[0]
    if rows < chunks * 4:
        return None
    lum = luminance(band)
    picks_s, picks_e = [], []
    for k in range(chunks):
        r0, r1 = k * rows // chunks, (k + 1) * rows // chunks
        prof = cv2.GaussianBlur(lum[r0:r1].mean(axis=0)[None, :].astype(np.float32),
                                (0, 0), 1.5)[0]
        g = np.gradient(prof)
        outer = g[: max(8, len(g) // 6)]
        noise = 1.4826 * float(np.median(np.abs(outer - np.median(outer)))) + 1e-5
        thr = max(5.0 * noise, 0.0015)
        idx = np.flatnonzero(np.abs(g) > thr)
        for i in idx:
            if i < 6 or i > len(g) - 6:
                continue
            stepv = abs(prof[i + 4] - prof[i - 4])
            if stepv >= 0.007:                              # ~2 levels, within 8 px
                picks_s.append(float(s[(r0 + r1) // 2]))
                picks_e.append(float(o[i]))
                break
    if len(picks_s) < 0.7 * chunks:
        return None
    ps, pe = np.array(picks_s), np.array(picks_e)
    coef = np.polyfit(ps, pe, 1)
    res = np.abs(pe - np.polyval(coef, ps))
    inl = res < 4.0
    if inl.sum() < 0.7 * chunks or abs(coef[0]) > 0.03:
        return None
    coef = np.polyfit(ps[inl], pe[inl], 1)
    full = np.zeros(3)
    full[1:] = coef
    return full, float(inl.sum()) / chunks


def _off_border_line(a, b, n, w, h, off_sides):
    """If side a-b lies along an image border the cover runs past, return
    that border as a (point, direction) line, else None."""
    if not off_sides:
        return None
    tol = 0.02 * min(w, h)
    if "top" in off_sides and n[1] < -0.7 and max(a[1], b[1]) < tol:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
    if "bottom" in off_sides and n[1] > 0.7 and min(a[1], b[1]) > h - 1 - tol:
        return np.array([0.0, h - 1.0]), np.array([1.0, 0.0])
    if "left" in off_sides and n[0] < -0.7 and max(a[0], b[0]) < tol:
        return np.array([0.0, 0.0]), np.array([0.0, 1.0])
    if "right" in off_sides and n[0] > 0.7 and min(a[0], b[0]) > w - 1 - tol:
        return np.array([w - 1.0, 0.0]), np.array([0.0, 1.0])
    return None


def _on_image_border(a, b, w, h, tol=4.0) -> bool:
    """True when side a-b runs along one of the image borders."""
    return (max(a[0], b[0]) < tol or min(a[0], b[0]) > w - 1 - tol
            or max(a[1], b[1]) < tol or min(a[1], b[1]) > h - 1 - tol)


def _cut_looks_like_lid(band, o, s, coef, lid_colour) -> bool:
    """Is everything between the rough line (offset 0) and a newly found edge
    further inside plain scanner lid?  Printed picture has halftone or
    colour; the lid is smooth and close to the background colour."""
    edge = np.polyval(coef, s)[:, None]
    region = (o[None, :] <= 0) & (o[None, :] > edge + 6)
    if region.sum() < 50:
        return True
    hp = np.abs(band - cv2.GaussianBlur(band, (0, 0), 2.0)).max(axis=2)
    texture = float(hp[region].mean())
    colour = np.median(band[region], axis=0)
    return texture < LID_TEXTURE and float(np.abs(colour - lid_colour).max()) < 0.12


WEAR_DEPTH = 40         # px inspected inward from each paper edge
WEAR_REF = (24, 37)     # depths whose colour stands for "the picture there"


def edge_wear(img_blur, a, t, n, length, max_depth):
    """Width (px) of a worn or discoloured band along one paper edge.

    Worn edges, whitened folds, the white core of a cut edge and lid light
    bleeding onto dark covers all shift the colour near the edge the same
    way along (almost) the whole side.  Picture that simply runs to the
    edge is sometimes lighter, sometimes darker there.  So a depth counts as
    worn when at least 75% of positions along the side differ from their
    own colour further inside in one direction and hardly any in the other.
    Wear is also strongest at the very edge and fades inward; a band of
    constant strength is a printed stripe or a picture feature running
    parallel to the edge, and is kept.

    Returns (width, direction) with direction "lighter" / "darker" / "".
    """
    s, o, band = _sample_band(img_blur, a, t, n, length, 0, margin=0.05, inner=WEAR_DEPTH)
    if len(s) < 20:
        return 0, ""
    band = cv2.blur(band, (1, 9))                                # along the side
    ref = np.median(band[:, WEAR_REF[0]:WEAR_REF[1]], axis=1)    # (pos, 3)
    diff = band - ref[:, None, :]
    lum = luminance(diff)
    size = np.abs(diff).max(axis=2)
    shifted = size > 0.03                                        # about 8 levels
    lighter = shifted & (lum > 0)
    darker = shifted & (lum < 0)
    ref_lum = luminance(ref[:, None, :])[:, 0]
    best = (0, "")
    for kind, hit, other, room in (("lighter", lighter, darker, ref_lum < 0.90),
                                   ("darker", darker, lighter, ref_lum > 0.08)):
        if room.sum() < 0.3 * len(s):
            continue                     # the edge can't get any lighter/darker
        frac = hit[room].mean(axis=0)
        opp = other[room].mean(axis=0)
        strength = np.median(size[room], axis=0)                 # per depth
        end = 0
        # depths 0-1 straddle the edge itself and are trimmed anyway
        for d in range(2, max_depth + 1):
            if frac[d] < 0.75 or opp[d] > 0.15:
                end = d
                break
        else:
            end = max_depth
        visible = strength[2] >= 0.06                            # ~15 levels
        # fading inside the band, not just stopping at its end
        fades = strength[end] <= 0.5 * strength[2] and \
            (end - 3 <= 2 or strength[end - 3] <= 0.7 * strength[2])
        if end > 2 and visible and fades and end > best[0]:
            best = (end, kind)
    return best


def _margin_edge(band, o, s, coef, length):
    """A weaker straight paper edge outside an edge found further in.

    White cover stock is often as bright as the lid, so the first clear edge
    can be printing with a white paper margin outside it.  Look between the
    outside and that edge at double sensitivity for a straight edge of its
    own (the paper meeting the lid).  Returns (coef, confidence, picks) or
    None.
    """
    stop = float(np.polyval(coef, length / 2)) + 10
    keep = o >= stop
    if keep.sum() <= 20:
        return None
    e = np.full(len(s), np.nan)
    sub = _pick_edges(band[:, keep], o[keep], sens=2.0)
    e[:] = sub
    fit = _fit_side(s, e)
    if fit is None or fit[1] < 0.35:
        return None
    return fit[0], fit[1], e


def _cut_is_bright_lid(band, o, s, coef, lid_colour) -> bool:
    """The cut-away region is indistinguishable from bright, even lid (as
    opposed to a darker, shaded part of the lid)."""
    edge = np.polyval(coef, s)[:, None]
    region = (o[None, :] <= 0) & (o[None, :] > edge + 6)
    if region.sum() < 50:
        return False
    colour = np.median(band[region], axis=0)
    return float(np.abs(colour - lid_colour).max()) < 0.02


def _robust_poly(x, y, basis, iters=6, scale=1.0):
    """IRLS (Huber) least squares on a given basis matrix."""
    w = np.ones_like(y)
    coef = np.zeros(basis.shape[1])
    for _ in range(iters):
        bw = basis * w[:, None]
        coef, *_ = np.linalg.lstsq(bw, y * w, rcond=None)
        r = np.abs(y - basis @ coef)
        w = np.where(r <= scale, 1.0, scale / np.maximum(r, 1e-9))
    return coef


# Standard jacket sizes (width, height) in mm, used to place edges that
# cannot be seen.  Slim-case inserts are wider than tall and have notches cut
# out along their spine side, which is why that edge is often hard to find.
STANDARD_FORMATS_MM = {
    "CD booklet": (120.0, 120.0),
    "slim-case insert": (135.5, 120.0),
}


def _snap_to_standard(corners, conf, weak, dpi, notes, img_blur=None, fixed=(False,) * 4,
                      lid=None):
    """Use standard jacket sizes to place edges that could not be seen.

    From each reliable side, the opposite side is predicted at each standard
    size (at the scan's DPI).  For a side that was not reliably found:
      * an edge found near a prediction (+-2 %) is used - this catches a
        rough outline far outside the cover (shaded lid) or inside it (white
        margin); the prediction with the clearest edge wins;
      * without such an edge, the side is snapped to the nearest standard
        size only when within 4 % (12 % for a square booklet whose other
        dimension is reliably 120 mm).
    When both sides look reliable but the length is 2-6 % off the nearest
    size, one of them may be a lid feature: if exactly one side has a clear
    edge at its predicted position, it moves there.
    """
    if not dpi or dpi < 100 or img_blur is None:
        return corners
    px = dpi / 25.4
    c = corners.copy()
    w, h = output_size(c)
    center = c.mean(axis=0)

    def reliable(i):
        return conf[i] >= 0.5 and not weak[i]

    def edge_near(side, dev):
        """Offset of a real edge near side+dev (along its normal), and confidence."""
        _, n, _ = _side_frame(c[side], c[(side + 1) % 4], center)
        pa = c[side] + n * dev
        pb = c[(side + 1) % 4] + n * dev
        t2, n2, l2 = _side_frame(pa, pb, center + n * dev)
        r = int(max(40, 0.02 * 120 * px))
        s2, o2, band2 = _sample_band(img_blur, pa, t2, n2, l2, r)
        fit = _fit_side(s2, _pick_edges(band2, o2, sens=2.0))
        if fit is None:
            return None
        return float(np.polyval(fit[0], l2 / 2)), fit[1]

    def move(side, delta, text):
        nonlocal c
        _, n, _ = _side_frame(c[side], c[(side + 1) % 4], center)
        c[side] = c[side] + n * delta
        c[(side + 1) % 4] = c[(side + 1) % 4] + n * delta
        notes.append(f"{SIDE_NAMES[side]} edge {delta:+.0f} px: {text}.")

    sizes = {"w": sorted({f[0] for f in STANDARD_FORMATS_MM.values()}),
             "h": sorted({f[1] for f in STANDARD_FORMATS_MM.values()})}
    names = {}
    for nm, (fw, fh) in STANDARD_FORMATS_MM.items():
        names.setdefault(("w", fw), nm)
        names.setdefault(("h", fh), nm)
    square = {"w": reliable(0) and reliable(2) and abs(h - 120 * px) <= 0.015 * 120 * px,
              "h": reliable(1) and reliable(3) and abs(w - 120 * px) <= 0.015 * 120 * px}

    for dim, length, pair in (("w", w, (1, 3)), ("h", h, (0, 2))):
        if any(fixed[k] for k in pair):
            continue                                   # a side is the scan's own edge
        a_ok, b_ok = reliable(pair[0]), reliable(pair[1])
        if not a_ok and not b_ok:
            continue
        nearest = min(sizes[dim], key=lambda mm: abs(mm * px - length))
        if a_ok and b_ok:
            dev = nearest * px - length
            if 0.02 * nearest * px <= abs(dev) <= 0.06 * nearest * px:
                hits = {}
                for side in pair:
                    r_ = edge_near(side, dev)
                    if r_ is not None and r_[1] >= 0.5:
                        hits[side] = dev + r_[0]
                if len(hits) == 1:
                    side, delta = next(iter(hits.items()))
                    move(side, delta, f"moved to the edge found where the "
                                      f"{names[(dim, nearest)]} size puts it")
            continue
        side = pair[1] if a_ok else pair[0]
        best = None                                    # (confidence, delta, mm)
        for mm in sizes[dim]:
            dev = mm * px - length
            if abs(dev) > 0.40 * mm * px:
                continue
            r_ = edge_near(side, dev)
            if r_ is not None and r_[1] >= 0.4 and (best is None or r_[1] > best[0]):
                best = (r_[1], dev + r_[0], mm)
        if best is not None:
            if abs(best[1]) >= 3:
                move(side, best[1], f"placed on the edge found where the "
                                    f"{names[(dim, best[2])]} size ({best[2]:g} mm) puts it")
            continue
        dev = nearest * px - length
        tol = 0.04
        booklet_dev = 120.0 * px - length
        if square[dim] and lid is not None and booklet_dev < 0                 and abs(booklet_dev) <= 0.12 * 120.0 * px:
            # Is everything between this side and the square-booklet position
            # plain lid?  Then the cover cannot reach further: it is a booklet
            # (a slim-case insert would show printed cover there).
            _, n, slen = _side_frame(c[side], c[(side + 1) % 4], center)
            t_, n_, l_ = _side_frame(c[side], c[(side + 1) % 4], center)
            s_, o_, band_ = _sample_band(img_blur, c[side], t_, n_, l_, 4,
                                         inner=int(abs(booklet_dev)) + 4)
            coef_ = np.array([0.0, 0.0, booklet_dev + 0.0])
            if _cut_looks_like_lid(band_, o_, s_, coef_, lid):
                nearest, dev, tol = 120.0, booklet_dev, 0.12
        if 3 <= abs(dev) <= tol * nearest * px:
            move(side, dev, f"placed from the standard {names[(dim, nearest)]} size "
                            f"({nearest:g} mm); the edge itself is not clearly visible")
    return c


RIM_MAX = 45      # px: thin rims outside the print are always cropped off
UNDER_MAX = 85    # px: wider bands only when clearly a separate object


def _under_element(img_blur, a, t, n, length, lid_colour):
    """Width of a band to crop off between this side's edge and the print.

    The cleanest crop follows the edge of the printed area.  Outside it,
    along a side, there is often a thin smooth band: the white core of the
    cut card, a dark lid shadow, or the edge of a booklet or tray card lying
    under the jacket.  If a straight, consistent edge lies within RIM_MAX px
    inside, with a smooth (unprinted) band outside it, the crop moves to it.
    Wider bands (up to UNDER_MAX) are only cropped when they are clearly a
    separate object under the jacket: darker than white paper and not
    parallel to it (a printed border has an exactly constant width; a white
    paper margin is kept).
    Returns the inward shift (px), or 0.
    """
    s, o, band = _sample_band(img_blur, a, t, n, length, 6, inner=UNDER_MAX + 5)
    keep = o <= -4
    if keep.sum() < 20:
        return 0.0
    e = _pick_edges(band[:, keep], o[keep])
    fit = _fit_side(s, e)
    if fit is None or fit[1] < 0.6:
        return 0.0
    inner = np.polyval(fit[0], s)                         # offsets, negative
    width = -inner
    w = float(np.median(width))
    if not (4 <= w <= UNDER_MAX):
        return 0.0
    region = (o[None, :] < -1) & (o[None, :] > inner[:, None] + 2)
    if region.sum() < 50:
        return 0.0
    tex = np.abs(band - cv2.GaussianBlur(band, (1, 5), 1.0)).max(axis=2)
    if float(tex[region].mean()) > 0.0075:                 # printed: part of the cover
        return 0.0
    vals = band[region]
    lum = float(np.median(vals.mean(axis=1)))
    chroma = float(np.median(vals.max(axis=1) - vals.min(axis=1)))
    white = lum > float(lid_colour.mean()) - 0.06
    spread = float(np.percentile(width, 90) - np.percentile(width, 10))
    separate = spread >= 4.0             # not parallel: a different object
    if w <= RIM_MAX:
        if white:
            return w if w <= 15 else 0.0                   # cut card core / paper margin
        if w <= 20 or chroma < 0.06 or separate:           # shadow line / something underneath
            return w
        return 0.0                                         # coloured, parallel: printed border
    if white or not separate:                              # paper margin / printed border
        return 0.0
    return w


def _square_up(corners: np.ndarray, conf) -> np.ndarray:
    """Turn the four fitted sides into a true rotated rectangle.

    Covers are rectangles scanned flat, so the outline must not be warped
    by perspective.  One common angle is taken from the side directions
    (weighted by confidence and length, folded to -45..45 degrees); every
    side keeps its own position (through its midpoint).
    """
    c = np.asarray(corners, np.float64)
    angs, wts = [], []
    for i in range(4):
        d = c[(i + 1) % 4] - c[i]
        ang = np.degrees(np.arctan2(d[1], d[0])) - 90.0 * i     # side i runs at 90*i
        ang = (ang + 45.0) % 90.0 - 45.0
        angs.append(ang)
        wts.append(max(conf[i], 0.02) * np.hypot(*d))
    angs = np.array(angs)
    ref = angs[int(np.argmax(wts))]
    angs = ref + ((angs - ref + 45.0) % 90.0 - 45.0)            # unwrap near the best
    theta = np.radians(float(np.average(angs, weights=wts)))
    u = np.array([np.cos(theta), np.sin(theta)])                # along the top
    v = np.array([-np.sin(theta), np.cos(theta)])               # down the left side
    mids = [(c[i] + c[(i + 1) % 4]) / 2 for i in range(4)]
    lines = [(mids[0], u), (mids[1], v), (mids[2], u), (mids[3], v)]
    out = np.zeros((4, 2))
    for i in range(4):
        p = _intersect(*lines[i - 1], *lines[i])
        out[i] = c[i] if p is None else p
    return out


def detect_cover(img: np.ndarray, fit_bow: bool = False, dpi: float | None = None) -> CoverDetection:
    """Locate the cover in a scan. `img` is float32 BGR 0..1."""
    h, w = img.shape[:2]
    det = CoverDetection(corners=full_corners(img.shape),
                         bow=np.zeros((4, BOW_SAMPLES)))
    coarse = _coarse_rect(img, det)
    if coarse is None:
        det.has_background = False
        return det

    img_blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    if det.disc is not None:
        cx, cy, r, conf = _refine_disc(img_blur, *det.disc)
        # a second round from the better centre: rays then cross the rim squarely
        cx, cy, r, conf = _refine_disc(img_blur, cx, cy, r)
        if dpi:
            # discs are 120 mm (CD/DVD) or 80 mm (mini CD) across; a round
            # picture printed on a white cover is not a disc
            mm = 2 * r / dpi * 25.4
            if not any(abs(mm - d) <= 0.04 * d for d in (120.0, 80.0)):
                conf = 0.0
        if conf >= 0.5:
            det.disc = [cx, cy, r]
            det.corners = np.array([[cx - r, cy - r], [cx + r, cy - r],
                                    [cx + r, cy + r], [cx - r, cy + r]])
            det.side_confidence = [conf] * 4
            det.confidence = conf
            det.trim = [0, 0, 0, 0]
            det.notes.append(f"Disc found: {2 * r:.0f} px across; everything outside it "
                             "becomes white.")
            return det
        det.disc = None
    side_len = [np.hypot(*(coarse[(i + 1) % 4] - coarse[i])) for i in range(4)]
    radius = int(np.clip(0.02 * min(side_len), 16, 90))

    # Two passes: the rough rectangle can be off by the width of the lid
    # shadow, so the second pass re-centres the search bands on the first fit.
    corners = coarse.copy()
    found = [False] * 4
    conf = [0.0] * 4
    samples = [np.zeros((0, 2))] * 4
    lid = np.asarray(det.bg_color, np.float32)
    moved = [False] * 4
    flat_jump = [False] * 4
    faint_used = [False] * 4
    at_border = [False] * 4
    for _pass in range(3):
        center = corners.mean(axis=0)
        starts, ends = [], []       # tangent lines of each side at its two ends
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            t, n, length = _side_frame(a, b, center)
            kept = _off_border_line(a, b, n, w, h, det.off_sides)
            if kept is not None:
                # the cover runs past this edge of the scan: keep the edge
                starts.append(kept)
                ends.append(kept)
                conf[i], found[i], at_border[i] = 1.0, True, True
                continue
            s, o, band = _sample_band(img_blur, a, t, n, length, radius)
            e = _pick_edges(band, o)
            fit = _fit_side(s, e)
            suspicious = fit is None or fit[1] < 0.5 or _on_image_border(a, b, w, h)
            # Run in both passes: once neighbouring sides have been pulled in,
            # a side's band no longer spans empty lid and its edge scores higher.
            if suspicious and not moved[i]:
                # The rough outline may be far outside the paper (lid shadow
                # or a lid lifted by a booklet spine counted as cover).  Look
                # much further inward, but accept only a strong straight edge
                # with nothing but plain lid between it and the rough line.
                inner = int(0.45 * min(side_len))
                s2, o2, band2 = _sample_band(img_blur, a, t, n, length, radius, inner=inner)
                e2 = _pick_edges(band2, o2)
                fit2 = _fit_side(s2, e2)
                if fit2 is not None and fit2[1] >= 0.4 \
                        and (fit is None or fit2[1] >= fit[1] + 0.1):
                    move = float(np.polyval(fit2[0], length / 2))
                    if move < -0.5 * radius \
                            and _cut_looks_like_lid(band2, o2, s2, fit2[0], lid):
                        s, o, band, e, fit = s2, o2, band2, e2, fit2
                        moved[i] = True
                        # jumped across plain white: might be white paper margin
                        flat_jump[i] = _cut_is_bright_lid(band2, o2, s2, fit2[0], lid)
                        margin = _margin_edge(band2, o2, s2, fit2[0], length)
                        if margin is not None:
                            coef_m, conf_m, e_m = margin
                            fit = (coef_m, max(conf_m, 0.5), np.isfinite(e_m) &
                                   (np.abs(e_m - np.polyval(coef_m, s2)) < 1.5))
                            e = e_m
                            flat_jump[i] = False
                            move = float(np.polyval(coef_m, length / 2))
                        det.notes.append(
                            f"{SIDE_NAMES[i]} edge found {-move:.0f} px inside the rough "
                            "outline (uneven scanner lid).")
            if fit is None:
                # last resort: a faint but straight edge (white on white)
                inner = int(0.45 * min(side_len))
                s3, o3, band3 = _sample_band(img_blur, a, t, n, length, radius, inner=inner)
                faint = _faint_edge(band3, o3, s3)
                if faint is None or faint[1] < 0.7:
                    # paper tint against the neutral lid (soft, shaded edges)
                    faint = _tint_edge(band3, o3, s3)
                    if faint is not None and not _cut_looks_like_lid(band3, o3, s3, faint[0], lid):
                        faint = None
                    if faint is not None:
                        # an untinted (white) margin may lie outside the tint
                        # edge: prefer a weaker straight paper edge before it
                        stop = float(np.polyval(faint[0], length / 2)) + 10
                        keep = o3 >= stop
                        if keep.sum() > 20:
                            e4 = _pick_edges(band3[:, keep], o3[keep], sens=2.0)
                            fit4 = _fit_side(s3, e4)
                            if fit4 is not None and fit4[1] >= 0.35:
                                faint = (fit4[0], max(fit4[1], 0.7))
                if faint is not None and faint[1] >= 0.7:
                    coef = faint[0]
                    for sv, store in ((0.0, starts), (length, ends)):
                        d = t + n * np.polyval(np.polyder(coef), sv)
                        store.append((a + t * sv + n * np.polyval(coef, sv), d / np.hypot(*d)))
                    conf[i] = 0.35                     # usable, but not trusted
                    found[i] = True
                    faint_used[i] = True
                    continue
                starts.append((a, t))
                ends.append((a, t))
                if _pass == 2 and not found[i]:
                    det.notes.append(f"{SIDE_NAMES[i]} edge not visible against the "
                                     "background; using a rough estimate.")
                continue
            coef, frac, inl = fit
            for sv, store in ((0.0, starts), (length, ends)):
                d = t + n * np.polyval(np.polyder(coef), sv)
                store.append((a + t * sv + n * np.polyval(coef, sv), d / np.hypot(*d)))
            samples[i] = a + np.outer(s[inl], t) + np.outer(e[inl], n)
            conf[i] = frac
            found[i] = True
        # A cover is a rectangle: a side without a trustworthy edge of its
        # own runs parallel to its opposite side when that one is reliable.
        for i in range(4):
            if conf[i] >= 0.5:
                continue
            j = (i + 2) % 4
            pi = (starts[i][0] + ends[i][0]) / 2
            cur = (starts[i][1] + ends[i][1]) / 2
            if conf[j] >= 0.5:
                d = -(starts[j][1] + ends[j][1]) / 2           # opposite sides run reversed
            else:
                nb = [k for k in ((i - 1) % 4, (i + 1) % 4) if conf[k] >= 0.5]
                if not nb:
                    continue
                k = max(nb, key=lambda q: conf[q])
                dk = (starts[k][1] + ends[k][1]) / 2
                d = np.array([-dk[1], dk[0]])                  # perpendicular to a neighbour
                if np.dot(d, cur) < 0:
                    d = -d
            d = d / max(np.hypot(*d), 1e-9)
            starts[i] = (pi, d)
            ends[i] = (pi, d)
        new = np.zeros((4, 2))
        for i in range(4):
            p = _intersect(*ends[i - 1], *starts[i])
            new[i] = corners[i] if p is None else p
        corners = new
    corners = _square_up(corners, conf)
    # items lying under the jacket and sticking out past its edge
    center = corners.mean(axis=0)
    moved_under = []
    for i in range(4):
        if at_border[i] or not found[i]:
            continue
        a, b = corners[i], corners[(i + 1) % 4]
        t, n, length = _side_frame(a, b, center)
        shift = _under_element(img_blur, a, t, n, length, lid)
        if shift > 0:
            corners[i] = corners[i] - n * shift
            corners[(i + 1) % 4] = corners[(i + 1) % 4] - n * shift
            moved_under.append(f"{SIDE_NAMES[i]} {shift:.0f} px")
    if moved_under:
        corners = _square_up(corners, conf)
        det.notes.append("Cropped to the edge of the print, past a thin rim or something "
                         "under the jacket: " + ", ".join(moved_under) + ".")
    weak = [(not found[i]) or conf[i] < 0.3 or flat_jump[i] or faint_used[i] for i in range(4)]
    corners = _snap_to_standard(corners, conf, weak, dpi, det.notes, img_blur, at_border, lid)
    center = corners.mean(axis=0)

    # ---- bow (curved edges) and trim suggestions -------------------------
    bow = np.zeros((4, BOW_SAMPLES))
    resid_all = []
    tk = np.linspace(0, 1, BOW_SAMPLES)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        t, n, length = _side_frame(a, b, center)
        pts = samples[i]
        if len(pts) < 30:
            resid_all.append(np.zeros(0))
            continue
        u = (pts - a) @ t / length
        r = (pts - a) @ n
        near = np.abs(r) < 0.02 * length
        u, r = u[near], r[near]
        covered = len(u) and (np.histogram(u, bins=8, range=(0, 1))[0] > 3).sum() >= 7
        if fit_bow and conf[i] >= 0.5 and covered:
            basis = np.column_stack([u * (1 - u), u * (1 - u) * u])
            coef = _robust_poly(u, r, basis)
            curve = lambda x, c=coef: x * (1 - x) * (c[0] + c[1] * x)
            bow[i] = curve(tk)
            r = r - curve(u)
        resid_all.append(r)
    if np.abs(bow).max() < 1.0:
        bow[:] = 0.0                      # not worth resampling for
    else:
        for i in range(4):
            if np.abs(bow[i]).max() < 0.6:
                bow[i] = 0.0

    # Trim: remove the anti-aliased edge, typical inward nicks and a worn or
    # discoloured band along the edge. Order l,t,r,b.
    trims = []
    worn = []
    max_wear = int(min(18, 0.008 * min(side_len)))
    for i in range(4):
        r = resid_all[i]
        inward = np.percentile(np.maximum(0.0, -r), 90) if len(r) else 0.0
        trim = int(np.ceil(2.0 + min(inward, 0.01 * min(side_len))))
        if at_border[i]:
            trims.append(0)               # the scan's own edge: nothing to trim
            continue
        if found[i] and max_wear > 2:
            a, b = corners[i], corners[(i + 1) % 4]
            t, n, length = _side_frame(a, b, center)
            width, kind = edge_wear(img_blur, a, t, n, length, max_wear)
            if width + 1 > trim:
                trim = width + 1
                worn.append(f"{SIDE_NAMES[i]} {width} px ({kind})")
        trims.append(trim)
    if worn:
        det.notes.append("Worn / discoloured paper edge trimmed: " + ", ".join(worn) + ".")
    det.trim = [trims[3], trims[0], trims[1], trims[2]]

    top = corners[1] - corners[0]
    det.corners = corners
    det.bow = bow
    det.max_bow = float(np.abs(bow).max())
    det.angle_deg = float(np.degrees(np.arctan2(top[1], top[0])))
    det.side_confidence = [float(c) for c in conf]
    det.confidence = float(min(conf))
    return det


# --------------------------------------------------------------------------
# warp
# --------------------------------------------------------------------------

def _curve(a, b, n, bow_row, count):
    """Points along side a->b displaced by the bow profile."""
    u = np.linspace(0.0, 1.0, count)
    pts = a[None, :] + np.outer(u, b - a)
    if bow_row is not None and np.any(bow_row):
        from scipy.interpolate import CubicSpline
        off = CubicSpline(np.linspace(0, 1, len(bow_row)), bow_row)(u)
        pts = pts + np.outer(off, n)
    return pts


def output_size(corners: np.ndarray) -> tuple[int, int]:
    c = np.asarray(corners, np.float64)
    wt = np.hypot(*(c[1] - c[0])); wb = np.hypot(*(c[2] - c[3]))
    hl = np.hypot(*(c[3] - c[0])); hr = np.hypot(*(c[2] - c[1]))
    return max(8, int(round((wt + wb) / 2))), max(8, int(round((hl + hr) / 2)))


def build_maps(corners, bow=None, size=None):
    """Coons-patch remap from the (possibly curved) quad to a rectangle."""
    c = np.asarray(corners, np.float64)
    W, H = size or output_size(c)
    center = c.mean(axis=0)
    bw = np.zeros((4, BOW_SAMPLES)) if bow is None else np.asarray(bow, np.float64)
    nrm = [_side_frame(c[i], c[(i + 1) % 4], center)[1] for i in range(4)]
    top = _curve(c[0], c[1], nrm[0], bw[0], W)             # u: 0..1
    right = _curve(c[1], c[2], nrm[1], bw[1], H)           # v: 0..1
    bottom = _curve(c[3], c[2], nrm[2], bw[2][::-1], W)    # stored BR->BL
    left = _curve(c[0], c[3], nrm[3], bw[3][::-1], H)      # stored BL->TL
    u = np.linspace(0, 1, W)[None, :, None]
    v = np.linspace(0, 1, H)[:, None, None]
    s = ((1 - v) * top[None] + v * bottom[None]
         + (1 - u) * left[:, None] + u * right[:, None]
         - ((1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1]
            + u * v * c[2] + (1 - u) * v * c[3]))
    return s[..., 0].astype(np.float32), s[..., 1].astype(np.float32)


def outline_points(corners, bow=None, per_side=40) -> np.ndarray:
    """Closed polyline of the detected outline in source coordinates."""
    c = np.asarray(corners, np.float64)
    center = c.mean(axis=0)
    bw = np.zeros((4, BOW_SAMPLES)) if bow is None else np.asarray(bow, np.float64)
    out = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        out.append(_curve(a, b, _side_frame(a, b, center)[1], bw[i], per_side)[:-1])
    return np.concatenate(out)


def text_angle(img: np.ndarray, disc: bool = False):
    """Skew of straight text / line art, in degrees (-45..45), or None.

    Projection-profile method: the image's edge map is rotated through
    candidate angles and the angle at which its row sums are most uneven
    (text lines and gaps lined up) wins.  Returns (angle, strength) where
    strength is the winning score relative to the median over all angles.
    Only the angle modulo 90 degrees can be found this way.
    """
    small = cv2.resize(img, None, fx=600.0 / max(img.shape[:2]), fy=600.0 / max(img.shape[:2]),
                       interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F))
    h, w = edges.shape
    if disc:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / (min(w, h) / 2)
        edges *= ((r > 0.30) & (r < 0.92)).astype(np.float32)   # skip hub and rim
    edges = np.minimum(edges, np.percentile(edges, 99.5) + 1e-6)

    def score(a):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(edges, m, (w, h), flags=cv2.INTER_LINEAR)
        prof = rot.sum(axis=1)
        return float(np.var(np.diff(prof)))

    coarse = np.arange(-45.0, 45.0, 1.0)
    sc = np.array([score(a) for a in coarse])
    best = float(coarse[int(np.argmax(sc))])
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    sf = np.array([score(a) for a in fine])
    angle = float(fine[int(np.argmax(sf))])
    strength = float(sf.max() / (np.median(sc) + 1e-9))
    return angle, strength


def apply_geometry(img: np.ndarray, corners, bow=None, trim=(0, 0, 0, 0),
                   rotate90: int = 0, disc: bool = False, angle: float = 0.0) -> np.ndarray:
    """Rectify, trim and rotate. Returns a new float32 image.

    With `disc`, the corners enclose a round disc: everything outside the
    inscribed circle is set to white (with an anti-aliased rim).  `angle`
    rotates the result by that many degrees (counter-clockwise) about its
    centre; a disc keeps its full size, a rectangular cover is enlarged as
    little as needed so no blank corners show.
    """
    if corners is None:
        out = img
    else:
        mx, my = build_maps(corners, bow)
        out = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        out = np.clip(out, 0.0, 1.0)
    if abs(angle) > 0.01:
        h, w = out.shape[:2]
        if disc:
            zoom = 1.0
        else:                                        # crop rotated corners away
            rad = np.radians(abs(angle))
            zoom = float(np.cos(rad) + max(w, h) / min(w, h) * np.sin(rad))
        m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, zoom)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0))
        out = np.clip(out, 0.0, 1.0)
    if corners is not None:
        if disc:
            h, w = out.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
            inside = np.clip((min(w, h) - 1) / 2 - dist + 0.5, 0.0, 1.0)[:, :, None]
            out = out * inside + (1.0 - inside)
    l, t, r, b = [max(0, int(x)) for x in trim]
    h, w = out.shape[:2]
    if l + r < w - 8 and t + b < h - 8 and (l or t or r or b):
        out = out[t:h - b, l:w - r]
    k = int(rotate90) % 4
    if k:
        out = np.rot90(out, -k)
    return np.ascontiguousarray(out)
