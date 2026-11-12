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

def _background_model(small: np.ndarray, m: int):
    """Per-pixel estimate of the scanner background (lid) colour.

    The lid is not always evenly lit: where it does not lie flat, e.g. lifted
    by a booklet's spine, a whole side of the scan can be several levels
    darker than the rest.  Each image border gives a colour profile along
    that side (median filtered, so dust and small objects drop out); inside
    the image the four profiles are blended by inverse squared distance, so
    every pixel is compared with the lid colour of its nearest side.

    Returns (global background colour, model image).
    """
    sh, sw = small.shape[:2]
    strips = [small[:m], small[-m:], small[:, :m], small[:, -m:]]
    glob = np.median(np.concatenate([s.reshape(-1, 3) for s in strips]), axis=0)

    def profile(values: np.ndarray, length: int) -> np.ndarray:
        k = max(3, int(0.08 * length) | 1)
        prof = median_filter(values, size=(k, 1), mode="nearest")
        # Far from the overall lid colour = the cover itself reaches the
        # image edge here; fall back to the overall colour.
        far = np.abs(prof - glob).max(axis=1) > 0.10
        prof[far] = glob
        return prof.astype(np.float32)

    top = profile(np.median(small[:m], axis=0), sw)          # (sw, 3)
    bottom = profile(np.median(small[-m:], axis=0), sw)
    left = profile(np.median(small[:, :m], axis=1), sh)      # (sh, 3)
    right = profile(np.median(small[:, -m:], axis=1), sh)
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
    det.bg_color = tuple(float(x) for x in glob)
    dist = np.abs(small - model).max(axis=2)
    bdist = np.concatenate([dist[:m].ravel(), dist[-m:].ravel(),
                            dist[:, :m].ravel(), dist[:, -m:].ravel()])
    if np.percentile(bdist, 85) > 0.10:
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
    return _order_corners(box / scale)


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


def _pick_edges(band: np.ndarray, o: np.ndarray):
    """Locate the paper edge in every row of a band. Returns offsets (nan = none)."""
    band = cv2.blur(band, (1, 15))                       # average along the edge
    lum = luminance(band)
    dl = np.gradient(lum, axis=1)
    dc = np.gradient(band, axis=1)
    mag = np.sqrt((dc ** 2).sum(axis=2))
    g = np.where(dl >= 0, mag, -mag)                     # signed colour gradient
    outer = g[:, : max(4, g.shape[1] // 4)]
    noise = 1.4826 * np.median(np.abs(outer - np.median(outer)))
    thr = max(6.0 * noise, 0.010)
    out = np.full(g.shape[0], np.nan)
    ncol = g.shape[1]
    for r in range(g.shape[0]):
        row = g[r]
        idx = np.flatnonzero(np.abs(row) > thr)
        if idx.size == 0:
            continue
        i0 = int(idx[0])
        if i0 < 2:                                        # edge beyond the band
            continue
        sgn = np.sign(row[i0])
        j = i0
        while j + 1 < ncol and row[j + 1] * sgn > thr * 0.3:
            j += 1
        p = i0 + int(np.argmax(np.abs(row[i0:j + 1])))
        if sgn < 0:
            # A darkening run followed directly by a brightening run is the
            # lid shadow next to lighter paper: the paper edge is the rise.
            k = j + 1
            lim = min(ncol - 1, j + 5)
            while k <= lim and row[k] <= thr:
                k += 1
            if k <= lim:
                k2 = k
                while k2 + 1 < ncol and row[k2 + 1] > thr * 0.3:
                    k2 += 1
                p = k + int(np.argmax(row[k:k2 + 1]))
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
    return coef, float(inl.sum()) / len(e), mask


LID_TEXTURE = 0.004     # high-frequency level: bare lid < ~0.0025, print > ~0.005


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


def detect_cover(img: np.ndarray, fit_bow: bool = True) -> CoverDetection:
    """Locate the cover in a scan. `img` is float32 BGR 0..1."""
    h, w = img.shape[:2]
    det = CoverDetection(corners=full_corners(img.shape),
                         bow=np.zeros((4, BOW_SAMPLES)))
    coarse = _coarse_rect(img, det)
    if coarse is None:
        det.has_background = False
        return det

    img_blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    side_len = [np.hypot(*(coarse[(i + 1) % 4] - coarse[i])) for i in range(4)]
    radius = int(np.clip(0.02 * min(side_len), 16, 90))

    # Two passes: the rough rectangle can be off by the width of the lid
    # shadow, so the second pass re-centres the search bands on the first fit.
    corners = coarse.copy()
    found = [False] * 4
    conf = [0.0] * 4
    samples = [np.zeros((0, 2))] * 4
    lid = np.asarray(det.bg_color, np.float32)
    for _pass in range(2):
        center = corners.mean(axis=0)
        starts, ends = [], []       # tangent lines of each side at its two ends
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            t, n, length = _side_frame(a, b, center)
            s, o, band = _sample_band(img_blur, a, t, n, length, radius)
            e = _pick_edges(band, o)
            fit = _fit_side(s, e)
            suspicious = fit is None or fit[1] < 0.3 or _on_image_border(a, b, w, h)
            if _pass == 0 and suspicious:
                # The rough outline may be far outside the paper (lid
                # differences counted as cover).  Look much further inward,
                # but accept only a strong straight edge with nothing but
                # plain lid between it and the rough line.
                inner = int(min(0.15 * length, 600))
                s2, o2, band2 = _sample_band(img_blur, a, t, n, length, radius, inner=inner)
                e2 = _pick_edges(band2, o2)
                fit2 = _fit_side(s2, e2)
                if fit2 is not None and fit2[1] >= 0.6 \
                        and (fit is None or fit2[1] >= fit[1] + 0.15):
                    move = float(np.polyval(fit2[0], length / 2))
                    if move < -0.5 * radius \
                            and _cut_looks_like_lid(band2, o2, s2, fit2[0], lid):
                        s, o, band, e, fit = s2, o2, band2, e2, fit2
                        det.notes.append(
                            f"{SIDE_NAMES[i]} edge found {-move:.0f} px inside the rough "
                            "outline (uneven scanner lid).")
            if fit is None:
                starts.append((a, t))
                ends.append((a, t))
                if _pass == 1 and not found[i]:
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
        new = np.zeros((4, 2))
        for i in range(4):
            p = _intersect(*ends[i - 1], *starts[i])
            new[i] = corners[i] if p is None else p
        corners = new
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

    # Trim: remove the anti-aliased edge and typical inward nicks. Order l,t,r,b.
    trims = []
    for i in range(4):
        r = resid_all[i]
        inward = np.percentile(np.maximum(0.0, -r), 90) if len(r) else 0.0
        trims.append(int(np.ceil(2.0 + min(inward, 0.01 * min(side_len)))))
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


def apply_geometry(img: np.ndarray, corners, bow=None, trim=(0, 0, 0, 0),
                   rotate90: int = 0) -> np.ndarray:
    """Rectify, trim and rotate. Returns a new float32 image."""
    if corners is None:
        out = img
    else:
        mx, my = build_maps(corners, bow)
        out = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        out = np.clip(out, 0.0, 1.0)
    l, t, r, b = [max(0, int(x)) for x in trim]
    h, w = out.shape[:2]
    if l + r < w - 8 and t + b < h - 8 and (l or t or r or b):
        out = out[t:h - b, l:w - r]
    k = int(rotate90) % 4
    if k:
        out = np.rot90(out, -k)
    return np.ascontiguousarray(out)
