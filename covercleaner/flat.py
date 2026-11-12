"""Flat colour cleanup.

Artwork often contains areas that were a single solid colour in the
original file (backgrounds, stripes, panels, lettering).  After printing and
scanning they carry paper grain, ink mottle and slight shading.  This stage
finds such areas and paints each one with one even colour:

1. Pixels whose neighbourhood varies less than the tolerance form candidate
   regions (connected components).
2. A region is accepted when its colour is consistent across its whole
   extent (so smooth gradients and photos are rejected) and, optionally,
   when it is bounded by other flat areas across thin, crisp edges (vector
   style artwork) rather than fading softly into a picture.
3. Regions of (almost) the same colour anywhere on the cover are unified to
   one palette colour.
4. Regions are grown up to their true edges and blended in with a narrow
   feather, so the anti-aliased outlines of shapes and letters are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FlatResult:
    labels: np.ndarray                      # int32, 0 = not flat
    colours: np.ndarray                     # (n_labels + 1, 3) float32 BGR, final
    observed: np.ndarray = None             # same, as measured in the scan
    coverage: float = 0.0                   # fraction of the image flattened
    n_regions: int = 0
    n_colours: int = 0
    palette: list = field(default_factory=list)   # [(b, g, r, area_fraction)]


def _region_stats(labels, n, img):
    """Per-label pixel count, mean and std (max over channels)."""
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    safe = np.maximum(cnt, 1)
    mean = np.zeros((n, 3))
    var = np.zeros((n, 3))
    for c in range(3):
        v = img[:, :, c].ravel().astype(np.float64)
        mean[:, c] = np.bincount(flat, v, minlength=n) / safe
        var[:, c] = np.bincount(flat, v * v, minlength=n) / safe - mean[:, c] ** 2
    return cnt, mean, np.sqrt(np.maximum(var, 0)).max(axis=1)


def find_flat_regions(img: np.ndarray, tolerance: float = 0.06, min_area_pct: float = 0.05,
                      unify: bool = True, crisp_only: bool = True,
                      pure_bw: bool = True) -> FlatResult:
    """Segment a float BGR image into flat-colour regions."""
    h, w = img.shape[:2]
    tol = float(tolerance)
    smooth = cv2.GaussianBlur(img, (0, 0), 2.0)
    k = np.ones((9, 9), np.uint8)
    rng = (cv2.dilate(smooth, k) - cv2.erode(smooth, k)).max(axis=2)   # local range
    cand = (rng < 0.6 * tol).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(cand, connectivity=4)
    empty = FlatResult(np.zeros((h, w), np.int32), np.zeros((1, 3), np.float32),
                       np.zeros((1, 3), np.float32))
    if n <= 1:
        return empty

    cnt, mean, std = _region_stats(labels, n, smooth)
    min_area = max(30.0, h * w * min_area_pct / 100.0)
    consistent = std < 0.45 * tol
    # Consistent colour over the whole region.  Real scans of a solid ink
    # drift and mottle with a std of up to ~0.02, so some slack is needed;
    # a gradient of total range R has std R / 3.5 and is rejected once it
    # spans clearly more than the tolerance.
    ok = (cnt >= min_area) & consistent
    ok[0] = False
    small = (cnt >= 30) & (cnt < min_area) & consistent      # may join a colour later
    small[0] = False
    frac = np.ones(n)

    if crisp_only and ok.any():
        # Does the region border other flat candidates across a thin edge
        # (artwork), or does it fade into non-flat picture content?
        big = cnt >= 30
        big[0] = False
        lab_big = np.where(big[labels], labels, 0).astype(np.float32)
        def lowest(lab, kernel):
            inv_ = np.where(lab > 0, n + 1 - lab, 0).astype(np.float32)
            d = cv2.dilate(inv_, kernel)
            return np.where(d > 0, n + 1 - d, 0)

        k7 = np.ones((7, 7), np.uint8)
        kw = np.ones((41, 41), np.uint8)
        near_hi = cv2.dilate(lab_big, k7)              # adjacent labels (largest
        near_lo = lowest(lab_big, k7)                  #  and smallest id)
        wide_hi = cv2.dilate(lab_big, kw)
        wide_lo = lowest(lab_big, kw)
        ring = (lab_big == 0) & (near_hi > 0)
        two = ring & (wide_hi != wide_lo)              # another region close by
        total = np.zeros(n)
        crisp = np.zeros(n)
        for near in (near_hi, near_lo):
            total += np.bincount(near[ring].astype(np.int64), minlength=n)[:n]
            crisp += np.bincount(near[two].astype(np.int64), minlength=n)[:n]
        # the image border counts as a crisp boundary
        edge = np.zeros((h, w), bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        on_edge = np.bincount(labels[edge], minlength=n).astype(np.float64)
        frac = (crisp + 3.0 * on_edge) / np.maximum(total + 3.0 * on_edge, 1.0)
        ok &= frac >= 0.20

    if not ok.any():
        return empty

    # Refine each colour using only pixels close to the first estimate, so
    # specks and stains inside a region do not shift its colour.
    close = np.abs(smooth - mean[labels].astype(np.float32)).max(axis=2) < tol
    lab_close = np.where(close & (ok | small)[labels], labels, 0)
    cnt2, mean2, _ = _region_stats(lab_close, n, img)
    colour = np.where(cnt2[:, None] > 0, mean2, mean)

    # Unify matching colours across the cover (largest regions lead).
    ids = np.flatnonzero(ok)
    ids = ids[np.argsort(-cnt[ids])]
    n_main = len(ids)
    # Small pieces (letter fragments, counters) follow: they may only join
    # a colour that a full-size region has already established.
    extra = np.flatnonzero(small & (frac >= 0.50))
    ids = np.concatenate([ids, extra])
    palette: list[list] = []          # [colour sum (3), weight]
    cluster = {}
    for pos, i in enumerate(ids):
        target = None
        if unify:
            for j, (csum, wgt) in enumerate(palette):
                if np.abs(csum / wgt - colour[i]).max() < tol:
                    target = j
                    break
        if target is None and pos >= n_main:
            continue
        if target is None:
            palette.append([colour[i] * cnt[i], cnt[i]])
            target = len(palette) - 1
        elif palette[target][1] < 20 * cnt[i]:
            palette[target][0] = palette[target][0] + colour[i] * cnt[i]
            palette[target][1] += cnt[i]
        cluster[int(i)] = target
    pal_colour = []
    for csum, wgt in palette:
        c = csum / wgt
        if pure_bw:
            lum = 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
            if lum < 0.10:
                c = np.zeros(3)             # solid ink black
            elif c.min() > 0.94:
                c = np.ones(3)              # bare paper white
        pal_colour.append(c)
    final = np.zeros((n, 3), np.float32)
    seen = np.zeros((n, 3), np.float32)
    for i, j in cluster.items():
        final[i] = pal_colour[j]
        seen[i] = palette[j][0] / palette[j][1]

    ok = np.zeros(n, bool)
    ok[list(cluster)] = True
    out_labels = np.where(ok[labels], labels, 0).astype(np.int32)
    area = float(cnt[ok].sum())
    pal = sorted(((*c, pw[1] / (h * w)) for c, pw in zip(pal_colour, palette)),
                 key=lambda t: -t[3])
    return FlatResult(out_labels, final, seen, area / (h * w), int(ok.sum()), len(palette),
                      [tuple(float(v) for v in p) for p in pal])


def _grow(labels, colours, img, tol, steps):
    """Extend regions outward over pixels that already match their colour."""
    lab = labels.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        grown = cv2.dilate(lab, k)
        new = (lab == 0) & (grown > 0)
        if not new.any():
            break
        ys, xs = np.nonzero(new)
        gl = grown[ys, xs].astype(np.int64)
        match = np.abs(img[ys, xs] - colours[gl]).max(axis=1) < tol
        if not match.any():
            break
        lab[ys[match], xs[match]] = gl[match]
    return lab.astype(np.int32)


def flatten(img: np.ndarray, tolerance: float = 0.06, min_area_pct: float = 0.05,
            strength: float = 1.0, unify: bool = True, crisp_only: bool = True,
            fill_marks_pct: float = 0.0, pure_bw: bool = True, clean_edges: bool = True,
            result: FlatResult | None = None, protect: np.ndarray | None = None):
    """Even out flat-colour regions. Returns (image, mask uint8)."""
    res = result or find_flat_regions(img, tolerance, min_area_pct, unify, crisp_only, pure_bw)
    h, w = img.shape[:2]
    if res.n_regions == 0 or strength <= 0:
        return img, np.zeros((h, w), np.uint8)
    smooth = cv2.GaussianBlur(img, (0, 0), 1.0)
    labels = _grow(res.labels, res.observed, smooth, 0.9 * tolerance, steps=24)
    # Enclosed holes inside a single region join it when they are
    #  - up to 4 px: noise outliers, smaller than any printed detail;
    #  - up to 0.5% of the short side and only faintly different from the
    #    region (ink voids, light stains) - real detail has more contrast;
    #  - up to the user's "absorb marks" size, whatever their colour.
    faint_px = max(4.0, 0.005 * min(h, w))
    user_px = min(h, w) * fill_marks_pct / 100.0
    holes = (labels == 0).astype(np.uint8)
    nh, hl, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    size = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    cand = size <= max(faint_px, user_px)
    cand[0] = False
    if cand.any():
        lf = labels.astype(np.float32)
        kk = np.ones((3, 3), np.uint8)

        def around(lab_f):
            hi_ = cv2.dilate(lab_f, kk)
            inv_ = np.where(lab_f > 0, 1e7 - lab_f, 0).astype(np.float32)
            return hi_, 1e7 - cv2.dilate(inv_, kk)

        hi, lo = around(lf)
        # contrast of each hole against the region around it
        sel = cand[hl]
        rim = sel & (hi > 0) & (hi == lo)
        diff = np.zeros(nh)
        if rim.any():
            dv = np.abs(smooth[rim] - res.observed[hi[rim].astype(np.int64)]).max(axis=1)
            ids = hl[rim]
            diff = np.bincount(ids, dv, minlength=nh) / np.maximum(np.bincount(ids, minlength=nh), 1)
            # mean over the whole hole, not just its rim
            allv = np.abs(smooth[sel] - res.observed[
                cv2.dilate(lf, np.ones((31, 31), np.uint8))[sel].astype(np.int64)]).max(axis=1)
            diff = np.bincount(hl[sel], allv, minlength=nh) / np.maximum(
                np.bincount(hl[sel], minlength=nh), 1)
        take = cand & ((size <= 4) | (size <= user_px) | (diff < 3.0 * tolerance))
        sel = take[hl]
        for _ in range(int(max(faint_px, user_px)) + 2):   # fill from the rim inwards
            edge = sel & (labels == 0) & (hi > 0) & (hi == lo)
            if not edge.any():
                break
            labels[edge] = hi[edge].astype(np.int32)
            hi, lo = around(labels.astype(np.float32))
    inside = labels > 0
    st = float(min(strength, 1.0))
    target = res.colours[labels]
    out = img.copy()
    out[inside] = img[inside] + st * (target[inside] - img[inside])
    mask = inside.astype(np.uint8) * 255
    if clean_edges:
        # Pixels on the seam between two flat regions are a blend of their
        # two colours (anti-aliasing) plus noise.  Re-create the blend from
        # the clean colours: this keeps the edge shape but removes the noise.
        lf = labels.astype(np.float32)
        kk = np.ones((11, 11), np.uint8)
        hi = cv2.dilate(lf, kk)
        inv = np.where(lf > 0, 1e7 - lf, 0).astype(np.float32)
        lo = 1e7 - cv2.dilate(inv, kk)
        seam = (~inside) & (hi > 0) & (lo < 1e7 - 0.5) & (hi != lo)
        ys, xs = np.nonzero(seam)
        if ys.size:
            ia = hi[ys, xs].astype(np.int64)
            ib = lo[ys, xs].astype(np.int64)
            # blend factor from the colours as scanned ...
            cb = res.observed[ib]
            d = res.observed[ia] - cb
            den = (d * d).sum(axis=1)
            px = img[ys, xs]

            def blend(v):
                a_ = np.where(den > 1e-6, ((v - cb) * d).sum(axis=1) / np.maximum(den, 1e-6), 0.5)
                return np.clip(a_, 0.0, 1.0)[:, None]

            alpha = blend(px)
            # ... only where the pixel really is such a blend (not a third colour)
            sm = smooth[ys, xs]
            good = np.abs(sm - (cb + blend(sm) * d)).max(axis=1) < 2.5 * tolerance
            # ... applied to the clean colours
            fa, fb = res.colours[ia], res.colours[ib]
            mix = fb + alpha * (fa - fb)
            out[ys[good], xs[good]] = px[good] + st * (mix[good] - px[good])
            mask[ys[good], xs[good]] = 255
    if protect is not None and protect.any():
        keep = protect > 0
        out[keep] = img[keep]
        mask[keep] = 0
    return out.astype(np.float32), mask
