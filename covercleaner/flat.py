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
4. Each section is grown out to its true edge over pixels of its colour
   (judged after a median that removes grain and specks), takes in small
   enclosed specks, and is painted with its one colour - fully up to the
   tolerance, fading out by twice it so stains and soft shading blend in;
   fine detail that clearly stands out (dots, hairlines) is kept.
5. Only the thin anti-aliased boundary between two sections is redrawn, as
   a mix of their two clean colours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEBUG = None      # set to {} to capture edge-rebuild internals

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

    # A flat area is bounded by an edge: just outside it the colour changes.
    # A pocket of a larger textured or grainy area (dark hair, shadows in a
    # photo) is surrounded by more of the same tone; flattening it would
    # leave a visible blotch, so such regions are dropped.
    cand_ids = np.flatnonzero(ok | small)
    if cand_ids.size:
        stats = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8), 4)[2] \
            if False else None
        ys, xs = np.nonzero(np.isin(labels, cand_ids))
        lab_of = labels[ys, xs]
        x0 = np.full(n, 1 << 30); y0 = np.full(n, 1 << 30); x1 = np.zeros(n, int); y1 = np.zeros(n, int)
        np.minimum.at(x0, lab_of, xs); np.minimum.at(y0, lab_of, ys)
        np.maximum.at(x1, lab_of, xs); np.maximum.at(y1, lab_of, ys)
        k_in = np.ones((13, 13), np.uint8)       # region stops ~4-5 px short of its edge,
        k_out = np.ones((25, 25), np.uint8)      # so look 7-12 px out, past the edge
        for i in cand_ids:
            pad = 13
            sy0, sx0 = max(0, y0[i] - pad), max(0, x0[i] - pad)
            sy1, sx1 = min(h, y1[i] + pad + 1), min(w, x1[i] + pad + 1)
            reg = (labels[sy0:sy1, sx0:sx1] == i).astype(np.uint8)
            ring = (cv2.dilate(reg, k_out) > 0) & (cv2.dilate(reg, k_in) == 0)
            others = (labels[sy0:sy1, sx0:sx1] > 0) & ~reg.astype(bool)
            ring &= ~others                                  # neighbours that are flat themselves
            if ring.sum() < 20:
                continue
            d = np.abs(smooth[sy0:sy1, sx0:sx1][ring] - mean[i]).max(axis=1)
            if float((d < tol).mean()) > 0.5:
                ok[i] = False
                small[i] = False

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


def _thin_strokes(labels, res, img, tolerance):
    """Adopt thin strokes of a flat colour into its region.

    A stroke only a few pixels wide (the hairline arc of a letter, a thin
    rule) never passes the region test - its neighbourhood is never quiet -
    so it keeps its mottle and fringe while the letters around it become
    solid.  Pixels whose median colour matches a flat colour join it when
    they form a long, thin component that is darker or lighter than all
    around it and has flat areas of other colours close by on all sides
    (not a photo).  Returns (labels, adopted mask).
    """
    h, w = labels.shape
    inside = labels > 0
    adopted = np.zeros((h, w), bool)
    if not inside.any():
        return labels, adopted
    band = max(10.0, 0.008 * min(h, w))
    med = cv2.medianBlur(np.ascontiguousarray(img, np.float32), 5)
    cnt = np.bincount(labels.ravel(), minlength=len(res.colours))
    groups: dict = {}
    for i in np.unique(labels[inside]):
        groups.setdefault(tuple(np.round(res.colours[i], 4)), []).append(int(i))
    out = labels.copy()
    k3 = np.ones((3, 3), np.uint8)
    for members in groups.values():
        lead = max(members, key=lambda i: cnt[i])
        cand = (~inside) & (np.abs(med - res.observed[lead]).max(axis=2) < 1.5 * tolerance)
        if not cand.any():
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
        if n <= 1:
            continue
        depth = cv2.distanceTransform(cand.astype(np.uint8), cv2.DIST_L2, 3)
        deepest = np.zeros(n, np.float32)
        np.maximum.at(deepest, comp.ravel(), depth.ravel())
        ring = (cv2.dilate(cand.astype(np.uint8), k3) > 0) & ~cand
        rc = cv2.dilate(comp.astype(np.float32), k3).astype(np.int32)
        ring_n = np.bincount(rc[ring], minlength=n)
        # flat colour of another kind close by on all sides (not a photo)
        other_flat = inside & ~np.isin(labels, members)
        to_other = cv2.distanceTransform((~other_flat).astype(np.uint8), cv2.DIST_L2, 3)
        ring_flat = np.bincount(rc[ring & (to_other <= 0.5 * band)], minlength=n)
        between = ring_flat >= 0.8 * np.maximum(ring_n, 1)
        # a stroke stands out from what surrounds it: just outside it there
        # is little of its own colour (unlike mottle at a region's edge, or
        # paper grain), ...
        grown5 = cv2.dilate(cand.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        outer = (cv2.dilate(cand.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & ~grown5
        rc2 = cv2.dilate(comp.astype(np.float32), np.ones((13, 13), np.uint8)).astype(np.int32)
        same_col = (np.abs(med - res.observed[lead]).max(axis=2) < 1.5 * tolerance) | np.isin(labels, members)
        outer_n = np.bincount(rc2[outer], minlength=n)
        outer_same = np.bincount(rc2[outer & same_col], minlength=n)
        stands_out = outer_same <= 0.25 * np.maximum(outer_n, 1)
        # a stroke is darker (or lighter) than everything around it; the
        # blur band between two other colours lies between them instead
        lum = med @ np.float32([0.114, 0.587, 0.299])
        own_lum = float(res.observed[lead] @ np.float32([0.114, 0.587, 0.299]))
        darker = np.bincount(rc2[outer & (lum < own_lum - 0.08)], minlength=n)
        lighter = np.bincount(rc2[outer & (lum > own_lum + 0.08)], minlength=n)
        tot = np.maximum(outer_n, 1)
        # mostly clearly lighter around it (a dark stroke) or mostly darker (a
        # light one); a ring of flare next to black ink has black on one side
        extreme = (((darker <= 0.1 * tot) & (lighter >= 0.7 * tot))
                   | ((lighter <= 0.1 * tot) & (darker >= 0.7 * tot)))
        stands_out &= extreme
        # ... and it is long and thin, not a blob or speckle of a photo
        area = stats[:, cv2.CC_STAT_AREA]
        long_thin = (area >= 400) & (area >= 30.0 * np.maximum(deepest, 1.0) ** 2)
        take = (deepest <= 0.4 * band) & long_thin & stands_out & between
        take[0] = False
        sel = take[comp]
        # hairline gaps of grain inside the stroke belong to it too
        sel = (cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0) & ~inside
        out[sel] = lead
        adopted |= sel
    return out, adopted


def _extend_sections(labels, res, img, tolerance, reach):
    """Grow every flat section out to its true edge and take in its specks.

    The region test stops a few pixels short of an outline, and mottle,
    dust or a ring of scanner flare inside a section stays unlabelled.
    Here each pixel within `reach` px of a section is given to it when its
    colour - after a median that removes grain and specks - is within
    1.5x the tolerance of the section's colour, and closer to it than to any
    other flat colour (connected to the section, so a photo nearby is not
    claimed beyond that reach).  Small enclosed holes surrounded by one
    section join it.  What remains between sections is the anti-aliased
    edge, redrawn afterwards.
    """
    h, w = labels.shape
    inside = labels > 0
    if not inside.any():
        return labels
    med = cv2.GaussianBlur(cv2.medianBlur(np.ascontiguousarray(img, np.float32), 5), (0, 0), 1.0)
    cnt = np.bincount(labels.ravel(), minlength=len(res.colours))
    groups: dict = {}
    for i in np.unique(labels[inside]):
        groups.setdefault(tuple(np.round(res.colours[i], 4)), []).append(int(i))
    members = list(groups.values())
    leads = [max(m, key=lambda i: cnt[i]) for m in members]
    obs = np.float32([res.observed[l] for l in leads])
    dist = np.stack([np.abs(med - o).max(axis=2) for o in obs])          # (K, h, w)
    nearest = dist.argmin(axis=0)
    dmin = dist.min(axis=0)
    out = labels.copy()
    for k, mem in enumerate(members):
        own = np.isin(labels, mem)
        near = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 3) <= reach
        cand = (~inside) & near & (nearest == k) & (dmin < 2.0 * tolerance)
        if not cand.any():
            continue
        # only pixels connected to the section itself
        n, comp = cv2.connectedComponents((cand | own).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(comp[own])] = True
        keep[0] = False
        grow = cand & keep[comp]
        out[grow] = leads[k]
    # enclosed holes (specks, dust, flare) up to ~1 mm across, inside one section
    free = (out == 0).astype(np.uint8)
    n, hl, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    if n > 1:
        size = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        small = size <= max(6, int(0.009 * min(h, w)))
        small[0] = False
        lut = np.zeros(len(res.colours), np.int32)
        for k, mem in enumerate(members):
            lut[mem] = k + 1
        grp = lut[out]
        ring = (cv2.dilate(free, np.ones((3, 3), np.uint8)) > 0) & (free == 0)
        rid = cv2.dilate(hl.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        sel = ring & small[rid]
        if sel.any():
            g_hi = np.zeros(n, np.int64)
            g_lo = np.full(n, 1 << 30, np.int64)
            np.maximum.at(g_hi, rid[sel], grp[sel])
            np.minimum.at(g_lo, rid[sel], grp[sel])
            one = small & (g_hi == g_lo) & (g_hi > 0)
            fill = one[hl]
            out[fill] = np.array([0] + leads)[g_hi[hl[fill]]]
    return out


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
    labels, thin = _thin_strokes(labels, res, img, tolerance)
    seeds = labels > 0
    labels = _extend_sections(labels, res, img, tolerance, reach=12.0)
    inside = labels > 0
    st = float(min(strength, 1.0))
    # Blend by closeness: a pixel near a region's colour (as scanned) becomes
    # the flat colour; one further off - grain or detail - is changed less,
    # fading out at the tolerance.  Pixels up to REACH px outside a region
    # (pockets the region test skipped, e.g. next to a stray hair) are
    # treated the same way, fading with distance, so a region never shows
    # as a block of flat colour among same-toned unflattened pixels.
    REACH = 12.0
    near_lab = labels.copy()
    if (~inside).any():
        grown = cv2.dilate(labels.astype(np.float32), np.ones((25, 25), np.uint8)).astype(np.int32)
        near_lab[~inside] = grown[~inside]
    dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
    prox = np.clip(1.0 - dist / REACH, 0.0, 1.0) * (near_lab > 0)
    seen = res.observed[near_lab]
    target = res.colours[near_lab]
    d = np.abs(cv2.GaussianBlur(img, (0, 0), 1.2) - seen).max(axis=2)
    # Inside a region, specks of mottle a few px across are not detail:
    # judge closeness after a median that removes them.
    med = cv2.GaussianBlur(cv2.medianBlur(np.ascontiguousarray(img, np.float32), 5), (0, 0), 1.0)
    close = np.clip((tolerance - d) / (0.5 * tolerance), 0.0, 1.0)
    # A section is one colour: grain, mottle and flare (judged after the
    # median) are painted fully up to the tolerance, fading out by twice it,
    # so a stain or soft shading blends in instead of ending in a hard edge.
    # Fine detail that clearly stands out - dots, hairlines - is kept.
    d_med = np.abs(med - seen).max(axis=2)
    d_fine = np.abs(cv2.GaussianBlur(img, (0, 0), 1.0) - seen).max(axis=2)
    sec = (np.clip((2.0 * tolerance - d_med) / tolerance, 0.0, 1.0)
           * np.clip((6.0 * tolerance - d_fine) / (2.0 * tolerance), 0.0, 1.0))
    close = np.where(inside, sec, close)
    wgt = cv2.GaussianBlur((close * prox).astype(np.float32), (0, 0), 1.0) * (near_lab > 0)
    out = img + (st * wgt)[:, :, None] * (target - img)
    mask = (wgt > 0.5).astype(np.uint8) * 255
    if clean_edges:
        out, seam = _rebuild_edges(out, img, smooth, labels, res, tolerance, st, wgt, near_lab, thin)
        mask[seam] = 255
    if protect is not None and protect.any():
        keep = protect > 0
        out[keep] = img[keep]
        mask[keep] = 0
    return out.astype(np.float32), mask


def _rebuild_edges(out, img, smooth, labels, res, tolerance, st, wmain, near_lab, thin):
    """Redraw the gap between two flat colours as a clean blend of them.

    A region stops a few pixels short of its true outline (the local-range
    test needs a quiet neighbourhood), so between, say, black lettering and
    a pink background lies a band of scanned pixels: ink fringe, mottle and
    the scanner's blur.  Each pixel in that band that lies within reach of
    two different flat colours, and whose colour is a mix of the two as
    scanned, is replaced by the same mix of the clean colours.  The mix is
    read from a lightly smoothed copy (no grain) and its ends are snapped,
    so the outline comes out crisp and evenly anti-aliased.
    """
    h, w = labels.shape
    inside = labels > 0
    band = max(10.0, 0.008 * min(h, w))
    # one entry per palette colour: the clean colour, as scanned, its area
    ids = np.unique(labels[inside])
    groups: dict = {}
    for i in ids:
        key = tuple(np.round(res.colours[i], 4))
        groups.setdefault(key, []).append(int(i))
    if len(groups) < 2:
        return out, np.zeros((h, w), bool)
    cnt = np.bincount(labels.ravel(), minlength=len(res.colours))
    # the three nearest colours and their distances, per pixel
    dist = np.full((3, h, w), np.inf, np.float32)
    kid = np.full((3, h, w), -1, np.int32)
    clean, seen = [], []
    for k, (key, members) in enumerate(groups.items()):
        m = np.isin(labels, members)
        lead = max(members, key=lambda i: cnt[i])
        clean.append(res.colours[lead]); seen.append(res.observed[lead])
        d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        for r in range(3):                      # insert into the sorted top three
            nearer = d < dist[r]
            for q in range(2, r, -1):
                dist[q] = np.where(nearer, dist[q - 1], dist[q])
                kid[q] = np.where(nearer, kid[q - 1], kid[q])
            dist[r] = np.where(nearer, d, dist[r])
            kid[r] = np.where(nearer, k, kid[r])
            d = np.where(nearer, np.inf, d)
    clean = np.float32(clean); seen = np.float32(seen)
    # region pixels are included only along the region's own border, where
    # ink mottle right at a letter's edge escaped the closeness test
    depth_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    rim = depth_in <= 4
    cand = ((kid[1] >= 0) & (dist[0] <= band) & (dist[1] <= 2 * band)
            & (~inside | (rim & (dist[1] <= band))))
    ys, xs = np.nonzero(cand)
    if not ys.size:
        return out, np.zeros((h, w), bool)
    sm = smooth[ys, xs]
    # Where three colours meet, use the pair the pixel is actually a mix of.
    best = None
    for p_, q_ in ((0, 1), (0, 2), (1, 2)):
        ka, kb = kid[p_][ys, xs], kid[q_][ys, xs]
        # a narrow gap between the two colours, not a wide strip of picture
        valid = (ka >= 0) & (kb >= 0) & (np.maximum(dist[p_][ys, xs], dist[q_][ys, xs]) <= band)
        ka_, kb_ = np.maximum(ka, 0), np.maximum(kb, 0)
        oa, ob = seen[ka_], seen[kb_]
        dv = oa - ob
        den = np.maximum((dv * dv).sum(axis=1), 1e-6)
        al = np.clip(((sm - ob) * dv).sum(axis=1) / den, 0.0, 1.0)
        rs = np.abs(sm - (ob + al[:, None] * dv)).max(axis=1)
        rs = np.where(valid & (den > (1.5 * tolerance) ** 2), rs, np.inf)
        # prefer the colours that are actually next to the pixel
        score = rs + tolerance * (dist[p_][ys, xs] + dist[q_][ys, xs]) / band
        if best is None:
            best = [score, rs, al, ka_, kb_, dist[p_][ys, xs], dist[q_][ys, xs]]
        else:
            take = score < best[0]
            for j, v in enumerate((score, rs, al, ka_, kb_, dist[p_][ys, xs], dist[q_][ys, xs])):
                best[j] = np.where(take, v, best[j])
    _, resid, alpha, a_, b_, dist_a, dist_b = best
    # Confidence that the pixel is a mix of the two colours, as soft weights
    # smoothed over neighbours, so the decision does not flicker with grain.
    fit = np.clip((1.5 * tolerance - resid) / (0.5 * tolerance), 0.0, 1.0)
    # A true mix only occurs where the colour is changing (the blur across
    # an edge) or just off one of the two colours (flare beside dark ink).
    # A steady band of an in-between colour - a grey shadow between a green
    # card and white paper - is a third colour and is left alone.
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = cv2.dilate(np.sqrt(gx * gx + gy * gy).max(axis=2), np.ones((3, 3), np.uint8))[ys, xs]
    dab = np.sqrt(np.maximum(((seen[a_] - seen[b_]) ** 2).sum(axis=1), 1e-6))
    off = np.minimum(np.abs(sm - seen[a_]).max(axis=1), np.abs(sm - seen[b_]).max(axis=1))
    # Pixels (almost) purely one colour are repainted only when as close to
    # it as the main blend requires - otherwise unflattened paper next to an
    # edge would turn into a clean ring.  In-between pixels need the edge.
    mixed = np.clip(8.0 * alpha * (1.0 - alpha), 0.0, 1.0)
    # printed edges go from one colour to the other within a few pixels;
    # photographic shading between black and white changes slowly
    steep = np.clip((grad / dab - 0.06) / 0.05, 0.0, 1.0)
    # How much of the surrounding area of the pixel's main colour the main
    # blend flattened: where it is all flat (a solid background) the pixel
    # follows; where the paper around was left as scanned, it stays too.
    lut = np.full(len(res.colours), -1, np.int32)
    for k, (key, members) in enumerate(groups.items()):
        lut[members] = k
    ncol = np.where(near_lab > 0, lut[near_lab], -1)
    major = np.where(alpha >= 0.5, a_, b_)
    cover = np.zeros(ys.size, np.float32)
    for k in np.unique(major):
        mk = ((ncol == k) & ((depth_in > 6) | thin)).astype(np.float32)   # region core
        num = cv2.GaussianBlur(wmain * mk, (0, 0), 16.0)
        den_ = cv2.GaussianBlur(mk, (0, 0), 16.0)
        sel = major == k
        dd = den_[ys[sel], xs[sel]]
        cover[sel] = np.where(dd > 1e-4, num[ys[sel], xs[sel]] / np.maximum(dd, 1e-4), 0.0)
    cover = np.clip((cover - 0.6) / 0.3, 0.0, 1.0)
    gate = np.maximum.reduce([np.clip((tolerance - off) / (0.5 * tolerance), 0.0, 1.0),
                              mixed * steep,
                              cover * np.maximum(mixed * steep,
                                                 np.clip((2.0 * tolerance - off) / (0.8 * tolerance), 0.0, 1.0))])
    # between two sections that are flat all round, every pixel of the edge
    # band is the edge: redraw it as the clean mix, no partial leftovers
    gate = np.maximum(gate, cover * ((dist_a + dist_b) <= 0.75 * band))
    if DEBUG is not None:
        DEBUG.update(ys=ys, xs=xs, fit=fit, gate=gate, cover=cover, off=off, alpha=alpha,
                     mixed=mixed, steep=steep, dist_a=dist_a, dist_b=dist_b, inside=inside[ys, xs],
                     a=a_, b=b_, clean=clean, seen=seen, labels=labels)
    wmap = np.zeros((h, w), np.float32)
    # Fade with distance from the colours painted, as the main blend does,
    # so plain paper further out is not painted where its neighbours are not.
    dw = alpha * dist_a + (1 - alpha) * dist_b
    near = np.clip(1.5 * (1.0 - dw / band), 0.0, 1.0)
    near = np.maximum(near, cover)       # a fully flattened colour may reach further
    wmap[ys, xs] = fit * gate * near
    wmap = cv2.GaussianBlur(wmap, (0, 0), 1.5)
    wv = wmap[ys, xs] * (fit > 0)        # no blur onto pixels that are no mix
    # crisp ends: within 10% of either colour it is that colour; steep
    # (printed) edges are also tightened a little
    lin = np.clip((alpha - 0.1) / 0.8, 0.0, 1.0)
    alpha = lin + steep * (lin * lin * (3 - 2 * lin) - lin)
    mix = clean[b_] + alpha[:, None] * (clean[a_] - clean[b_])
    k = (st * wv)[:, None]
    out[ys, xs] = out[ys, xs] + k * (mix - out[ys, xs])
    seam = np.zeros((h, w), bool)
    seam[ys, xs] = wv > 0.5
    return out, seam
