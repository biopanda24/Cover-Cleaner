"""Staples along a booklet's spine.

On a stapled booklet the staples show, right at the folded spine edge, as
thin silver or grey lines 7-17 mm long.  Along each edge of the rectified
cover the strip within ~0.5 mm of the edge is compared with the paper
1.3-2.2 mm in: a staple makes that strip darker and neutral, while the paper
just behind it (0.6-1.3 mm in) is back to normal - printing that runs off
the edge stays dark further in and is not taken for a staple.

Each staple is then either filled in (it sits on solid colour) or the side
is cropped just past it (it sits on picture).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LEN_MM = (7.0, 17.0)    # staple length range
DEBUG = False


@dataclass
class Staple:
    side: int                     # 0 top, 1 right, 2 bottom, 3 left
    box: tuple                    # x0, y0, x1, y1 in the image
    depth: float                  # px from the edge to the staple's inner side
    solid: bool                   # surrounded by solid colour


@dataclass
class StapleResult:
    staples: list = field(default_factory=list)
    side: int = -1


def _side_view(img: np.ndarray, side: int, depth: int) -> np.ndarray:
    """The strip along one side, rotated so the edge is at the top."""
    if side == 0:
        return img[:depth]
    if side == 1:
        return np.rot90(img[:, -depth:], 1)
    if side == 2:
        return img[-depth:][::-1, ::-1]
    return np.rot90(img[:, :depth], -1)


def _to_image_box(side, x0, y0, x1, y1, h, w):
    """Map a box from the side view (edge at top) back to image coordinates."""
    if side == 0:
        return x0, y0, x1, y1
    if side == 1:            # view = rot90(img[:, -d:], 1): view (x, y) -> img (w-1-y, x)
        return w - 1 - y1, x0, w - 1 - y0, x1
    if side == 2:            # view = img[-d:][::-1, ::-1]
        return w - 1 - x1, h - 1 - y1, w - 1 - x0, h - 1 - y0
    # side 3: view = rot90(img[:, :d], -1): view (x, y) -> img (y, h-1-x)
    return y0, h - 1 - x1, y1, h - 1 - x0


def _runs(mask: np.ndarray, join: int) -> list:
    """(start, end) runs of True, joining runs closer than ``join``."""
    runs, start = [], None
    for x in range(len(mask) + 1):
        on = x < len(mask) and mask[x]
        if on and start is None:
            start = x
        elif not on and start is not None:
            if runs and start - runs[-1][1] < join:
                runs[-1] = (runs[-1][0], x)
            else:
                runs.append((start, x))
            start = None
    return runs


def detect(img: np.ndarray, dpi: float | None = None) -> StapleResult:
    """Find staples along the edges of a rectified cover (float BGR).

    Only the side with the strongest evidence is kept: a booklet has one
    stapled spine.  Staples come in pairs placed symmetrically about the
    middle of the spine, so when only one is seen (the other lost in a
    picture or already trimmed off) its partner is assumed at the mirrored
    place.
    """
    h, w = img.shape[:2]
    px = (dpi or 600.0) / 25.4
    line = max(4, int(0.5 * px))            # staple zone at the edge
    gap0, gap1 = int(0.6 * px), int(1.3 * px)
    ref0, ref1 = int(1.3 * px), int(2.2 * px)
    lmin, lmax = LEN_MM[0] * px, LEN_MM[1] * px
    k = int(0.6 * px) | 1                   # smoothing along the edge
    o = int(px)
    best = StapleResult()
    best_score = 0.0
    for side in range(4):
        view = np.ascontiguousarray(_side_view(img, side, ref1))
        vw = view.shape[1]
        if vw < 4 * lmax:
            continue

        def prof(a, b):
            return cv2.blur(view[a:b].mean(axis=0)[None].astype(np.float32), (k, 1))[0]

        paper = prof(ref0, ref1)                                     # (vw, 3)
        behind = prof(gap0, gap1)
        darkest = cv2.blur(np.sort(view[:line], axis=0)[1][None].astype(np.float32), (k, 1))[0]
        dark = (paper - darkest).mean(axis=1)                        # >0: zone darker
        neutral = (darkest.max(axis=1) - darkest.min(axis=1)) < 0.10
        clean_behind = np.abs(behind - paper).max(axis=1) < 0.05
        hit = (dark > 0.05) & neutral & clean_behind
        runs = []
        for x0, x1 in _runs(hit, o):
            if not lmin <= x1 - x0 <= lmax:
                continue
            frac = (x0 + x1) / 2 / vw
            inside = float(np.median(dark[x0:x1]))
            outside = max(float(np.median(dark[max(0, x0 - 5 * o):x0 - o])) if x0 - o > 0 else 0.0,
                          float(np.median(dark[x1 + o:x1 + 5 * o])) if x1 + o < vw else 0.0)
            bright = float(np.median(paper[x0:x1].mean(axis=1)))
            ok = (outside <= 0.45 * inside
                  and bright - inside >= 0.25          # silver, not black print
                  and ((bright >= 0.75 and inside >= 0.1) or (inside >= 0.2 and inside >= 0.33 * bright))
                  and (0.1 <= frac <= 0.4 or 0.6 <= frac <= 0.9))
            if DEBUG:
                print("  side", side, "run", x0, x1, "len %.1fmm in %.3f out %.3f paper %.2f frac %.2f ok %s"
                      % ((x1 - x0) / px, inside, outside, bright, frac, ok))
            if ok:
                runs.append((x0, x1, inside))
        if not runs or len(runs) > 2:
            continue
        score = sum((x1 - x0) * v for x0, x1, v in runs)
        if score <= best_score:
            continue
        spans = [(x0, x1) for x0, x1, _ in runs]
        if len(spans) == 1:                  # the partner, near the mirrored place
            x0, x1 = spans[0]
            m0, m1 = vw - x1, vw - x0
            a, b = max(0, m0 - 6 * o), min(vw, m1 + 6 * o)
            near = [(a + s, a + e) for s, e in _runs((dark[a:b] > 0.03) & neutral[a:b], o)]
            near = [r for r in near if r[1] - r[0] >= 5 * o]
            if near:
                spans.append(max(near, key=lambda r: r[1] - r[0]))
            else:
                spans.append((max(0, m0 - 2 * o), min(vw, m1 + 2 * o)))
        found = []
        pad = int(0.4 * px)
        for x0, x1 in spans:
            bx0, bx1 = max(0, x0 - pad), min(vw - 1, x1 + pad)
            seg = view[:gap1, bx0:bx1]
            rows = (paper[bx0:bx1].mean(axis=0) - seg.mean(axis=1)).mean(axis=1)
            hits = np.flatnonzero(rows[:line + 3] > 0.03)
            dep = int(hits.max() if hits.size else line) + 3
            around = np.concatenate([view[ref0:ref1, bx0:bx1].reshape(-1, 3),
                                     view[:dep, max(0, bx0 - pad):bx0].reshape(-1, 3),
                                     view[:dep, bx1:min(vw, bx1 + pad)].reshape(-1, 3),
                                     view[dep:gap1, bx0:bx1].reshape(-1, 3)])
            spread = np.abs(around - np.median(around, axis=0)).max(axis=1)
            solid = float(np.percentile(spread, 90)) < 0.06
            box = _to_image_box(side, bx0, 0, bx1, dep, h, w)
            found.append(Staple(side, tuple(int(v) for v in box), float(dep), solid))
        best, best_score = StapleResult(found, side), score
    return best
