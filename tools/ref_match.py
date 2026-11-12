"""Locate hand-cropped reference covers in their source scans.

    python tools/ref_match.py References Scans out.json

For each reference image (same file name as its scan) a similarity transform
(rotation, scale, shift) from the reference to the scan is estimated with
SIFT features + RANSAC.  The reference's corners mapped into the scan are
the hand-chosen crop.  Written to out.json as
    {name: {"corners": [[x, y] * 4 TL TR BR BL], "angle": deg, "scale": s,
            "inliers": n, "ref_size": [w, h], "alpha_corners": bool}}
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


def _read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)


def one(args):
    ref_dir, scan_dir, name = args
    ref = _read(os.path.join(ref_dir, name))
    scan = _read(os.path.join(scan_dir, name))
    alpha = ref.shape[2] == 4 and bool((ref[:, :, 3] < 250).mean() > 0.001)
    ref3 = ref[:, :, :3]
    scan3 = scan[:, :, :3]
    s = 0.35
    a = cv2.cvtColor(cv2.resize(ref3, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(scan3, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good]) / s
    pb = np.float32([kb[g.trainIdx].pt for g in good]) / s
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000)
    # refine on full resolution with ECC (sub-pixel)
    h, w = ref3.shape[:2]
    try:
        warp = M.astype(np.float32)
        gr = cv2.cvtColor(ref3, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gs = cv2.cvtColor(scan3, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        inv = cv2.invertAffineTransform(warp)
        # warp scan into ref frame and align there (ECC wants template = ref)
        _, inv = cv2.findTransformECC(gr, gs, cv2.invertAffineTransform(inv).astype(np.float32)
                                      if False else warp, cv2.MOTION_AFFINE,
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6),
                                      None, 5)
        M = inv
    except cv2.error:
        pass
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    cs = corners @ M[:, :2].T + M[:, 2]
    ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    return name, dict(corners=cs.tolist(), angle=ang, scale=scale, inliers=int(inl.sum()),
                      ref_size=[w, h], alpha_corners=alpha)


if __name__ == "__main__":
    ref_dir, scan_dir, out = sys.argv[1:4]
    names = sorted(n for n in os.listdir(ref_dir) if n.lower().endswith(".png"))
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(one, [(ref_dir, scan_dir, n) for n in names]))
    json.dump(res, open(out, "w"), indent=1)
    for n, r in res.items():
        print(f"{n[8:-4]:22s} angle {r['angle']:+.2f} scale {r['scale']:.4f} inliers {r['inliers']:4d} "
              f"alpha {r['alpha_corners']}")
