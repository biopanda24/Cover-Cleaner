"""Compare Auto's crop with hand-cropped references (from ref_match.py).

    python tools/ref_compare.py Scans output/ref_truth.json [--only a,b] [--out sheet.jpg]

Per scan and side: signed offset in px of Auto's final crop (after trims)
from the reference crop, measured along the reference's inward normal.
+ = Auto cut deeper than the reference, - = Auto left extra outside it.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SIDES = ("top", "right", "bottom", "left")


def auto_crop(path):
    from covercleaner import geometry as G
    from covercleaner.imageio import load_image, read_dpi
    img = load_image(path)
    det = G.detect_cover(img, dpi=read_dpi(path))
    mx, my = G.build_maps(det.corners, det.bow if det.max_bow >= 1.5 else None)
    h, w = mx.shape
    l, t, r, b = det.trim
    pts = []
    for x, y in ((l, t), (w - 1 - r, t), (w - 1 - r, h - 1 - b), (l, h - 1 - b)):
        pts.append([float(mx[y, x]), float(my[y, x])])
    return np.array(pts), det


def offsets(ref, auto):
    """Signed offsets per side (top, right, bottom, left)."""
    c = ref.mean(axis=0)
    out = []
    for i in range(4):
        a, b = ref[i], ref[(i + 1) % 4]
        t = (b - a) / np.hypot(*(b - a))
        n = np.array([-t[1], t[0]])
        if np.dot(c - a, n) < 0:
            n = -n                                         # inward normal
        p, q = auto[i], auto[(i + 1) % 4]
        out.append(float((np.dot(p - a, n) + np.dot(q - a, n)) / 2))
    return out


def one(args):
    scans, name, ref = args
    auto, det = auto_crop(os.path.join(scans, name))
    off = offsets(np.array(ref["corners"]), auto)
    return name, off, det.notes, [round(c, 2) for c in det.side_confidence]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("truth")
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    truth = json.load(open(a.truth))
    names = sorted(truth)
    if a.only:
        names = [n for n in names if any(k in n for k in a.only.split(","))]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, [(a.scans, n, truth[n]) for n in names]))
    allv = []
    for name, off, notes, conf in res:
        flag = " <<" if max(abs(v) for v in off) > 15 else ""
        print(f"{name[8:-4]:22s} " + " ".join(f"{s[0]}{v:+6.0f}" for s, v in zip(SIDES, off)) + flag)
        allv.extend(off)
    v = np.array(allv)
    print(f"\nsides {len(v)}: median {np.median(v):+.1f}  mean|.| {np.abs(v).mean():.1f}  "
          f"within 5px {np.mean(np.abs(v) <= 5) * 100:.0f}%  within 15px {np.mean(np.abs(v) <= 15) * 100:.0f}%  "
          f"worse than 50px {np.sum(np.abs(v) > 50)}")
    if a.json:
        json.dump({n: dict(off=o, notes=nt, conf=c) for n, o, nt, c in res}, open(a.json, "w"), indent=1)
