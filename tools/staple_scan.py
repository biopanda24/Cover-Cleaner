"""Run staple detection on every scan's crop and draw the hits.

    python tools/staple_scan.py Scans out.jpg
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def one(path):
    from covercleaner import geometry as G, staples
    from covercleaner.imageio import load_image, read_dpi, to_uint8
    img = load_image(path)
    dpi = read_dpi(path)
    d = G.detect_cover(img, dpi=dpi)
    rect = G.apply_geometry(img, d.corners, None, d.trim) if d.has_background else img
    r = staples.detect(rect, dpi)
    if not r.staples:
        return None
    vis = to_uint8(rect).copy()
    for s in r.staples:
        cv2.rectangle(vis, s.box[:2], s.box[2:], (0, 0, 255), 8)
    t = cv2.resize(vis, (400, 400), interpolation=cv2.INTER_AREA)
    cv2.putText(t, os.path.basename(path)[-14:-4], (4, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return os.path.basename(path), [(s.side, s.box, round(s.depth), s.solid) for s in r.staples], t


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    names = sorted(n for n in os.listdir(src) if n.endswith(".png"))
    with ProcessPoolExecutor(8) as ex:
        res = [r for r in ex.map(one, [os.path.join(src, n) for n in names]) if r]
    for n, st, _ in res:
        print(n, st)
    tiles = [t for _, _, t in res]
    while tiles and len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    if tiles:
        cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)]))
    print(len(res), "scans with staples")
