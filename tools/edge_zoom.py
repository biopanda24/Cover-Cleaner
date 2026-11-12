"""Full-resolution crops of the detected outline at the middle of each side.

    python tools/edge_zoom.py Scans/x.png out.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from covercleaner import geometry as G  # noqa: E402
from covercleaner.imageio import load_image, read_dpi, to_uint8  # noqa: E402

img = load_image(sys.argv[1])
d = G.detect_cover(img, dpi=read_dpi(sys.argv[1]))
vis = to_uint8(img).copy()
cv2.polylines(vis, [G.outline_points(d.corners, d.bow).astype(np.int32)], True, (0, 0, 255), 1)
c = d.corners
tiles = []
R = 60
for i in range(4):
    m = (c[i] + c[(i + 1) % 4]) / 2
    x, y = int(m[0]), int(m[1])
    x0 = int(np.clip(x - R, 0, vis.shape[1] - 2 * R))
    y0 = int(np.clip(y - R, 0, vis.shape[0] - 2 * R))
    t = vis[y0:y0 + 2 * R, x0:x0 + 2 * R]
    if i in (1, 3):                       # make every edge horizontal, outside up
        t = np.rot90(t, 1 if i == 1 else -1)
    if i == 2:
        t = t[::-1]
    tiles.append(np.pad(cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                        ((0, 6), (0, 6), (0, 0))))
print(G.output_size(d.corners), [round(v, 2) for v in d.side_confidence], d.trim)
cv2.imwrite(sys.argv[2], np.hstack(tiles))
