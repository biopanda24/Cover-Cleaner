"""ML backend: LaMa inpainting (Resolution-robust Large Mask Inpainting).

The traditional pipeline fills defects by diffusing the surrounding colours
inwards (Telea), which is exact on flat colour but smears texture.  LaMa is
a neural network trained to continue texture and structure through a hole,
so repairs inside photos, hair, fabric or patterns look natural.

The model file comes from OpenCV's model zoo
(huggingface.co/opencv/inpainting_lama, Apache-2.0) and runs with OpenCV's
own DNN module, so no extra framework is needed.  It works on 512x512
windows; each window is cut at native resolution around a defect so no
detail is lost to resizing.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import cv2
import numpy as np

MODEL_FILE = "inpainting_lama_2025jan.onnx"
MODEL_URL = "https://huggingface.co/opencv/inpainting_lama/resolve/main/" + MODEL_FILE
SIZE = 512

_net = None
_lock = threading.Lock()


def model_path() -> str:
    env = os.environ.get("COVERCLEANER_MODELS")
    root = env or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    return os.path.join(root, MODEL_FILE)


def available() -> bool:
    return os.path.isfile(model_path())


def _load():
    global _net
    if _net is None:
        if not available():
            raise FileNotFoundError(
                f"ML model not found: {model_path()}\nDownload {MODEL_URL} into that folder.")
        _net = cv2.dnn.readNetFromONNX(model_path())
    return _net


def _infer(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint one 512x512 float BGR patch (0..1). mask: uint8, >0 = hole."""
    net = _load()
    rgb = np.ascontiguousarray(patch[:, :, ::-1])
    hole = (mask > 0).astype(np.float32)
    # the network expects the hole to be blanked out
    blob = (rgb * (1.0 - hole[:, :, None])).transpose(2, 0, 1)[None].astype(np.float32)
    net.setInput(blob, "image")
    net.setInput(hole[None, None], "mask")
    out = net.forward()[0].transpose(1, 2, 0)            # RGB, 0..255
    return np.clip(out[:, :, ::-1] / 255.0, 0.0, 1.0).astype(np.float32)


def _window(lo: int, hi: int, limit: int, size: int) -> int:
    """Start of a `size` window covering [lo, hi) as centrally as possible."""
    start = (lo + hi) // 2 - size // 2
    return int(np.clip(start, 0, max(0, limit - size)))


def inpaint(img: np.ndarray, mask: np.ndarray,
            progress: Optional[Callable[[str, float], None]] = None) -> np.ndarray:
    """Fill masked pixels of a float BGR image with LaMa."""
    if mask is None or not mask.any():
        return img
    h, w = img.shape[:2]
    todo = (mask > 0).astype(np.uint8)
    # Holes surrounded by flat colour are filled exactly by plain diffusion;
    # the network is only better where there is texture or structure.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(todo, 8)
    near = cv2.dilate(labels.astype(np.float32), np.ones((13, 13), np.uint8))
    ring = (labels == 0) & (near > 0)
    ids = near[ring].astype(np.int64)
    cnt = np.maximum(np.bincount(ids, minlength=n), 1).astype(np.float64)
    spread = np.zeros(n)
    for c in range(3):
        v = img[:, :, c][ring].astype(np.float64)
        mean = np.bincount(ids, v, minlength=n) / cnt
        var = np.bincount(ids, v * v, minlength=n) / cnt - mean * mean
        spread = np.maximum(spread, np.sqrt(np.maximum(var, 0)))
    simple = spread < 0.012
    simple[0] = False
    out = img
    if simple.any():
        from .defects import inpaint as diffuse
        easy = (simple[labels]).astype(np.uint8) * 255
        out = diffuse(img, easy)
        todo[easy > 0] = 0
    if out is img:
        out = img.copy()
    total = int(todo.sum())
    if total == 0:
        return out
    with _lock:                                          # the net is not re-entrant
        while True:
            n, _, stats, _ = cv2.connectedComponentsWithStats(todo, 8)
            if n <= 1:
                break
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            # Context: at least as much surrounding picture as hole.
            need = int(max(bw, bh) * 2.0) + 64
            win = max(SIZE, need)
            win = min(win, max(h, w))
            x0 = _window(x, x + bw, w, win)
            y0 = _window(y, y + bh, h, win)
            x1, y1 = min(w, x0 + win), min(h, y0 + win)
            patch = out[y0:y1, x0:x1]
            pm = todo[y0:y1, x0:x1] * 255
            ph, pw = patch.shape[:2]
            if (ph, pw) != (SIZE, SIZE):
                small = cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                sm = cv2.resize(pm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                sm = cv2.dilate(sm, np.ones((3, 3), np.uint8))
                fill = cv2.resize(_infer(small, sm), (pw, ph), interpolation=cv2.INTER_CUBIC)
            else:
                fill = _infer(patch, pm)
            # feather the paste so there is no visible seam
            soft = cv2.GaussianBlur((pm > 0).astype(np.float32), (0, 0), 1.0)
            soft = np.maximum(soft, (pm > 0).astype(np.float32))[:, :, None]
            patch[:] = patch + soft * (fill - patch)
            todo[y0:y1, x0:x1] = 0
            if progress:
                progress("ML inpainting", 1.0 - float(todo.sum()) / max(total, 1))
    return np.clip(out, 0.0, 1.0)
