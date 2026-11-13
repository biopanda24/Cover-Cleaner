"""Which way up is it?  Text detection + recognition (OpenCV model zoo).

A disc has no edges that say which way is up, but its text does: text
recognised the right way up reads with high confidence, sideways or upside
down it does not.  The image is tried at all four quarter turns; in each,
lines of text are found (PP-OCRv3 DB detector) and read (CRNN recogniser),
and the turn whose text reads most confidently wins.

Model files (Apache-2.0), from huggingface.co/opencv:
    text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx
    text_recognition_crnn/text_recognition_CRNN_CH_2021sep.onnx
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

DET_FILE = "text_detection_en_ppocrv3_2023may.onnx"
REC_FILE = "text_recognition_CRNN_CH_2021sep.onnx"
DET_SIZE = 736                       # detector input, multiple of 32

_lock = threading.Lock()
_det = None
_rec = None


def _models_dir() -> str:
    return os.environ.get("COVERCLEANER_MODELS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def model_paths() -> list[str]:
    d = _models_dir()
    return [os.path.join(d, DET_FILE), os.path.join(d, REC_FILE)]


def available() -> bool:
    d = _models_dir()
    return os.path.isfile(os.path.join(d, DET_FILE)) and os.path.isfile(os.path.join(d, REC_FILE))


def _load():
    global _det, _rec
    if _det is None:
        d = _models_dir()
        det = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(os.path.join(d, DET_FILE)))
        det.setBinaryThreshold(0.3)
        det.setPolygonThreshold(0.5)
        det.setUnclipRatio(2.0)
        det.setMaxCandidates(200)
        det.setInputSize((DET_SIZE, DET_SIZE))
        det.setInputMean((123.675, 116.28, 103.53))
        det.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
        _det = det
        _rec = cv2.dnn.readNet(os.path.join(d, REC_FILE))
    return _det, _rec


_REC_W, _REC_H = 100, 32
_TARGET = np.array([[0, _REC_H - 1], [0, 0], [_REC_W - 1, 0], [_REC_W - 1, _REC_H - 1]],
                   np.float32)


def _read_confidence(img8: np.ndarray, box: np.ndarray, rec) -> tuple[float, int]:
    """Mean confidence of the characters read in one text box, and their count."""
    m = cv2.getPerspectiveTransform(box.astype(np.float32), _TARGET)
    crop = cv2.warpPerspective(img8, m, (_REC_W, _REC_H))
    blob = cv2.dnn.blobFromImage(crop, size=(_REC_W, _REC_H), mean=127.5, scalefactor=1 / 127.5)
    rec.setInput(blob)
    out = rec.forward()[:, 0, :]                     # (frames, classes), 0 = blank
    e = np.exp(out - out.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    best = prob.argmax(axis=1)
    chars = best != 0
    if chars.sum() == 0:
        return 0.0, 0
    return float(prob.max(axis=1)[chars].mean()), int(chars.sum())


def _turn_score(img8: np.ndarray, det, rec) -> tuple[float, int]:
    """How well the text reads with the image the way it is.

    Box corners are re-ordered by their position in the image, so every crop
    is read left-to-right as the image stands (the detector's own corner
    order would turn sideways text upright again).  The score is the number
    of characters read with at least 80 % confidence.
    """
    boxes, _ = det.detect(cv2.resize(img8, (DET_SIZE, DET_SIZE), interpolation=cv2.INTER_AREA))
    sx = img8.shape[1] / DET_SIZE
    sy = img8.shape[0] / DET_SIZE
    total, lines = 0, 0
    for b in boxes:
        b = np.asarray(b, np.float32) * [sx, sy]
        tl = b[np.argmin(b[:, 0] + b[:, 1])]
        br = b[np.argmax(b[:, 0] + b[:, 1])]
        tr = b[np.argmin(b[:, 1] - b[:, 0])]
        bl = b[np.argmax(b[:, 1] - b[:, 0])]
        w = np.hypot(*(tr - tl))
        hgt = np.hypot(*(bl - tl))
        if hgt < 8 or w < 2.0 * hgt:                  # horizontal lines of text only
            continue
        conf, n = _read_confidence(img8, np.array([bl, tl, tr, br]), rec)
        if conf >= 0.8 and n >= 3:
            total += n
            lines += 1
    return float(total), lines


def upright_turns(img: np.ndarray):
    """Counter-clockwise quarter turns (0-3) that make the text upright.

    Returns (turns, margin, scores): scores are (confidently read characters,
    lines) per turn, margin is the winner's character count divided by the
    runner-up's (inf when only one turn reads).  (None, 0, scores) when no
    text reads.
    """
    det, rec = _load()
    img8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    scores = []
    with _lock:
        for k in range(4):
            scores.append(_turn_score(np.ascontiguousarray(np.rot90(img8, k)), det, rec))
    vals = [s for s, _ in scores]
    order = np.argsort(vals)[::-1]
    best, second = int(order[0]), int(order[1])
    if vals[best] == 0:
        return None, 0.0, scores
    margin = float(vals[best] / vals[second]) if vals[second] > 0 else float("inf")
    return best, margin, scores
