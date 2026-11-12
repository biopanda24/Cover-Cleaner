"""Zoomable image view with crop handles, mask overlay and brush input."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsScene,
                               QGraphicsView)

from ..imageio import to_uint8


def to_pixmap(img: np.ndarray) -> QPixmap:
    """float BGR (0..1) or uint8 BGR -> QPixmap."""
    u8 = img if img.dtype == np.uint8 else to_uint8(img)
    rgb = np.ascontiguousarray(cv2.cvtColor(u8, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def mask_pixmap(layers: list[tuple[np.ndarray, tuple[int, int, int]]], alpha: int = 150) -> QPixmap:
    """Coloured translucent overlay from one or more uint8 masks (r, g, b)."""
    h, w = layers[0][0].shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    for mask, (r, g, b) in layers:
        sel = mask > 0
        rgba[sel] = (r, g, b, alpha)
    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class _Handle(QGraphicsEllipseItem):
    def __init__(self, index: int, view: "ImageView"):
        r = 7
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.index = index
        self.view = view
        self.setBrush(QBrush(QColor(0, 220, 90)))
        self.setPen(QPen(QColor(0, 0, 0), 1.5))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges
                      | QGraphicsItem.ItemIgnoresTransformations)
        self.setCursor(Qt.CrossCursor)
        self.setZValue(20)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and not self.view._moving_handles:
            self.view.cornerMoved.emit(self.index, value.x(), value.y())
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.view.cornerReleased.emit()


class ImageView(QGraphicsView):
    cornerMoved = Signal(int, float, float)
    cornerReleased = Signal()
    strokeFinished = Signal(object)        # list of (x, y) in image pixels
    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(48, 48, 52))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.SmoothTransformation)
        self._overlay = QGraphicsPixmapItem()
        self._overlay.setZValue(5)
        self._outline = QGraphicsPolygonItem()
        pen = QPen(QColor(0, 230, 90), 0)
        self._outline.setPen(pen)
        self._outline.setZValue(10)
        self._stroke_item = QGraphicsPathItem()
        self._stroke_item.setZValue(15)
        for it in (self._pix, self._overlay, self._outline, self._stroke_item):
            self.scene().addItem(it)
        self._handles = [_Handle(i, self) for i in range(4)]
        for hnd in self._handles:
            self.scene().addItem(hnd)
            hnd.hide()
        self._moving_handles = False
        self.mode = "pan"                 # pan | brush | pick
        self.brush_size = 24
        self.brush_color = QColor(255, 60, 60, 150)
        self._stroke: list[tuple[float, float]] = []
        self._has_image = False

    # ---- content ---------------------------------------------------------
    def set_image(self, pixmap: QPixmap, keep_view: bool = True) -> None:
        old = self._pix.pixmap().size()
        self._pix.setPixmap(pixmap)
        self.scene().setSceneRect(self._pix.boundingRect())
        if not self._has_image or not keep_view or old != pixmap.size():
            self.fit()
        self._has_image = True

    def set_overlay(self, pixmap: QPixmap | None, scale: float = 1.0) -> None:
        if pixmap is None:
            self._overlay.setPixmap(QPixmap())
        else:
            self._overlay.setPixmap(pixmap)
            self._overlay.setScale(scale)

    def set_outline(self, points: np.ndarray | None, corners: np.ndarray | None) -> None:
        if points is None:
            self._outline.setPolygon(QPolygonF())
            for hnd in self._handles:
                hnd.hide()
            return
        self._outline.setPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
        self._moving_handles = True
        for hnd, (x, y) in zip(self._handles, corners):
            hnd.setPos(float(x), float(y))
            hnd.show()
        self._moving_handles = False

    def update_outline_only(self, points: np.ndarray) -> None:
        self._outline.setPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))

    def fit(self) -> None:
        if not self._pix.pixmap().isNull():
            self.fitInView(self._pix, Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self.resetTransform()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "pan" else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.CrossCursor if mode != "pan" else Qt.OpenHandCursor)

    # ---- events ----------------------------------------------------------
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        scale = self.transform().m11() * factor
        if 0.02 < scale < 40:
            self.scale(factor, factor)

    def _img_pos(self, event) -> tuple[float, float]:
        p = self.mapToScene(event.position().toPoint())
        return p.x(), p.y()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode == "brush":
            self._stroke = [self._img_pos(event)]
            pen = QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            self._stroke_item.setPen(pen)
            self._draw_stroke()
            return
        if event.button() == Qt.LeftButton and self.mode == "pick":
            self.picked.emit(*self._img_pos(event))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._stroke and self.mode == "brush":
            self._stroke.append(self._img_pos(event))
            self._draw_stroke()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._stroke and self.mode == "brush":
            pts, self._stroke = self._stroke, []
            self._stroke_item.setPath(QPainterPath())
            self.strokeFinished.emit(pts)
            return
        super().mouseReleaseEvent(event)

    def _draw_stroke(self):
        path = QPainterPath(QPointF(*self._stroke[0]))
        for x, y in self._stroke[1:]:
            path.lineTo(x, y)
        if len(self._stroke) == 1:
            path.lineTo(self._stroke[0][0] + 0.01, self._stroke[0][1])
        self._stroke_item.setPath(path)
