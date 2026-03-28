from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame
from PyQt6.QtGui import (QPixmap, QImage, QWheelEvent, QMouseEvent,
                         QKeyEvent, QColor, QPainter, QCursor)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal

from .constants import MIN_ZOOM, MAX_ZOOM, ZOOM_STEP, BG_COLOR


class AnnotationCanvas(QGraphicsView):
    """Phase 2 — image display, zoom, pan only."""

    zoom_changed  = pyqtSignal(float)
    cursor_moved  = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_factor  = 1.0
        self._panning      = False
        self._pan_start    = QPointF()
        self._space_held   = False

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(BG_COLOR))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # ── Public API ────────────────────────────────────────────────────────

    def load_image(self, path: str) -> bool:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
        qimg   = QImage(img.data, w, h, ch * w, fmt)
        pixmap = QPixmap.fromImage(qimg)

        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()
        return True

    def fit_in_view(self):
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self._zoom_factor = 1.0
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        self.zoom_changed.emit(self._zoom_factor)

    def set_zoom(self, factor: float):
        factor = max(MIN_ZOOM, min(MAX_ZOOM, factor))
        self.scale(factor / self._zoom_factor, factor / self._zoom_factor)
        self._zoom_factor = factor
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_in(self):  self.set_zoom(self._zoom_factor * ZOOM_STEP)
    def zoom_out(self): self.set_zoom(self._zoom_factor / ZOOM_STEP)

    def image_size(self) -> tuple[int, int]:
        if self._pixmap_item:
            p = self._pixmap_item.pixmap()
            return p.width(), p.height()
        return 640, 480

    # ── Events ────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        self.zoom_in() if event.angleDelta().y() > 0 else self.zoom_out()

    def mousePressEvent(self, event: QMouseEvent):
        if (event.button() == Qt.MouseButton.MiddleButton or
                (event.button() == Qt.MouseButton.LeftButton and self._space_held)):
            self._panning   = True
            self._pan_start = event.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y()))
        self.cursor_moved.emit(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = True
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        elif event.key() == Qt.Key.Key_F:
            self.fit_in_view()
        elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            if not self._panning:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().keyReleaseEvent(event)