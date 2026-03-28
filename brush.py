from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QSlider, QPushButton)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PyQt6.QtCore import Qt, pyqtSignal

from .constants import (PANEL_COLOR, BORDER_COLOR,
                         BRUSH_MIN_SIZE, BRUSH_MAX_SIZE,
                         BRUSH_DEFAULT)


class BrushSettingsWidget(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.brush_size = BRUSH_DEFAULT
        self.eraser     = False
        self.opacity    = 180

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Brush size ────────────────────────────────────────────
        layout.addWidget(self._label("Brush size"))
        self._size_slider = self._slider(
            BRUSH_MIN_SIZE, BRUSH_MAX_SIZE, self.brush_size)
        self._size_lbl = QLabel(str(self.brush_size))
        self._size_lbl.setStyleSheet(
            "color:#aaa; font-size:11px; min-width:28px;")
        row = QHBoxLayout()
        row.addWidget(self._size_slider)
        row.addWidget(self._size_lbl)
        layout.addLayout(row)
        self._size_slider.valueChanged.connect(self._on_size)

        # ── Opacity ───────────────────────────────────────────────
        layout.addWidget(self._label("Opacity"))
        self._opacity_slider = self._slider(20, 255, self.opacity)
        self._opacity_lbl = QLabel(str(self.opacity))
        self._opacity_lbl.setStyleSheet(
            "color:#aaa; font-size:11px; min-width:28px;")
        row2 = QHBoxLayout()
        row2.addWidget(self._opacity_slider)
        row2.addWidget(self._opacity_lbl)
        layout.addLayout(row2)
        self._opacity_slider.valueChanged.connect(self._on_opacity)

        # ── Eraser toggle ─────────────────────────────────────────
        self._erase_btn = QPushButton("⌫  Eraser  [E]")
        self._erase_btn.setCheckable(True)
        self._erase_btn.setStyleSheet("""
            QPushButton {
                background:#3c3c3c; color:#aaa; border:none;
                border-radius:4px; padding:5px; font-size:12px;
            }
            QPushButton:checked { background:#c0392b; color:#fff; }
            QPushButton:hover:!checked { background:#505050; }
        """)
        self._erase_btn.toggled.connect(self._on_eraser)
        layout.addWidget(self._erase_btn)

        # ── Clear mask ────────────────────────────────────────────
        self._clear_btn = QPushButton("🗑  Clear brush mask")
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background:#3c3c3c; color:#aaa; border:none;
                border-radius:4px; padding:5px; font-size:12px;
            }
            QPushButton:hover { background:#e67e22; color:#fff; }
        """)
        layout.addWidget(self._clear_btn)

        # ── Commit hint ───────────────────────────────────────────
        hint = QLabel("Enter = commit to polygon")
        hint.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic;")
        layout.addWidget(hint)

        layout.addStretch()

    def _on_size(self, v: int):
        self.brush_size = v
        self._size_lbl.setText(str(v))
        self.settings_changed.emit()

    def _on_opacity(self, v: int):
        self.opacity = v
        self._opacity_lbl.setText(str(v))
        self.settings_changed.emit()

    def _on_eraser(self, checked: bool):
        self.eraser = checked
        self.settings_changed.emit()

    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#cccccc; font-size:12px;")
        return lbl

    @staticmethod
    def _slider(lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setStyleSheet("""
            QSlider::groove:horizontal {
                height:4px; background:#3c3c3c; border-radius:2px;
            }
            QSlider::handle:horizontal {
                width:12px; height:12px; margin:-4px 0;
                background:#007acc; border-radius:6px;
            }
        """)
        return s


class BrushOverlay:
    """
    Per-image float32 mask painted by the brush tool.
    Also used for mask editing (erasing from SAM/polygon masks).
    """

    def __init__(self):
        self._masks: dict[str, np.ndarray] = {}
        self._current_path: str = ""
        self._w = 0
        self._h = 0

    def set_image(self, path: str, w: int, h: int):
        self._current_path = path
        self._w = w
        self._h = h
        if path not in self._masks:
            self._masks[path] = np.zeros(
                (h, w), dtype=np.float32)

    def current_mask(self) -> Optional[np.ndarray]:
        return self._masks.get(self._current_path)

    def paint(self, x: int, y: int,
              radius: int, erase: bool = False):
        mask = self.current_mask()
        if mask is None:
            return
        # Clamp radius to at least 1px
        r     = max(1, radius)
        value = 0.0 if erase else 1.0
        cv2.circle(mask, (x, y), r, value, -1)

    def paint_stroke(self, x0: int, y0: int,
                     x1: int, y1: int,
                     radius: int, erase: bool = False):
        mask = self.current_mask()
        if mask is None:
            return
        r     = max(1, radius)
        value = 0.0 if erase else 1.0
        cv2.line(mask, (x0, y0), (x1, y1), value, r * 2)
        cv2.circle(mask, (x1, y1), r, value, -1)

    def load_from_polygon(
            self,
            points: list[list[float]],
            img_w: int, img_h: int):
        """
        Rasterize an existing polygon annotation into the brush
        mask so the user can paint-erase parts of it.
        """
        mask = self.current_mask()
        if mask is None:
            return
        mask[:] = 0.0
        pts = np.array(
            [[int(p[0]*img_w), int(p[1]*img_h)]
             for p in points],
            dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    def load_from_numpy_mask(self, src: np.ndarray):
        """
        Load an existing uint8 binary mask (0/255) directly
        into the current brush overlay for editing.
        """
        mask = self.current_mask()
        if mask is None or src is None:
            return
        resized = cv2.resize(
            src, (self._w, self._h),
            interpolation=cv2.INTER_NEAREST)
        mask[:] = (resized > 127).astype(np.float32)

    def clear(self):
        mask = self.current_mask()
        if mask is not None:
            mask[:] = 0.0

    def to_polygons(self) -> list[list[list[float]]]:
        mask = self.current_mask()
        if mask is None or self._w == 0 or self._h == 0:
            return []
        binary = (mask > 0.5).astype(np.uint8) * 255

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel_open)

        blurred = cv2.GaussianBlur(binary, (21, 21), 0)
        _, binary = cv2.threshold(
            blurred, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 30:
                continue
            smoothed = self._smooth_contour(cnt, smoothing=8)
            if smoothed is None or len(smoothed) < 3:
                continue
            eps    = max(1.0,
                         cv2.arcLength(smoothed, True) * 0.003)
            approx = cv2.approxPolyDP(smoothed, eps, True)
            if len(approx) < 3:
                continue
            pts = [[float(p[0][0]) / self._w,
                    float(p[0][1]) / self._h]
                   for p in approx]
            polygons.append(pts)
        return polygons

    @staticmethod
    def _smooth_contour(
            contour: np.ndarray,
            smoothing: int = 8) -> np.ndarray | None:
        pts = contour[:, 0, :].astype(np.float32)
        n   = len(pts)
        if n < 6:
            return contour
        diffs   = np.diff(pts, axis=0, prepend=pts[-1:])
        lengths = np.linalg.norm(diffs, axis=1)
        cumlen  = np.cumsum(lengths)
        total   = cumlen[-1]
        if total < 1:
            return contour
        n_samples = int(np.clip(total / 3.0, 24, 512))
        sample_at = np.linspace(
            0, total, n_samples, endpoint=False)
        xs = np.interp(sample_at, cumlen, pts[:, 0])
        ys = np.interp(sample_at, cumlen, pts[:, 1])
        hw     = smoothing * 3
        kernel = np.exp(
            -0.5 * (np.arange(-hw, hw+1) / smoothing) ** 2)
        kernel /= kernel.sum()
        pad      = hw
        xs_pad   = np.concatenate([xs[-pad:], xs, xs[:pad]])
        ys_pad   = np.concatenate([ys[-pad:], ys, ys[:pad]])
        xs_s     = np.convolve(
            xs_pad, kernel, mode='valid')[:n_samples]
        ys_s     = np.convolve(
            ys_pad, kernel, mode='valid')[:n_samples]
        smoothed = np.stack(
            [xs_s, ys_s], axis=1).astype(np.int32)
        return smoothed.reshape(-1, 1, 2)

    def to_qpixmap(self, color: QColor,
                   opacity: int,
                   view_w: int,
                   view_h: int) -> QPixmap:
        mask = self.current_mask()
        if mask is None:
            return QPixmap(view_w, view_h)
        display = cv2.resize(
            mask, (view_w, view_h),
            interpolation=cv2.INTER_NEAREST)
        rgba    = np.zeros(
            (view_h, view_w, 4), dtype=np.uint8)
        painted = display > 0.5
        rgba[painted, 0] = color.red()
        rgba[painted, 1] = color.green()
        rgba[painted, 2] = color.blue()
        rgba[painted, 3] = opacity
        qimg = QImage(
            rgba.data, view_w, view_h,
            view_w * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def serialize(self) -> dict:
        """Legacy RLE — kept for backward compat. Use session.py PNG path."""
        out = {}
        for path, mask in self._masks.items():
            if not np.any(mask > 0.5):
                continue
            binary = (mask > 0.5).astype(np.uint8).flatten().tolist()
            out[path] = {
                "w": mask.shape[1],
                "h": mask.shape[0],
                "rle": binary,
            }
        return out

    def deserialize(self, data: dict):
        self._masks = {}
        for path, d in data.items():
            w, h = d["w"], d["h"]
            # Support both new PNG format and old RLE format
            if "png" in d:
                import base64
                import cv2 as _cv2
                png_bytes = base64.b64decode(d["png"])
                arr = np.frombuffer(png_bytes, dtype=np.uint8)
                img = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self._masks[path] = (
                        img.astype(np.float32) / 255.0)
            elif "rle" in d:
                flat = np.array(d["rle"], dtype=np.float32)
                self._masks[path] = flat.reshape((h, w))