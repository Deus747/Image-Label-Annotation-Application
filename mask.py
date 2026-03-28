from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QSlider, QComboBox, QFileDialog)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import Qt

from .annotations import Annotation, LabelClass, MaskType, MaskFormat
from .constants import BG_COLOR, PANEL_COLOR, BORDER_COLOR


class MaskGenerator:
    @staticmethod
    def generate(annotations: list[Annotation], classes: list[LabelClass],
                 img_w: int, img_h: int,
                 mask_type: MaskType, mask_fmt: MaskFormat) -> np.ndarray:

        class_map = {c.id: c for c in classes}
        canvas = (np.zeros((img_h, img_w), dtype=np.uint8)
                  if mask_fmt == MaskFormat.GRAYSCALE
                  else np.zeros((img_h, img_w, 3), dtype=np.uint8))

        for idx, ann in enumerate(annotations, start=1):
            pts = MaskGenerator._to_pixels(ann, img_w, img_h)
            if pts is None:
                continue
            cls = class_map.get(ann.class_id)

            if mask_type == MaskType.SEMANTIC:
                if mask_fmt == MaskFormat.GRAYSCALE:
                    # +1 so class 0 is not invisible (0 = background/unlabeled)
                    cv2.fillPoly(canvas, [pts], color=int((ann.class_id + 1) % 256))
                else:
                    color = MaskGenerator._qcolor_bgr(cls.color) if cls else (128, 128, 128)
                    cv2.fillPoly(canvas, [pts], color=color)
            else:  # INSTANCE
                if mask_fmt == MaskFormat.GRAYSCALE:
                    cv2.fillPoly(canvas, [pts], color=int(idx % 256))
                else:
                    cv2.fillPoly(canvas, [pts], color=MaskGenerator._instance_color(idx))

        return canvas

    @staticmethod
    def _to_pixels(ann: Annotation, w: int, h: int) -> Optional[np.ndarray]:
        if ann.ann_type == "bbox":
            cx = ann.x_center * w;  cy = ann.y_center * h
            bw = ann.width * w;     bh = ann.height * h
            return np.array([[cx-bw/2, cy-bh/2], [cx+bw/2, cy-bh/2],
                              [cx+bw/2, cy+bh/2], [cx-bw/2, cy+bh/2]], dtype=np.int32)
        elif ann.ann_type == "polygon" and len(ann.points) >= 3:
            return np.array([[p[0]*w, p[1]*h] for p in ann.points], dtype=np.int32)
        return None

    @staticmethod
    def _qcolor_bgr(c: QColor) -> tuple:
        return (c.blue(), c.green(), c.red())

    @staticmethod
    def _instance_color(idx: int) -> tuple:
        hue = int((idx * 137.508) % 360)
        c   = QColor.fromHsv(hue, 220, 210)
        return (c.blue(), c.green(), c.red())

    @staticmethod
    def to_qpixmap(mask: np.ndarray) -> QPixmap:
        if mask.ndim == 2:
            # True grayscale preview — scale values up so they are visible
            # class_id 0 would be black (background), scale others to be bright
            vis = np.zeros_like(mask)
            unique_vals = np.unique(mask[mask > 0])  # ignore background (0)
            if len(unique_vals) > 0:
                # Map each unique value to a spread across 60-255 range
                for i, val in enumerate(unique_vals):
                    brightness = int(60 + (i / max(len(unique_vals) - 1, 1)) * 195)
                    vis[mask == val] = brightness
            # Convert single channel to RGB for QImage
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            h, w    = vis_rgb.shape[:2]
            qimg    = QImage(vis_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        else:
            vis  = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            h, w = vis.shape[:2]
            qimg = QImage(vis.data, w, h, w * 3, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(qimg)


class MaskPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 220)
        self._last_mask:   Optional[np.ndarray] = None
        self._last_pixmap: Optional[QPixmap]    = None
        self._annotations: list[Annotation]     = []
        self._classes:     list[LabelClass]     = []
        self._img_w = 0;  self._img_h = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        ctrl = QHBoxLayout()
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Semantic", "Instance"])
        self._fmt_combo  = QComboBox()
        self._fmt_combo.addItems(["Color", "Grayscale"])
        self._opacity    = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(20, 100); self._opacity.setValue(70)
        self._opacity.setFixedWidth(70)

        for w in [self._type_combo, self._fmt_combo]:
            w.setStyleSheet(
                f"QComboBox{{background:{PANEL_COLOR};color:#ccc;"
                f"border:1px solid {BORDER_COLOR};border-radius:3px;"
                "padding:1px 4px;font-size:11px;}}")

        ctrl.addWidget(QLabel("Type:")); ctrl.addWidget(self._type_combo)
        ctrl.addWidget(QLabel("Fmt:"));  ctrl.addWidget(self._fmt_combo)
        ctrl.addWidget(QLabel("α:"));    ctrl.addWidget(self._opacity)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(
            f"background:{BG_COLOR};border:1px solid {BORDER_COLOR};border-radius:4px;")
        self._preview.setMinimumHeight(180)
        layout.addWidget(self._preview, stretch=1)

        save_btn = QPushButton("💾  Save mask PNG…")
        save_btn.setStyleSheet(
            "QPushButton{background:#007acc;color:#fff;border:none;"
            "border-radius:4px;padding:5px 12px;font-size:12px;}"
            "QPushButton:hover{background:#005f9e;}")
        save_btn.clicked.connect(self._save_mask)
        layout.addWidget(save_btn)

        self._type_combo.currentIndexChanged.connect(lambda _: self._regenerate())
        self._fmt_combo.currentIndexChanged.connect(lambda _: self._regenerate())
        self._opacity.valueChanged.connect(lambda _: self._refresh_display())

    def update_mask(self, annotations: list[Annotation],
                    classes: list[LabelClass], img_w: int, img_h: int):
        self._annotations = annotations
        self._classes     = classes
        self._img_w       = img_w
        self._img_h       = img_h
        self._regenerate()

    def _regenerate(self):
        if not self._annotations and (self._img_w == 0 or self._img_h == 0):
            self._last_mask = None; self._last_pixmap = None
            self._preview.clear(); return
        try:
            mtype = (MaskType.SEMANTIC if self._type_combo.currentIndex() == 0
                     else MaskType.INSTANCE)
            mfmt  = (MaskFormat.COLOR if self._fmt_combo.currentIndex() == 0
                     else MaskFormat.GRAYSCALE)
            self._last_mask   = MaskGenerator.generate(
                self._annotations, self._classes,
                self._img_w, self._img_h, mtype, mfmt)
            self._last_pixmap = MaskGenerator.to_qpixmap(self._last_mask)
            self._refresh_display()
        except Exception as e:
            print(f"[MaskPreview] {e}")

    def _refresh_display(self):
        if not self._last_pixmap:
            return
        pw = self._preview.width()  or 240
        ph = self._preview.height() or 180
        scaled = self._last_pixmap.scaled(
            pw, ph,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        result = QPixmap(scaled.size())
        result.fill(QColor(BG_COLOR))
        p = QPainter(result)
        p.setOpacity(self._opacity.value() / 100.0)
        p.drawPixmap(0, 0, scaled)
        p.end()
        self._preview.setPixmap(result)

    def _save_mask(self):
        if self._last_mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "mask.png", "PNG files (*.png)")
        if not path:
            return

        mask = self._last_mask

        if mask.ndim == 2:
            # Ask user whether to save raw (true label values) or scaled (visible)
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Grayscale save mode",
                "Save as raw label values (for training)\n"
                "or scaled to 0-255 (for visualization)?\n\n"
                "Yes = Raw label values\nNo = Scaled for visibility",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)

            if reply == QMessageBox.StandardButton.No:
                # Scale for visibility
                unique_vals = np.unique(mask[mask > 0])
                vis = np.zeros_like(mask)
                for i, val in enumerate(unique_vals):
                    brightness = int(60 + (i / max(len(unique_vals) - 1, 1)) * 195)
                    vis[mask == val] = brightness
                cv2.imwrite(path, vis)
            else:
                # Raw — save as-is (class_id or instance index as pixel value)
                cv2.imwrite(path, mask)
        else:
            cv2.imwrite(path, mask)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()