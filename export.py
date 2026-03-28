from __future__ import annotations
import json
from pathlib import Path

import cv2

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                              QFileDialog, QLabel, QMessageBox)

from .annotations import AnnotationStore
from .mask import MaskPreviewWidget, MaskGenerator
from .annotations import MaskType, MaskFormat


class ExportPanel(QWidget):
    def __init__(self, store: AnnotationStore,
                 mask_preview: MaskPreviewWidget, parent=None):
        super().__init__(parent)
        self._store   = store
        self._preview = mask_preview

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6); layout.setSpacing(6)

        title = QLabel("Export")
        title.setStyleSheet("color:#cccccc;font-weight:500;font-size:13px;")
        layout.addWidget(title)

        for label, slot in [
            ("YOLO .txt",        self._export_yolo),
            ("COCO JSON",        self._export_coco),
            ("Mask PNG (cur.)",  self._export_mask_current),
            ("Mask PNG (all)",   self._export_mask_all),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet("""
                QPushButton{background:#3c3c3c;color:#ccc;border:none;
                            border-radius:4px;padding:5px;font-size:12px;}
                QPushButton:hover{background:#007acc;color:#fff;}
            """)
            btn.clicked.connect(slot)
            layout.addWidget(btn)
        layout.addStretch()

    def _export_yolo(self):
        folder = QFileDialog.getExistingDirectory(self, "Export YOLO labels")
        if not folder: return
        for img_path, anns in self._store._data.items():
            stem  = Path(img_path).stem
            lines = [a.to_yolo() for a in anns]
            (Path(folder) / f"{stem}.txt").write_text("\n".join(lines))

    def _export_coco(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export COCO JSON", "annotations.json", "JSON (*.json)")
        if not path: return
        images, annotations, ann_id = [], [], 1
        for img_id, (img_path, anns) in enumerate(
                self._store._data.items(), 1):
            img_cv = cv2.imread(img_path)
            h, w   = img_cv.shape[:2] if img_cv is not None else (0, 0)
            images.append({"id": img_id, "file_name": Path(img_path).name,
                            "width": w, "height": h})
            for ann in anns:
                seg, bbox = [], [0, 0, 0, 0]
                if ann.ann_type == "polygon":
                    seg  = [[c for p in ann.points for c in [p[0]*w, p[1]*h]]]
                    xs   = [p[0]*w for p in ann.points]
                    ys   = [p[1]*h for p in ann.points]
                    bbox = [min(xs), min(ys),
                            max(xs)-min(xs), max(ys)-min(ys)]
                elif ann.ann_type == "bbox":
                    cx = ann.x_center*w; cy = ann.y_center*h
                    bw = ann.width*w;    bh = ann.height*h
                    bbox = [cx-bw/2, cy-bh/2, bw, bh]
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": ann.class_id + 1,
                    "segmentation": seg, "bbox": bbox,
                    "area": bbox[2]*bbox[3], "iscrowd": 0})
                ann_id += 1
        Path(path).write_text(
            json.dumps({"images": images, "annotations": annotations},
                       indent=2))

    def _export_mask_current(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "mask.png", "PNG (*.png)")
        if path and self._preview._last_mask is not None:
            cv2.imwrite(path, self._preview._last_mask)

    def _export_mask_all(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Export all masks to folder")
        if not folder: return
        classes = self._preview._classes
        mtype   = (MaskType.SEMANTIC
                   if self._preview._type_combo.currentIndex() == 0
                   else MaskType.INSTANCE)
        mfmt    = (MaskFormat.COLOR
                   if self._preview._fmt_combo.currentIndex() == 0
                   else MaskFormat.GRAYSCALE)
        for img_path, anns in self._store._data.items():
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            mask = MaskGenerator.generate(anns, classes, w, h, mtype, mfmt)
            stem = Path(img_path).stem
            cv2.imwrite(str(Path(folder) / f"{stem}_mask.png"), mask)
        QMessageBox.information(self, "Export",
                                f"Masks saved to {folder}")