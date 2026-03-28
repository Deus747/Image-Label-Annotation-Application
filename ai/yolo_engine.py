from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .suggestion import Suggestion


class YoloEngine:
    """
    YOLO inference engine for auto-detection suggestions.
    Supports both detection-only and segmentation models.
    """

    def __init__(self):
        self._model       = None
        self._model_path  = ""
        self._lock        = threading.Lock()
        self.loaded       = False
        self.class_names: list[str] = []

    def load(
        self,
        model_path: str,
        on_ready:   Optional[Callable[[], None]]    = None,
        on_error:   Optional[Callable[[str], None]] = None,
    ):
        self._model_path = model_path
        self.loaded      = False

        def _run():
            try:
                from ultralytics import YOLO
                with self._lock:
                    self._model      = YOLO(model_path)
                    self.class_names = list(
                        self._model.names.values())
                self.loaded = True
                if on_ready:
                    on_ready()
            except Exception as e:
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def predict(
        self,
        img_path:   str,
        img_w:      int,
        img_h:      int,
        conf:       float,
        class_id:   int,
        on_result:  Callable[[list[Suggestion]], None],
        on_error:   Callable[[str], None],
    ):
        if not self.loaded or self._model is None:
            on_error("YOLO model not loaded yet.")
            return

        def _run():
            try:
                with self._lock:
                    results = self._model.predict(
                        source  = img_path,
                        conf    = conf,
                        verbose = False,
                    )
                on_result(self._parse(
                    results, img_w, img_h, class_id))
            except Exception as e:
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _parse(
        results,
        img_w:    int,
        img_h:    int,
        class_id: int,
    ) -> list[Suggestion]:
        import cv2
        suggestions = []

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue

            boxes_data = boxes.xyxyn.cpu().numpy()
            confs_data = boxes.conf.cpu().numpy()
            cls_data   = boxes.cls.cpu().numpy()

            for i, (box, conf, cls) in enumerate(
                    zip(boxes_data, confs_data, cls_data)):
                x1n, y1n, x2n, y2n = box
                cx = (x1n + x2n) / 2
                cy = (y1n + y2n) / 2
                bw = x2n - x1n
                bh = y2n - y1n

                pts  = []
                mask = None

                if masks is not None and i < len(masks.data):
                    raw = masks.data[i].cpu().numpy()
                    m   = (raw > 0.5).astype(np.uint8) * 255
                    m   = cv2.resize(
                        m, (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(
                        m, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_TC89_KCOS)
                    if contours:
                        cnt    = max(contours,
                                     key=cv2.contourArea)
                        eps    = max(
                            1.5,
                            cv2.arcLength(cnt, True) * 0.005)
                        approx = cv2.approxPolyDP(
                            cnt, eps, True)
                        pts = [
                            [float(p[0][0]) / img_w,
                             float(p[0][1]) / img_h]
                            for p in approx]
                        mask = m

                ann_type = "polygon" if pts else "bbox"
                suggestions.append(Suggestion(
                    ann_type = ann_type,
                    source   = "yolo",
                    score    = float(conf),
                    class_id = int(cls),
                    x_center = float(cx),
                    y_center = float(cy),
                    width    = float(bw),
                    height   = float(bh),
                    points   = pts,
                    mask     = mask,
                ))

        return suggestions