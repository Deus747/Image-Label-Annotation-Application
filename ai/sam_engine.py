from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from .suggestion import Suggestion


# ── Cache directory ───────────────────────────────────────────────────────

WEIGHTS_DIR = Path.home() / ".cache" / "ultralytics"

SAM_MODELS = {
    "SAM2-tiny":  "sam2_t.pt",
    "SAM2-small": "sam2_s.pt",
    "SAM2-base":  "sam2_b.pt",
    "SAM2-large": "sam2_l.pt",
}

SAM_MODEL_SIZES = {
    "SAM2-tiny":   40,
    "SAM2-small":  180,
    "SAM2-base":   310,
    "SAM2-large":  900,
}

DEFAULT_MODEL = "SAM2-small"


# ── Device detection ──────────────────────────────────────────────────────

def _best_device() -> str:
    """
    Detect the best available device using ultralytics-compatible
    device strings.

    Ultralytics device strings:
      "cpu"        — always works
      "0"          — CUDA GPU index 0
      "openvino"   — OpenVINO (ultralytics handles device selection
                     internally, picks GPU if available)

    DO NOT pass "openvino:GPU" or "openvino:CPU" — those are
    OpenVINO runtime strings, not ultralytics strings.
    """
    # 1. OpenVINO — ultralytics has built-in OpenVINO export+inference
    #    Just pass "cpu" and let ultralytics use OpenVINO internally
    #    via its own export pipeline, OR use the correct string below.
    try:
        from openvino.runtime import Core
        devices = Core().available_devices
        if "GPU" in devices or "CPU" in devices:
            # ultralytics SAM with OpenVINO uses this string
            return "cpu"    # SAM runs on CPU via torch; OpenVINO
                            # export is separate — see note below
    except Exception:
        pass

    # 2. CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "0"      # first CUDA GPU
    except Exception:
        pass

    # 3. CPU fallback
    return "cpu"


# ── Weight helpers ────────────────────────────────────────────────────────

def weight_path(model_key: str) -> Path:
    return WEIGHTS_DIR / SAM_MODELS[model_key]


def is_downloaded(model_key: str) -> bool:
    return weight_path(model_key).exists()


def all_model_status() -> dict[str, dict]:
    result = {}
    for key, filename in SAM_MODELS.items():
        p = weight_path(key)
        result[key] = {
            "filename":   filename,
            "size_mb":    SAM_MODEL_SIZES[key],
            "downloaded": p.exists(),
            "path":       str(p),
        }
    return result


# ── Engine ────────────────────────────────────────────────────────────────

class SamEngine:
    """
    SAM 2 inference engine.

    Loading paths
    ─────────────
    load_auto()      — ultralytics auto-downloads to ~/.cache/ultralytics/
                       or loads from cache if already present.
    load_from_path() — load any local .pt file the user browses to.
    load()           — convenience alias for load_auto().

    All predict_*() run on a background thread.
    Callbacks are invoked from that thread — callers must marshal
    back to Qt main thread via QTimer.singleShot(0, ...).
    """

    def __init__(self):
        self._model       = None
        self._model_key   = DEFAULT_MODEL
        self._weight_path = ""
        self._device      = _best_device()
        self._lock        = threading.Lock()
        self._img_path    = ""
        self._img_w       = 0
        self._img_h       = 0
        self.loaded       = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_key(self) -> str:
        return self._model_key

    # =========================================================================
    # Loading
    # =========================================================================

    def load_auto(
        self,
        model_key:   str                             = DEFAULT_MODEL,
        on_ready:    Optional[Callable[[], None]]    = None,
        on_error:    Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        self._model_key = model_key
        self.loaded     = False

        def _prog(msg: str):
            if on_progress:
                on_progress(msg)

        def _run():
            try:
                # Disable network calls before importing ultralytics
                import os
                os.environ["YOLO_OFFLINE"]        = "1"
                os.environ["ULTRALYTICS_OFFLINE"] = "1"

                filename = SAM_MODELS[model_key]
                cached   = is_downloaded(model_key)

                if cached:
                    _prog(f"Found {filename} in cache")
                else:
                    mb = SAM_MODEL_SIZES[model_key]
                    _prog(
                        f"Downloading {filename} (~{mb} MB)…\n"
                        f"First run only — please wait.")

                _prog("Importing ultralytics…")
                try:
                    from ultralytics.utils import SETTINGS
                    SETTINGS.update({"sync": False, "api_key": ""})
                except Exception:
                    pass

                from ultralytics import SAM

                _prog("Loading weights into memory…")
                with self._lock:
                    self._model = SAM(filename)

                _prog(f"Warming up on {self._device}…")
                self._warmup()

                self._weight_path = str(weight_path(model_key))
                self.loaded       = True
                _prog("Model ready.")
                if on_ready:
                    on_ready()

            except Exception as e:
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()


    def load_from_path(
        self,
        path:        str,
        on_ready:    Optional[Callable[[], None]]    = None,
        on_error:    Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        self.loaded = False
        p           = Path(path)
        self._model_key = next(
            (k for k, v in SAM_MODELS.items() if v == p.name),
            "SAM2-custom")
        self._weight_path = path

        def _prog(msg: str):
            if on_progress:
                on_progress(msg)

        def _run():
            try:
                import os
                os.environ["YOLO_OFFLINE"]        = "1"
                os.environ["ULTRALYTICS_OFFLINE"] = "1"

                _prog("Importing ultralytics…")
                try:
                    from ultralytics.utils import SETTINGS
                    SETTINGS.update({"sync": False, "api_key": ""})
                except Exception:
                    pass

                from ultralytics import SAM

                _prog(f"Loading {p.name} into memory…")
                with self._lock:
                    self._model = SAM(path)

                _prog(f"Warming up on {self._device}…")
                self._warmup()

                self.loaded = True
                _prog("Model ready.")
                if on_ready:
                    on_ready()

            except Exception as e:
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def load(
        self,
        model_key:   str                             = DEFAULT_MODEL,
        on_ready:    Optional[Callable[[], None]]    = None,
        on_error:    Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        """Convenience alias for load_auto."""
        self.load_auto(model_key, on_ready, on_error, on_progress)

    # =========================================================================
    # Image context
    # =========================================================================

    def set_image(self, path: str, w: int, h: int):
        self._img_path = path
        self._img_w    = w
        self._img_h    = h

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict_point(
        self,
        points:    list[tuple[int, int]],
        labels:    list[int],
        class_id:  int,
        on_result: Callable[[list[Suggestion]], None],
        on_error:  Callable[[str], None],
    ):
        """
        Point prompt.
        points : [(x, y) …] image pixel space
        labels : 1 = foreground,  0 = background
        """
        if not self._ready(on_error):
            return
        img_path, img_w, img_h = (
            self._img_path, self._img_w, self._img_h)

        def _run():
            try:
                with self._lock:
                    results = self._model.predict(
                        source  = img_path,
                        points  = [points],
                        labels  = [labels],
                        device  = self._device,
                        verbose = False,
                    )
                on_result(self._parse(
                    results, img_w, img_h, class_id, "sam"))
            except Exception as e:
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def predict_box(
        self,
        box:       tuple[int, int, int, int],
        class_id:  int,
        on_result: Callable[[list[Suggestion]], None],
        on_error:  Callable[[str], None],
    ):
        """Box prompt. box = (x1, y1, x2, y2) image pixel space."""
        if not self._ready(on_error):
            return
        img_path, img_w, img_h = (
            self._img_path, self._img_w, self._img_h)

        def _run():
            try:
                with self._lock:
                    results = self._model.predict(
                        source  = img_path,
                        bboxes  = [list(box)],
                        device  = self._device,
                        verbose = False,
                    )
                on_result(self._parse(
                    results, img_w, img_h, class_id, "sam"))
            except Exception as e:
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def predict_box_and_points(
        self,
        box:       tuple[int, int, int, int],
        points:    list[tuple[int, int]],
        labels:    list[int],
        class_id:  int,
        on_result: Callable[[list[Suggestion]], None],
        on_error:  Callable[[str], None],
    ):
        """Combined box + point prompt."""
        if not self._ready(on_error):
            return
        img_path, img_w, img_h = (
            self._img_path, self._img_w, self._img_h)

        def _run():
            try:
                with self._lock:
                    results = self._model.predict(
                        source  = img_path,
                        bboxes  = [list(box)],
                        points  = [points],
                        labels  = [labels],
                        device  = self._device,
                        verbose = False,
                    )
                on_result(self._parse(
                    results, img_w, img_h, class_id, "sam"))
            except Exception as e:
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _ready(self, on_error: Callable) -> bool:
        if not self.loaded or self._model is None:
            on_error(
                "SAM model not loaded.\n"
                "Click 'Load / download' or 'Browse .pt…' "
                "in the AI Tools panel.")
            return False
        if not self._img_path:
            on_error("No image loaded.")
            return False
        return True

    def _warmup(self):
        """
        Tiny dummy inference so the first real prediction
        is not slow due to lazy kernel compilation.
        Non-fatal if it fails.
        """
        try:
            import numpy as np
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model.predict(
                source  = dummy,
                points  = [[[32, 32]]],
                labels  = [[1]],
                device  = self._device,
                verbose = False,
            )
        except Exception:
            pass

    @staticmethod
    def _parse(
        results,
        img_w:    int,
        img_h:    int,
        class_id: int,
        source:   str,
    ) -> list[Suggestion]:
        suggestions = []

        for result in results:
            if result.masks is None:
                continue

            for raw_mask in result.masks.data.cpu().numpy():

                # Resize to image dimensions
                mask_u8 = (raw_mask > 0.5).astype(np.uint8) * 255
                mask_u8 = cv2.resize(
                    mask_u8, (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST)

                # Smooth edges
                kernel  = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (5, 5))
                mask_u8 = cv2.morphologyEx(
                    mask_u8, cv2.MORPH_CLOSE, kernel)
                blurred = cv2.GaussianBlur(mask_u8, (11, 11), 0)
                _, mask_u8 = cv2.threshold(
                    blurred, 127, 255, cv2.THRESH_BINARY)

                # Contour → polygon
                contours, _ = cv2.findContours(
                    mask_u8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_TC89_KCOS)

                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) < 100:
                    continue

                eps    = max(1.5, cv2.arcLength(cnt, True) * 0.005)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) < 3:
                    continue

                pts = [
                    [float(p[0][0]) / img_w,
                     float(p[0][1]) / img_h]
                    for p in approx
                ]

                ys, xs = np.where(mask_u8 > 0)
                if len(xs) == 0:
                    continue

                x1n = xs.min() / img_w;  x2n = xs.max() / img_w
                y1n = ys.min() / img_h;  y2n = ys.max() / img_h

                suggestions.append(Suggestion(
                    ann_type = "polygon",
                    source   = source,
                    score    = 1.0,
                    class_id = class_id,
                    x_center = (x1n + x2n) / 2,
                    y_center = (y1n + y2n) / 2,
                    width    = x2n - x1n,
                    height   = y2n - y1n,
                    points   = pts,
                    mask     = mask_u8,
                ))

        return suggestions