from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QFileDialog,
    QButtonGroup, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal

from .sam_engine  import (SamEngine, SAM_MODELS, DEFAULT_MODEL,
                           all_model_status, is_downloaded,
                           SAM_MODEL_SIZES)
from .yolo_engine import YoloEngine
from ..constants  import PANEL_COLOR, BORDER_COLOR


class _LoadSignals(QObject):
    """
    Thread-safe signals for background load callbacks.
    QueuedConnection guarantees delivery on the main thread
    regardless of which thread emits them.
    """
    progress = pyqtSignal(str)
    ready    = pyqtSignal()
    error    = pyqtSignal(str)


class AiToolbar(QWidget):

    sam_mode_changed   = pyqtSignal(str)
    conf_changed       = pyqtSignal(float)
    yolo_run_requested = pyqtSignal()
    accept_all         = pyqtSignal()
    reject_all         = pyqtSignal()
    status_message     = pyqtSignal(str)

    _SPINNER = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(
        self,
        sam_engine:  SamEngine,
        yolo_engine: YoloEngine,
        parent=None,
    ):
        super().__init__(parent)
        self._sam  = sam_engine
        self._yolo = yolo_engine
        self._yolo_pending_path = ""

        # ── Thread-safe signal bridges ────────────────────────────────────
        self._sam_signals  = _LoadSignals()
        self._yolo_signals = _LoadSignals()

        self._sam_signals.progress.connect(
            self._on_sam_progress,
            Qt.ConnectionType.QueuedConnection)
        self._sam_signals.ready.connect(
            self._on_sam_ready,
            Qt.ConnectionType.QueuedConnection)
        self._sam_signals.error.connect(
            self._on_sam_error,
            Qt.ConnectionType.QueuedConnection)

        self._yolo_signals.ready.connect(
            self._on_yolo_ready,
            Qt.ConnectionType.QueuedConnection)
        self._yolo_signals.error.connect(
            self._on_yolo_error,
            Qt.ConnectionType.QueuedConnection)

        # ── Spinner ───────────────────────────────────────────────────────
        self._spinner_idx   = 0
        self._spinner_base  = ""
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        self.setMinimumWidth(240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── SAM 2 ─────────────────────────────────────────────────────────
        layout.addWidget(
            self._section_label("SAM 2  —  segment anything"))

        row_model = QHBoxLayout()
        lbl = QLabel("Model:")
        lbl.setStyleSheet("color:#ccc; font-size:12px;")
        self._sam_combo = QComboBox()
        self._sam_combo.setStyleSheet(self._combo_ss())
        self._sam_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed)
        self._refresh_model_combo()
        row_model.addWidget(lbl)
        row_model.addWidget(self._sam_combo, stretch=1)
        layout.addLayout(row_model)

        self._device_lbl = QLabel(
            f"Device:  {self._device_display(sam_engine.device)}")
        self._device_lbl.setStyleSheet(
            "color:#555; font-size:11px;")
        layout.addWidget(self._device_lbl)

        self._sam_status = QLabel("Not loaded")
        self._sam_status.setStyleSheet(
            "color:#888; font-size:11px; font-style:italic;")
        self._sam_status.setWordWrap(True)
        self._sam_status.setMinimumHeight(32)
        layout.addWidget(self._sam_status)

        self._sam_bar = QProgressBar()
        self._sam_bar.setRange(0, 0)
        self._sam_bar.setFixedHeight(4)
        self._sam_bar.setTextVisible(False)
        self._sam_bar.setStyleSheet(
            "QProgressBar {"
            "  background:#2a2a2a; border:none; border-radius:2px;"
            "}"
            "QProgressBar::chunk {"
            "  background:#007acc; border-radius:2px;"
            "}"
        )
        self._sam_bar.hide()
        layout.addWidget(self._sam_bar)

        row_load = QHBoxLayout()
        self._sam_load_btn = QPushButton("⬇  Load / download")
        self._sam_load_btn.setStyleSheet(self._btn_ss("#1e6a99"))
        self._sam_load_btn.setToolTip(
            "Auto-download via ultralytics on first run.\n"
            "Loads from cache instantly on subsequent runs.")
        self._sam_load_btn.clicked.connect(self._load_sam_auto)

        self._sam_browse_btn = QPushButton("📁  Browse .pt…")
        self._sam_browse_btn.setStyleSheet(self._btn_ss("#3c5a3c"))
        self._sam_browse_btn.setToolTip(
            "Load a SAM 2 .pt file already on disk.")
        self._sam_browse_btn.clicked.connect(self._browse_sam)

        row_load.addWidget(self._sam_load_btn)
        row_load.addWidget(self._sam_browse_btn)
        layout.addLayout(row_load)

        row_mode = QHBoxLayout()
        self._sam_btn_group = QButtonGroup(self)
        self._sam_btn_group.setExclusive(True)

        self._sam_off_btn   = self._toggle_btn("Off")
        self._sam_point_btn = self._toggle_btn("● Point", "#1e6a99")
        self._sam_box_btn   = self._toggle_btn("⬜ Box",   "#1e6a99")
        self._sam_off_btn.setChecked(True)

        for btn in [self._sam_off_btn,
                    self._sam_point_btn,
                    self._sam_box_btn]:
            self._sam_btn_group.addButton(btn)
            row_mode.addWidget(btn)

        self._sam_off_btn.clicked.connect(
            lambda: self.sam_mode_changed.emit("off"))
        self._sam_point_btn.clicked.connect(
            lambda: self.sam_mode_changed.emit("point"))
        self._sam_box_btn.clicked.connect(
            lambda: self.sam_mode_changed.emit("box"))

        self._sam_point_btn.setEnabled(False)
        self._sam_box_btn.setEnabled(False)
        layout.addLayout(row_mode)

        self._sam_hint = QLabel(
            "Load a SAM 2 model above, "
            "then choose a prompt mode.")
        self._sam_hint.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic;")
        self._sam_hint.setWordWrap(True)
        layout.addWidget(self._sam_hint)

        layout.addWidget(self._divider())

        # ── YOLO ──────────────────────────────────────────────────────────
        layout.addWidget(
            self._section_label("YOLO  —  auto detection"))

        row_yolo = QHBoxLayout()
        self._yolo_lbl = QLabel("No model loaded")
        self._yolo_lbl.setStyleSheet(
            "color:#666; font-size:11px;")
        self._yolo_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred)
        yolo_browse = QPushButton("Browse .pt…")
        yolo_browse.setStyleSheet(self._btn_ss("#3c5a3c"))
        yolo_browse.clicked.connect(self._browse_yolo)
        row_yolo.addWidget(self._yolo_lbl, stretch=1)
        row_yolo.addWidget(yolo_browse)
        layout.addLayout(row_yolo)

        row_conf = QHBoxLayout()
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setStyleSheet("color:#ccc; font-size:12px;")
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(25)
        self._conf_slider.setStyleSheet(self._slider_ss())
        self._conf_val = QLabel("0.25")
        self._conf_val.setStyleSheet(
            "color:#aaa; font-size:11px; min-width:32px;")
        self._conf_slider.valueChanged.connect(self._on_conf)
        row_conf.addWidget(conf_lbl)
        row_conf.addWidget(self._conf_slider, stretch=1)
        row_conf.addWidget(self._conf_val)
        layout.addLayout(row_conf)

        self._yolo_run_btn = QPushButton("▶  Run YOLO on image")
        self._yolo_run_btn.setStyleSheet(self._btn_ss("#2d6a2d"))
        self._yolo_run_btn.setEnabled(False)
        self._yolo_run_btn.clicked.connect(
            self.yolo_run_requested.emit)
        layout.addWidget(self._yolo_run_btn)

        self._yolo_bar = QProgressBar()
        self._yolo_bar.setRange(0, 0)
        self._yolo_bar.setFixedHeight(4)
        self._yolo_bar.setTextVisible(False)
        self._yolo_bar.setStyleSheet(
            "QProgressBar {"
            "  background:#2a2a2a; border:none; border-radius:2px;"
            "}"
            "QProgressBar::chunk {"
            "  background:#2ecc71; border-radius:2px;"
            "}"
        )
        self._yolo_bar.hide()
        layout.addWidget(self._yolo_bar)

        layout.addWidget(self._divider())

        # ── Suggestions ───────────────────────────────────────────────────
        layout.addWidget(self._section_label("Suggestions"))

        self._pending_lbl = QLabel("No pending suggestions")
        self._pending_lbl.setStyleSheet(
            "color:#666; font-size:11px;")
        self._pending_lbl.setWordWrap(True)
        layout.addWidget(self._pending_lbl)

        row_sug = QHBoxLayout()
        acc_btn = QPushButton("✔  Accept all")
        acc_btn.setStyleSheet(self._btn_ss("#2d6a2d"))
        acc_btn.clicked.connect(self.accept_all.emit)
        rej_btn = QPushButton("✕  Reject all")
        rej_btn.setStyleSheet(self._btn_ss("#7b2020"))
        rej_btn.clicked.connect(self.reject_all.emit)
        row_sug.addWidget(acc_btn)
        row_sug.addWidget(rej_btn)
        layout.addLayout(row_sug)

        layout.addStretch()

    # =========================================================================
    # Public API
    # =========================================================================

    def set_pending_count(self, n: int):
        if n == 0:
            self._pending_lbl.setText("No pending suggestions")
        elif n == 1:
            self._pending_lbl.setText(
                "1 pending suggestion\n"
                "Enter = accept  |  Esc = reject")
        else:
            self._pending_lbl.setText(
                f"{n} pending suggestions\n"
                "Enter = accept all  |  Esc = reject all")

    def set_sam_loading(self, loading: bool):
        self._sam_bar.setVisible(loading)
        self._sam_load_btn.setEnabled(not loading)
        self._sam_browse_btn.setEnabled(not loading)
        self._sam_load_btn.setText(
            "⬇  Load / download"
            if not loading else "Loading…")
        if loading:
            self._spinner_idx = 0
            self._spinner_timer.start()
        else:
            self._spinner_timer.stop()

    def set_sam_ready(self, ready: bool):
        self.set_sam_loading(False)
        self._sam_point_btn.setEnabled(ready)
        self._sam_box_btn.setEnabled(ready)
        self._sam_hint.setText(
            "Click = foreground  |  Shift+click = background\n"
            "Enter = accept  |  Esc = reject"
            if ready else
            "Load a SAM 2 model above, "
            "then choose a prompt mode.")

    def set_yolo_loading(self, loading: bool):
        self._yolo_bar.setVisible(loading)
        self._yolo_run_btn.setEnabled(not loading)

    def set_yolo_ready(self, ready: bool, path: str = ""):
        self.set_yolo_loading(False)
        self._yolo_run_btn.setEnabled(ready)
        if path:
            self._yolo_lbl.setText(Path(path).name)

    # =========================================================================
    # Spinner
    # =========================================================================

    def _tick_spinner(self):
        frame = self._SPINNER[
            self._spinner_idx % len(self._SPINNER)]
        self._spinner_idx += 1
        self._sam_status.setText(
            f"{frame}  {self._spinner_base}")

    # =========================================================================
    # SAM loading
    # =========================================================================

    def _refresh_model_combo(self):
        current = (self._sam_combo.currentData()
                   if self._sam_combo.count() > 0
                   else DEFAULT_MODEL)
        self._sam_combo.blockSignals(True)
        self._sam_combo.clear()
        for key, info in all_model_status().items():
            icon  = "✔" if info["downloaded"] else "⬇"
            label = (
                f"{icon}  {key}  ({info['size_mb']} MB"
                + ("  — cached" if info["downloaded"] else "")
                + ")")
            self._sam_combo.addItem(label, userData=key)
        for i in range(self._sam_combo.count()):
            if self._sam_combo.itemData(i) == current:
                self._sam_combo.setCurrentIndex(i)
                break
        self._sam_combo.blockSignals(False)

    def _load_sam_auto(self):
        key = self._sam_combo.currentData()
        if not key:
            return
        self.set_sam_loading(True)
        if is_downloaded(key):
            self._spinner_base = (
                f"Loading {SAM_MODELS[key]} from cache…")
        else:
            mb = SAM_MODEL_SIZES[key]
            self._spinner_base = (
                f"Downloading {SAM_MODELS[key]} (~{mb} MB)…")
        self._sam.load_auto(
            model_key   = key,
            on_ready    = self._sam_signals.ready.emit,
            on_error    = self._sam_signals.error.emit,
            on_progress = self._sam_signals.progress.emit,
        )

    def _browse_sam(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load SAM 2 weights",
            str(Path.home()),
            "PyTorch weights (*.pt *.pth);;All files (*)")
        if not path:
            return
        self.set_sam_loading(True)
        self._spinner_base = f"Loading {Path(path).name}…"
        self._sam.load_from_path(
            path        = path,
            on_ready    = self._sam_signals.ready.emit,
            on_error    = self._sam_signals.error.emit,
            on_progress = self._sam_signals.progress.emit,
        )

    # ── SAM callbacks — called via QueuedConnection (main thread) ─────────

    def _on_sam_progress(self, msg: str):
        self._spinner_base = msg

    def _on_sam_ready(self):
        self.set_sam_ready(True)
        self._sam_status.setText(
            f"✔  {self._sam.model_key}  on  {self._sam.device}")
        self._refresh_model_combo()
        self.status_message.emit(
            f"SAM 2 ready  —  "
            f"{self._sam.model_key}  on  {self._sam.device}")

    def _on_sam_error(self, msg: str):
        self.set_sam_loading(False)
        self._sam_status.setText(f"✗  {msg}")
        self.status_message.emit(f"SAM error: {msg}")

    # =========================================================================
    # YOLO loading
    # =========================================================================

    def _browse_yolo(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load YOLO model",
            str(Path.home()),
            "PyTorch weights (*.pt *.pth);;All files (*)")
        if not path:
            return
        self._yolo_pending_path = path
        self.set_yolo_loading(True)
        self._yolo_lbl.setText(f"Loading  {Path(path).name}…")
        self._yolo.load(
            model_path = path,
            on_ready   = self._yolo_signals.ready.emit,
            on_error   = self._yolo_signals.error.emit,
        )

    # ── YOLO callbacks — called via QueuedConnection (main thread) ────────

    def _on_yolo_ready(self):
        self.set_yolo_ready(True, self._yolo_pending_path)

    def _on_yolo_error(self, msg: str):
        self.set_yolo_loading(False)
        self.status_message.emit(f"YOLO error: {msg}")

    # =========================================================================
    # Conf slider
    # =========================================================================

    def _on_conf(self, v: int):
        conf = v / 100.0
        self._conf_val.setText(f"{conf:.2f}")
        self.conf_changed.emit(conf)

    # =========================================================================
    # Style helpers
    # =========================================================================

    @staticmethod
    def _device_display(device: str) -> str:
        if device == "cpu":
            try:
                from openvino.runtime import Core
                devs = Core().available_devices
                if "GPU" in devs:
                    return "CPU  (Intel Xe via OpenVINO available)"
                return "CPU"
            except Exception:
                return "CPU"
        if device in ("0", "cuda"):
            return "CUDA GPU"
        return device

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color:#aaa; font-size:12px; font-weight:500;")
        return lbl

    @staticmethod
    def _divider() -> QWidget:
        w = QWidget()
        w.setFixedHeight(1)
        w.setStyleSheet("background:" + BORDER_COLOR + ";")
        return w

    @staticmethod
    def _toggle_btn(
            text: str,
            active_bg: str = "#505050") -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setStyleSheet(
            "QPushButton {"
            "  background:#3c3c3c; color:#aaa; border:none;"
            "  border-radius:4px; padding:4px 8px;"
            "  font-size:12px;"
            "}"
            "QPushButton:checked {"
            "  background:" + active_bg + "; color:#fff;"
            "}"
            "QPushButton:hover:!checked { background:#505050; }"
            "QPushButton:disabled {"
            "  background:#2a2a2a; color:#444;"
            "}"
        )
        return btn

    @staticmethod
    def _btn_ss(bg: str) -> str:
        return (
            "QPushButton {"
            "  background:" + bg + "; color:#fff;"
            "  border:none; border-radius:4px;"
            "  padding:4px 12px; font-size:12px;"
            "}"
            "QPushButton:hover { background:" + bg + "cc; }"
            "QPushButton:disabled {"
            "  background:#2a2a2a; color:#555;"
            "}"
        )

    @staticmethod
    def _combo_ss() -> str:
        return (
            "QComboBox {"
            "  background:" + PANEL_COLOR + "; color:#ccc;"
            "  border:1px solid " + BORDER_COLOR + ";"
            "  border-radius:3px; padding:2px 6px;"
            "  font-size:12px;"
            "}"
            "QComboBox QAbstractItemView {"
            "  background:" + PANEL_COLOR + "; color:#ccc;"
            "  selection-background-color:#094771;"
            "}"
        )

    @staticmethod
    def _slider_ss() -> str:
        return (
            "QSlider::groove:horizontal {"
            "  height:4px; background:#3c3c3c;"
            "  border-radius:2px;"
            "}"
            "QSlider::handle:horizontal {"
            "  width:12px; height:12px; margin:-4px 0;"
            "  background:#007acc; border-radius:6px;"
            "}"
        )