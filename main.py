from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

# Disable ultralytics telemetry before any imports
os.environ["YOLO_OFFLINE"]        = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QHBoxLayout, QDockWidget, QToolBar,
                              QStatusBar, QLabel, QFileDialog,
                              QMessageBox)
from PyQt6.QtGui  import QKeyEvent, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QPointF, QObject, pyqtSignal

from .tools       import DrawingCanvas
from .widgets     import (ImageThumbnailStrip, ZoomWidget,
                          ClassManagerWidget, ToolbarWidget)
from .mask        import MaskPreviewWidget
from .export      import ExportPanel
from .session     import SessionManager
from .annotations import ToolMode
from .constants   import PANEL_COLOR, BORDER_COLOR
from .ai.sam_engine  import SamEngine
from .ai.yolo_engine import YoloEngine
from .ai.ai_toolbar  import AiToolbar


class _AiSignals(QObject):
    """Thread-safe bridge for AI inference results."""
    result = pyqtSignal(list)
    error  = pyqtSignal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabelApp")
        self.resize(1440, 860)
        self.setStyleSheet(
            "QMainWindow { background: #1e1e1e; }")

        # ── State ─────────────────────────────────────────────────
        self._session        = SessionManager()
        self._current_folder = ""
        self._active_image   = ""

        # ── Core widgets ──────────────────────────────────────────
        self._canvas      = DrawingCanvas()
        self._thumb_strip = ImageThumbnailStrip()
        self._class_mgr   = ClassManagerWidget()
        self._mask_prev   = MaskPreviewWidget()
        self._toolbar_w   = ToolbarWidget()
        self._export_pnl  = ExportPanel(
            self._canvas._store, self._mask_prev)

        # ── AI ────────────────────────────────────────────────────
        self._sam_engine  = SamEngine()
        self._yolo_engine = YoloEngine()
        self._yolo_conf   = 0.25
        self._ai_toolbar  = AiToolbar(
            self._sam_engine, self._yolo_engine)

        self._ai_signals = _AiSignals()
        self._ai_signals.result.connect(
            self._on_ai_result_main,
            Qt.ConnectionType.QueuedConnection)
        self._ai_signals.error.connect(
            self._on_ai_error_main,
            Qt.ConnectionType.QueuedConnection)

        # ── Docks ─────────────────────────────────────────────────
        self._mask_dock = self._make_dock(
            "Mask Preview", self._mask_prev,
            Qt.DockWidgetArea.RightDockWidgetArea, 280)
        self._export_dock = self._make_dock(
            "Export", self._export_pnl,
            Qt.DockWidgetArea.RightDockWidgetArea, 200)
        self._class_dock = self._make_dock(
            "Classes", self._class_mgr,
            Qt.DockWidgetArea.RightDockWidgetArea, 230)
        self.tabifyDockWidget(
            self._mask_dock, self._export_dock)
        self._mask_dock.raise_()

        self._brush_dock = self._make_dock(
            "Brush Settings",
            self._canvas._brush_settings,
            Qt.DockWidgetArea.RightDockWidgetArea, 220)
        self._brush_dock.hide()

        self._ai_dock = self._make_dock(
            "AI Tools", self._ai_toolbar,
            Qt.DockWidgetArea.LeftDockWidgetArea, 240)

        tool_dock = QDockWidget("Tools", self)
        tool_dock.setWidget(self._toolbar_w)
        tool_dock.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea, tool_dock)

        # ── Central ───────────────────────────────────────────────
        central  = QWidget()
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)
        h_layout.addWidget(self._thumb_strip)
        h_layout.addWidget(self._canvas, stretch=1)
        self.setCentralWidget(central)

        # ── Status bar ────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet(
            "QStatusBar {"
            "  background:" + PANEL_COLOR + "; color:#858585;"
            "  font-size:12px;"
            "  border-top:1px solid " + BORDER_COLOR + ";"
            "}")
        self._coord_label = QLabel("x: — , y: —")
        self._zoom_widget = ZoomWidget()
        sb.addPermanentWidget(self._coord_label)
        sb.addPermanentWidget(self._zoom_widget)
        self.setStatusBar(sb)

        # ── Top toolbar ───────────────────────────────────────────
        tb = QToolBar()
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar {"
            "  background:" + PANEL_COLOR + ";"
            "  border-bottom:1px solid " + BORDER_COLOR + ";"
            "  padding:2px 6px;"
            "}"
            "QToolButton {"
            "  color:#ccc; padding:4px 10px;"
            "  border-radius:4px; font-size:13px;"
            "}"
            "QToolButton:hover { background:#3c3c3c; }"
        )
        for label, slot in [
            ("📂  Open Folder",   self._open_folder),
            ("⊡  Fit",            self._canvas.fit_in_view),
            ("↩  Undo",           self._canvas._undo_stack.undo),
            ("↪  Redo",           self._canvas._undo_stack.redo),
            ("💾  Save Session",  self._save_session),
            ("📂  Load Session",  self._load_session),
            ("🗑  Clear Session", self._clear_session),
        ]:
            tb.addAction(label).triggered.connect(slot)
        self.addToolBar(tb)

        # ── Signals ───────────────────────────────────────────────
        self._thumb_strip.image_selected.connect(
            self._load_image)
        self._thumb_strip.image_selected.connect(
            self._auto_save)
        self._canvas.zoom_changed.connect(
            self._zoom_widget.update_zoom)
        self._canvas.cursor_moved.connect(
            self._update_coords)
        self._zoom_widget.zoom_requested.connect(
            self._canvas.set_zoom)
        self._class_mgr.class_selected.connect(
            self._canvas.set_active_class)
        self._class_mgr.classes_changed.connect(
            self._canvas.set_classes)
        self._toolbar_w.tool_selected.connect(
            self._on_tool_changed)
        self._canvas.annotations_changed.connect(
            self._update_mask_preview)
        self._canvas.annotations_changed.connect(
            self._sync_suggestion_count)
        self._mask_prev._type_combo.currentIndexChanged.connect(
            lambda _: self._update_mask_preview())
        self._mask_prev._fmt_combo.currentIndexChanged.connect(
            lambda _: self._update_mask_preview())

        self._ai_toolbar.sam_mode_changed.connect(
            self._on_sam_mode)
        self._ai_toolbar.conf_changed.connect(
            self._on_conf_changed)
        self._ai_toolbar.yolo_run_requested.connect(
            self._run_yolo)
        self._ai_toolbar.accept_all.connect(
            self._canvas.accept_suggestions)
        self._ai_toolbar.reject_all.connect(
            self._canvas.reject_suggestions)
        self._ai_toolbar.status_message.connect(
            lambda m: self.statusBar().showMessage(m, 3000))
        self._canvas.sam_point_added.connect(
            self._on_sam_point)
        self._canvas.sam_box_drawn.connect(
            self._on_sam_box)

        # ── Init ──────────────────────────────────────────────────
        self._canvas.set_classes(self._class_mgr.classes)

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1440, screen.width()),
            min(860,  screen.height()))

        # Register all application-level shortcuts
        self._register_shortcuts()

        if self._session.exists():
            self._load_session(silent=True)

    # =========================================================================
    # Shortcuts
    # =========================================================================

    def _register_shortcuts(self):
        def _sc(key: str, slot):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(
                Qt.ShortcutContext.ApplicationShortcut)
            s.activated.connect(slot)

        # Tools
        _sc("V",  lambda: self._set_tool(ToolMode.SELECT))
        _sc("B",  lambda: self._set_tool(ToolMode.BBOX))
        _sc("P",  lambda: self._set_tool(ToolMode.POLYGON))
        _sc("F2", lambda: self._set_tool(ToolMode.FREEHAND))
        _sc("G",  lambda: self._set_tool(ToolMode.BRUSH))
        _sc("S",  lambda: self._set_tool(ToolMode.SAM_POINT))
        _sc("X",  lambda: self._set_tool(ToolMode.SAM_BOX))

        # Canvas
        _sc("F",            self._canvas.fit_in_view)
        _sc("+",            self._canvas.zoom_in)
        _sc("=",            self._canvas.zoom_in)
        _sc("-",            self._canvas.zoom_out)
        _sc("Ctrl+Z",       self._canvas._undo_stack.undo)
        _sc("Ctrl+Y",       self._canvas._undo_stack.redo)
        _sc("Ctrl+Shift+Z", self._canvas._undo_stack.redo)
        _sc("Delete",       self._canvas.delete_selected)
        _sc("Escape",       self._on_escape)
        _sc("Return",       self._on_enter)
        _sc("Enter",        self._on_enter)

        # Brush
        _sc("E", self._toggle_eraser)

        # Navigation
        _sc("Right", self._thumb_strip.select_next)
        _sc("Left",  self._thumb_strip.select_prev)
        _sc("D",     self._thumb_strip.select_next)
        _sc("A",     self._thumb_strip.select_prev)

        # Session
        _sc("Ctrl+S", self._save_session)

        # AI suggestions
        _sc("Ctrl+A", self._canvas.accept_suggestions)
        _sc("Ctrl+R", self._canvas.reject_suggestions)

    def _set_tool(self, mode: ToolMode):
        """Set tool and sync toolbar button state."""
        self._canvas.set_tool(mode)
        self._toolbar_w.set_active_tool(mode)
        self._brush_dock.setVisible(mode == ToolMode.BRUSH)

    def _on_escape(self):
        if self._canvas._suggestions:
            self._canvas.reject_suggestions()
        else:
            self._canvas._abort_drawing()

    def _on_enter(self):
        if self._canvas._suggestions:
            self._canvas.accept_suggestions()
        elif self._canvas._mode == ToolMode.BRUSH:
            self._canvas._brush_commit()
        elif self._canvas._mode == ToolMode.POLYGON:
            self._canvas._finish_polygon()

    def _toggle_eraser(self):
        if self._canvas._mode == ToolMode.BRUSH:
            self._canvas._brush_settings._erase_btn.toggle()

    # =========================================================================
    # Dock helper
    # =========================================================================

    def _make_dock(
            self, title: str, widget: QWidget,
            area: Qt.DockWidgetArea,
            min_width: int) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setMinimumWidth(min_width)
        self.addDockWidget(area, dock)
        return dock

    # =========================================================================
    # Tool handling
    # =========================================================================

    def _on_tool_changed(self, mode: ToolMode):
        self._set_tool(mode)

    # =========================================================================
    # Image loading
    # =========================================================================

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open Image Folder")
        if folder:
            self._current_folder = folder
            self._thumb_strip.load_folder(folder)

    def _load_image(self, path: str):
        self._canvas.blockSignals(True)
        try:
            ok = self._canvas.load_image(path)
        finally:
            self._canvas.blockSignals(False)
        if ok:
            self._active_image = path
            self.setWindowTitle(
                f"LabelApp — {Path(path).name}")
            self._canvas.clear_suggestions()
            w, h = self._canvas.image_size()
            self._sam_engine.set_image(path, w, h)
        self._update_mask_preview()

    # =========================================================================
    # Mask preview
    # =========================================================================

    def _update_mask_preview(self):
        try:
            w, h = self._canvas.image_size()
            self._mask_prev.update_mask(
                self._canvas._store.current(),
                self._class_mgr.classes, w, h)
        except RuntimeError:
            pass

    # =========================================================================
    # Status bar
    # =========================================================================

    def _update_coords(self, pos: QPointF):
        if self._canvas._pixmap_item:
            try:
                br = self._canvas._pixmap_item.boundingRect()
                if br.contains(pos):
                    self._coord_label.setText(
                        f"x: {int(pos.x()):>5} , "
                        f"y: {int(pos.y()):>5}")
                    return
            except RuntimeError:
                pass
        self._coord_label.setText("x: — , y: —")

    # =========================================================================
    # Session
    # =========================================================================

    def _auto_save(self, _path: str):
        self._save_session(silent=True)

    def _save_session(self, silent: bool = False):
        active_cls = self._class_mgr.selected_class()
        self._session.save(
            folder          = self._current_folder,
            classes         = self._class_mgr.classes,
            store           = self._canvas._store,
            active_image    = self._active_image,
            active_class_id = (active_cls.id
                               if active_cls else 0),
            brush_overlay   = self._canvas._brush_overlay,
        )
        if not silent:
            self.statusBar().showMessage(
                "Session saved.", 2000)

    def _load_session(self, silent: bool = False):
        data = self._session.load()
        if not data:
            if not silent:
                self.statusBar().showMessage(
                    "No saved session found.", 2000)
            return

        folder = data.get("folder", "")
        if folder and Path(folder).exists():
            self._current_folder = folder
            self._thumb_strip.load_folder(folder)

        classes = SessionManager.classes_from_data(data)
        if classes:
            self._class_mgr._classes   = classes
            self._class_mgr._next_id   = (
                max(c.id for c in classes) + 1)
            self._class_mgr._color_idx = len(classes)
            self._class_mgr._refresh_list()
            self._class_mgr._list.setCurrentRow(0)
            self._class_mgr.classes_changed.emit(classes)

        restored = SessionManager.store_from_data(data)
        self._canvas._store     = restored
        self._export_pnl._store = restored

        brush_data = data.get("brush_masks", {})
        if brush_data:
            self._canvas._brush_overlay.deserialize(brush_data)

        active = data.get("active_image", "")
        if active and Path(active).exists():
            self._active_image = active
            self._canvas.blockSignals(True)
            try:
                self._canvas.load_image(active)
            finally:
                self._canvas.blockSignals(False)

            for i in range(self._thumb_strip.count()):
                item = self._thumb_strip.item(i)
                if (item and
                        item.data(Qt.ItemDataRole.UserRole)
                        == active):
                    self._thumb_strip.blockSignals(True)
                    self._thumb_strip.setCurrentRow(i)
                    self._thumb_strip.blockSignals(False)
                    break

            self.setWindowTitle(
                f"LabelApp — {Path(active).name}")
            w, h = self._canvas.image_size()
            self._sam_engine.set_image(active, w, h)
            self._update_mask_preview()

        active_cid = data.get("active_class_id", 0)
        for i, cls in enumerate(self._class_mgr.classes):
            if cls.id == active_cid:
                self._class_mgr._list.blockSignals(True)
                self._class_mgr._list.setCurrentRow(i)
                self._class_mgr._list.blockSignals(False)
                self._canvas.set_active_class(active_cid)
                break

        if not silent:
            self.statusBar().showMessage(
                "Session restored.", 2000)

    def _clear_session(self):
        reply = QMessageBox.question(
            self, "Clear session",
            "Delete the saved session file?",
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._session.delete()
            self.statusBar().showMessage(
                "Session cleared.", 2000)

    # =========================================================================
    # AI
    # =========================================================================

    def _on_sam_mode(self, mode: str):
        mapping = {
            "point": ToolMode.SAM_POINT,
            "box":   ToolMode.SAM_BOX,
            "off":   ToolMode.SELECT,
        }
        self._set_tool(mapping.get(mode, ToolMode.SELECT))

    def _on_conf_changed(self, conf: float):
        self._yolo_conf = conf

    def _on_sam_point(self, points: list, labels: list):
        if not self._sam_engine.loaded:
            self.statusBar().showMessage(
                "Load SAM model first.", 2000)
            return
        w, h = self._canvas.image_size()
        self._sam_engine.set_image(self._active_image, w, h)
        cls = self._class_mgr.selected_class()
        self._sam_engine.predict_point(
            points    = points,
            labels    = labels,
            class_id  = cls.id if cls else 0,
            on_result = self._ai_signals.result.emit,
            on_error  = self._ai_signals.error.emit,
        )

    def _on_sam_box(self, box: tuple):
        if not self._sam_engine.loaded:
            self.statusBar().showMessage(
                "Load SAM model first.", 2000)
            return
        w, h = self._canvas.image_size()
        self._sam_engine.set_image(self._active_image, w, h)
        cls = self._class_mgr.selected_class()
        self._sam_engine.predict_box(
            box       = box,
            class_id  = cls.id if cls else 0,
            on_result = self._ai_signals.result.emit,
            on_error  = self._ai_signals.error.emit,
        )

    def _run_yolo(self):
        if not self._yolo_engine.loaded:
            self.statusBar().showMessage(
                "Load a YOLO model first.", 2000)
            return
        if not self._active_image:
            self.statusBar().showMessage(
                "Open an image first.", 2000)
            return
        self._ai_toolbar.set_yolo_loading(True)
        w, h = self._canvas.image_size()
        cls  = self._class_mgr.selected_class()
        self._yolo_engine.predict(
            img_path  = self._active_image,
            img_w     = w,
            img_h     = h,
            conf      = self._yolo_conf,
            class_id  = cls.id if cls else 0,
            on_result = self._ai_signals.result.emit,
            on_error  = self._ai_signals.error.emit,
        )

    def _on_ai_result_main(self, suggestions: list):
        self._canvas.set_suggestions(suggestions)
        self._ai_toolbar.set_yolo_loading(False)

    def _on_ai_error_main(self, msg: str):
        self.statusBar().showMessage(
            f"AI error: {msg}", 4000)
        self._ai_toolbar.set_yolo_loading(False)

    def _sync_suggestion_count(self):
        self._ai_toolbar.set_pending_count(
            len(self._canvas._suggestions))

    # =========================================================================
    # Keyboard navigation (fallback — shortcuts cover most cases)
    # =========================================================================

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)


# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()