from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QHBoxLayout,
    QVBoxLayout, QLabel, QSlider, QPushButton, QButtonGroup,
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QColorDialog, QScrollArea, QGroupBox, QPlainTextEdit,
    QMessageBox, QCheckBox, QComboBox, QSizePolicy,
    QAbstractItemView, QTabWidget,
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor
from PyQt6.QtCore import (Qt, QSize, QThread, QObject,
                           QRunnable, QThreadPool, pyqtSignal,
                           QTimer)

from PIL import Image

from .annotations import LabelClass, ToolMode
from .presets     import get_all_presets, classes_from_preset
from .constants   import (THUMB_SIZE, PANEL_COLOR, BORDER_COLOR,
                           DEFAULT_COLORS, THUMB_PLACEHOLDER_COLOR,
                           THUMB_LOAD_WORKERS)


# =============================================================================
# Thumbnail loader (background thread pool)
# =============================================================================

class _ThumbSignals(QObject):
    done = pyqtSignal(str, QPixmap)   # path, pixmap


class _ThumbLoader(QRunnable):
    def __init__(self, path: str, signals: _ThumbSignals):
        super().__init__()
        self._path    = path
        self._signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            img = Image.open(self._path).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            data = img.tobytes("raw", "RGB")
            qimg = QImage(
                data, img.width, img.height,
                img.width * 3, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg)
        except Exception:
            pm = QPixmap(THUMB_SIZE, THUMB_SIZE)
            pm.fill(QColor(THUMB_PLACEHOLDER_COLOR))
        self._signals.done.emit(self._path, pm)


# =============================================================================
# Thumbnail strip
# =============================================================================

class ImageThumbnailStrip(QListWidget):
    image_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(THUMB_SIZE + 24)
        self.setSpacing(4)
        self.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.setMovement(QListWidget.Movement.Static)
        self.setStyleSheet(f"""
            QListWidget {{
                background : {PANEL_COLOR};
                border      : none;
                border-right: 1px solid {BORDER_COLOR};
            }}
            QListWidget::item {{
                border-radius: 4px; padding: 2px;
                color: #cccccc; font-size: 10px;
            }}
            QListWidget::item:selected {{
                background: #094771;
                border: 1px solid #007acc;
            }}
            QListWidget::item:hover {{ background: #2a2d2e; }}
        """)
        self.currentItemChanged.connect(
            lambda cur, _: self.image_selected.emit(
                cur.data(Qt.ItemDataRole.UserRole))
            if cur else None)

        # Thread pool for background thumb loading
        self._pool      = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(THUMB_LOAD_WORKERS)
        self._signals   = _ThumbSignals()
        self._signals.done.connect(
            self._on_thumb_loaded,
            Qt.ConnectionType.QueuedConnection)
        self._path_to_row: dict[str, int] = {}

        # Placeholder pixmap
        self._placeholder = QPixmap(THUMB_SIZE, THUMB_SIZE)
        self._placeholder.fill(QColor(THUMB_PLACEHOLDER_COLOR))

    def load_folder(self, folder: str):
        self.clear()
        self._path_to_row.clear()

        exts  = {".jpg", ".jpeg", ".png", ".bmp",
                  ".tif", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in exts)

        # Add all items instantly with placeholder thumbnails
        for row, path in enumerate(paths):
            item = QListWidgetItem(
                QIcon(self._placeholder), path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setSizeHint(
                QSize(THUMB_SIZE + 8, THUMB_SIZE + 20))
            self.addItem(item)
            self._path_to_row[str(path)] = row

            # Queue real thumbnail load in background
            loader = _ThumbLoader(str(path), self._signals)
            self._pool.start(loader)

        if self.count() > 0:
            self.setCurrentRow(0)

    def _on_thumb_loaded(self, path: str, pm: QPixmap):
        """Called on main thread when a background thumb finishes."""
        row = self._path_to_row.get(path)
        if row is None:
            return
        item = self.item(row)
        if item:
            item.setIcon(QIcon(pm))

    def select_next(self):
        if self.currentRow() < self.count() - 1:
            self.setCurrentRow(self.currentRow() + 1)

    def select_prev(self):
        if self.currentRow() > 0:
            self.setCurrentRow(self.currentRow() - 1)


# =============================================================================
# Zoom widget
# =============================================================================

class ZoomWidget(QWidget):
    zoom_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        self._label = QLabel("100%")
        self._label.setStyleSheet(
            "color:#cccccc; font-size:12px; min-width:44px;")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(5, 400)
        self._slider.setValue(100)
        self._slider.setFixedWidth(120)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height:4px; background:#3c3c3c; border-radius:2px;
            }
            QSlider::handle:horizontal {
                width:12px; height:12px; margin:-4px 0;
                background:#007acc; border-radius:6px;
            }
        """)
        self._slider.valueChanged.connect(
            lambda v: self.zoom_requested.emit(v / 100.0))

        layout.addWidget(QLabel("🔍"))
        layout.addWidget(self._slider)
        layout.addWidget(self._label)

    def update_zoom(self, factor: float):
        pct = int(factor * 100)
        self._label.setText(f"{pct}%")
        self._slider.blockSignals(True)
        self._slider.setValue(min(400, max(5, pct)))
        self._slider.blockSignals(False)


# =============================================================================
# Class manager
# =============================================================================

class ClassManagerWidget(QWidget):
    class_selected  = pyqtSignal(int)
    classes_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self._classes:    list[LabelClass] = []
        self._next_id   = 0
        self._color_idx = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("Classes")
        title.setStyleSheet(
            "color:#cccccc; font-weight:500; font-size:13px;")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background:{PANEL_COLOR};
                border:1px solid {BORDER_COLOR};
                border-radius:4px;
            }}
            QListWidget::item {{
                color:#cccccc; padding:4px 6px;
                border-radius:3px; font-size:12px;
            }}
            QListWidget::item:selected {{
                background:#094771;
                border:1px solid #007acc;
            }}
            QListWidget::item:hover {{ background:#2a2d2e; }}
        """)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        for label, slot in [("+ Add",  self._add_class),
                             ("✎ Edit", self._edit_class),
                             ("✕ Del",  self._del_class)]:
            b = QPushButton(label)
            b.setStyleSheet("""
                QPushButton {
                    background:#3c3c3c; color:#cccccc;
                    border:none; border-radius:3px;
                    padding:3px 8px; font-size:11px;
                }
                QPushButton:hover { background:#505050; }
            """)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        import_btn = QPushButton("⬇  Import preset…")
        import_btn.setStyleSheet("""
            QPushButton {
                background:#1e6a99; color:#fff;
                border:none; border-radius:3px;
                padding:4px 8px; font-size:11px;
            }
            QPushButton:hover { background:#007acc; }
        """)
        import_btn.clicked.connect(self._import_preset)
        layout.addWidget(import_btn)

        for name in ["object", "person", "vehicle"]:
            self._add_named_class(name)

    def _add_class(self):
        self._add_named_class(f"class_{self._next_id}")

    def _add_named_class(self, name: str):
        color = QColor(
            DEFAULT_COLORS[self._color_idx % len(DEFAULT_COLORS)])
        self._color_idx += 1
        cls = LabelClass(id=self._next_id, name=name, color=color)
        self._next_id += 1
        self._classes.append(cls)
        self._refresh_list()
        self._list.setCurrentRow(len(self._classes) - 1)
        self.classes_changed.emit(self._classes)

    def _edit_class(self):
        row = self._list.currentRow()
        if row < 0:
            return
        cls = self._classes[row]
        dlg = ClassEditDialog(cls, self)
        if dlg.exec():
            cls.name, cls.color = dlg.name, dlg.color
            self._refresh_list()
            self.classes_changed.emit(self._classes)

    def _del_class(self):
        row = self._list.currentRow()
        if row < 0 or len(self._classes) <= 1:
            return
        self._classes.pop(row)
        self._refresh_list()
        self.classes_changed.emit(self._classes)

    def _import_preset(self):
        dlg = PresetImportDialog(self)
        if not dlg.exec():
            return
        new_classes = dlg.result_classes
        if not new_classes:
            return
        if dlg.merge:
            next_id = (max(c.id for c in self._classes) + 1
                       if self._classes else 0)
            for cls in new_classes:
                cls.id   = next_id
                next_id += 1
                self._classes.append(cls)
        else:
            self._classes = new_classes
        self._next_id   = max(c.id for c in self._classes) + 1
        self._color_idx = len(self._classes)
        self._refresh_list()
        self._list.setCurrentRow(0)
        self.classes_changed.emit(self._classes)

    def _refresh_list(self):
        self._list.clear()
        for cls in self._classes:
            swatch = QPixmap(14, 14)
            swatch.fill(cls.color)
            self._list.addItem(QListWidgetItem(
                QIcon(swatch), f"  {cls.id}: {cls.name}"))

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._classes):
            self.class_selected.emit(self._classes[row].id)

    def selected_class(self) -> Optional[LabelClass]:
        row = self._list.currentRow()
        return (self._classes[row]
                if 0 <= row < len(self._classes) else None)

    @property
    def classes(self) -> list[LabelClass]:
        return self._classes


# =============================================================================
# Class edit dialog
# =============================================================================

class ClassEditDialog(QDialog):
    def __init__(self, cls: LabelClass, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit class")
        self.name  = cls.name
        self.color = cls.color
        layout = QFormLayout(self)
        self._name_edit = QLineEdit(cls.name)
        layout.addRow("Name:", self._name_edit)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(48, 24)
        self._color_btn.setStyleSheet(
            f"background:{cls.color.name()}; border-radius:3px;")
        self._color_btn.clicked.connect(self._pick_color)
        layout.addRow("Color:", self._color_btn)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _pick_color(self):
        c = QColorDialog.getColor(self.color, self)
        if c.isValid():
            self.color = c
            self._color_btn.setStyleSheet(
                f"background:{c.name()}; border-radius:3px;")

    def _accept(self):
        self.name = self._name_edit.text().strip() or self.name
        self.accept()


# =============================================================================
# Tool selector toolbar
# =============================================================================

class ToolbarWidget(QWidget):
    tool_selected = pyqtSignal(ToolMode)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        tools = [
            (ToolMode.SELECT,    "⬡  Select",    "V"),
            (ToolMode.BBOX,      "⬜  BBox",      "B"),
            (ToolMode.POLYGON,   "⬠  Polygon",   "P"),
            (ToolMode.FREEHAND,  "✏  Freehand",  "F2"),
            (ToolMode.BRUSH,     "🖌  Brush",     "G"),
            (ToolMode.SAM_POINT, "●  SAM point", "S"),
            (ToolMode.SAM_BOX,   "⬜ SAM box",   "X"),
        ]
        for mode, label, shortcut in tools:
            btn = QPushButton(f"{label}  [{shortcut}]")
            btn.setCheckable(True)
            btn.setProperty("tool_mode", mode)
            btn.setStyleSheet("""
                QPushButton {
                    background:#3c3c3c; color:#aaa; border:none;
                    border-radius:4px; padding:4px 10px;
                    font-size:12px;
                }
                QPushButton:checked {
                    background:#007acc; color:#fff;
                }
                QPushButton:hover:!checked { background:#505050; }
            """)
            btn.clicked.connect(
                lambda _, m=mode: self.tool_selected.emit(m))
            self._btn_group.addButton(btn)
            layout.addWidget(btn)
            if mode == ToolMode.SELECT:
                btn.setChecked(True)

        layout.addStretch()

    def set_active_tool(self, mode: ToolMode):
        """Sync button checked state when tool changed via shortcut."""
        for btn in self._btn_group.buttons():
            tool = btn.property("tool_mode")
            if tool is not None:
                btn.setChecked(tool == mode)


# =============================================================================
# Preset import dialog
# =============================================================================

class PresetImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import / edit classes")
        self.setMinimumSize(640, 600)
        self.result_classes: list[LabelClass] = []
        self.merge = False
        self._rows: list[dict] = []

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Source ────────────────────────────────────────────────
        src_box = QGroupBox(
            "1.  Load from source  (optional)")
        src_box.setStyleSheet(self._group_style())
        src_layout = QHBoxLayout(src_box)
        src_layout.setSpacing(6)

        self._preset_combo = QComboBox()
        self._preset_combo.setStyleSheet(self._combo_style())
        self._presets = get_all_presets()
        self._preset_combo.addItem("— select built-in preset —")
        for p in self._presets:
            self._preset_combo.addItem(
                f"{p.name}  ({len(p.classes)} classes)")

        load_btn  = QPushButton("Load preset")
        load_btn.setStyleSheet(self._btn_style("#1e6a99"))
        load_btn.clicked.connect(self._load_preset)

        paste_btn = QPushButton("⬇  Paste names + palette…")
        paste_btn.setStyleSheet(self._btn_style("#3c5a3c"))
        paste_btn.clicked.connect(self._open_paste_dialog)

        clear_btn = QPushButton("✕  Clear all")
        clear_btn.setStyleSheet(self._btn_style("#7b2020"))
        clear_btn.clicked.connect(self._clear_rows)

        src_layout.addWidget(QLabel("Preset:"))
        src_layout.addWidget(self._preset_combo, stretch=1)
        src_layout.addWidget(load_btn)
        src_layout.addSpacing(10)
        src_layout.addWidget(paste_btn)
        src_layout.addSpacing(6)
        src_layout.addWidget(clear_btn)
        root.addWidget(src_box)

        # ── Form ──────────────────────────────────────────────────
        form_box = QGroupBox(
            "2.  Edit classes  "
            "(name + color — click swatch to change)")
        form_box.setStyleSheet(self._group_style())
        form_vbox = QVBoxLayout(form_box)
        form_vbox.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(6, 0, 6, 0)
        for txt, w in [("#", 32), ("Class name", None),
                        ("Color", 58), ("Del", 28)]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                "color:#666; font-size:11px; font-weight:500;")
            if w:
                lbl.setFixedWidth(w)
                hdr.addWidget(lbl)
            else:
                hdr.addWidget(lbl, stretch=1)
        form_vbox.addLayout(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea {"
            "  border:1px solid " + BORDER_COLOR + ";"
            "  border-radius:4px; background:" + PANEL_COLOR + ";"
            "}"
            "QScrollBar:vertical {"
            "  background:" + PANEL_COLOR + "; width:8px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background:#555; border-radius:4px;"
            "}"
            "QScrollBar::add-line:vertical,"
            "QScrollBar::sub-line:vertical { height:0; }"
        )
        self._row_container = QWidget()
        self._row_container.setStyleSheet(
            "background:" + PANEL_COLOR + ";")
        self._row_layout = QVBoxLayout(self._row_container)
        self._row_layout.setSpacing(1)
        self._row_layout.setContentsMargins(4, 4, 4, 4)
        self._row_layout.addStretch()
        scroll.setWidget(self._row_container)
        form_vbox.addWidget(scroll, stretch=1)

        add_row_btn = QPushButton("＋  Add class")
        add_row_btn.setStyleSheet(self._btn_style("#2d6a2d"))
        add_row_btn.clicked.connect(
            lambda: self._add_row("", QColor("#888888")))
        form_vbox.addWidget(add_row_btn)
        root.addWidget(form_box, stretch=1)

        # ── Bottom ────────────────────────────────────────────────
        bottom = QHBoxLayout()
        self._merge_chk = QCheckBox(
            "Merge  (append to existing classes)")
        self._merge_chk.setStyleSheet(
            "color:#ccc; font-size:12px;")
        bottom.addWidget(self._merge_chk)
        bottom.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self._btn_style("#3c3c3c"))
        cancel_btn.clicked.connect(self.reject)

        self._import_btn = QPushButton("✔  Import  (0 classes)")
        self._import_btn.setStyleSheet(self._btn_style("#007acc"))
        self._import_btn.clicked.connect(self._on_import)

        bottom.addWidget(cancel_btn)
        bottom.addWidget(self._import_btn)
        root.addLayout(bottom)

        self._update_count()

    # ── Row management ────────────────────────────────────────────

    def _add_row(self, name: str, color: QColor):
        idx       = len(self._rows)
        container = QWidget()
        container.setFixedHeight(32)
        container.setStyleSheet(
            "background:" +
            ("#2a2d2e" if idx % 2 == 0 else PANEL_COLOR) +
            "; border-radius:3px;")
        h = QHBoxLayout(container)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(6)

        num_lbl = QLabel(str(idx + 1))
        num_lbl.setFixedWidth(26)
        num_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight |
            Qt.AlignmentFlag.AlignVCenter)
        num_lbl.setStyleSheet("color:#555; font-size:11px;")

        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("class name…")
        name_edit.setStyleSheet(
            "QLineEdit {"
            "  background:#1e1e1e; color:#ddd;"
            "  border:1px solid " + BORDER_COLOR + ";"
            "  border-radius:3px; padding:1px 6px;"
            "  font-size:12px;"
            "}"
            "QLineEdit:focus { border:1px solid #007acc; }"
        )
        name_edit.textChanged.connect(self._update_count)

        color_btn = QPushButton()
        color_btn.setFixedSize(52, 22)
        color_btn.setToolTip(color.name())
        color_btn.setStyleSheet(
            "background:" + color.name() + ";"
            "border:1px solid " + BORDER_COLOR + ";"
            "border-radius:3px;")

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet("""
            QPushButton {
                background:#3c3c3c; color:#777;
                border:none; border-radius:3px; font-size:10px;
            }
            QPushButton:hover {
                background:#c0392b; color:#fff;
            }
        """)

        h.addWidget(num_lbl)
        h.addWidget(name_edit, stretch=1)
        h.addWidget(color_btn)
        h.addWidget(del_btn)

        row_data = {
            "container":  container,
            "num_lbl":    num_lbl,
            "name_edit":  name_edit,
            "color_btn":  color_btn,
            "color":      color,
        }
        self._rows.append(row_data)
        color_btn.clicked.connect(
            lambda _, rd=row_data: self._pick_color(rd))
        del_btn.clicked.connect(
            lambda _, rd=row_data: self._remove_row(rd))
        self._row_layout.insertWidget(
            self._row_layout.count() - 1, container)
        self._update_count()
        self._row_container.adjustSize()

    def _remove_row(self, row_data: dict):
        if row_data not in self._rows:
            return
        self._rows.remove(row_data)
        row_data["container"].setParent(None)
        row_data["container"].deleteLater()
        self._renumber()
        self._update_count()

    def _renumber(self):
        for i, rd in enumerate(self._rows):
            rd["num_lbl"].setText(str(i + 1))
            rd["container"].setStyleSheet(
                "background:" +
                ("#2a2d2e" if i % 2 == 0 else PANEL_COLOR) +
                "; border-radius:3px;")

    def _clear_rows(self):
        for rd in list(self._rows):
            rd["container"].setParent(None)
            rd["container"].deleteLater()
        self._rows.clear()
        self._update_count()

    def _pick_color(self, row_data: dict):
        c = QColorDialog.getColor(row_data["color"], self)
        if c.isValid():
            row_data["color"] = c
            row_data["color_btn"].setStyleSheet(
                "background:" + c.name() + ";"
                "border:1px solid " + BORDER_COLOR + ";"
                "border-radius:3px;")
            row_data["color_btn"].setToolTip(c.name())

    def _update_count(self):
        n = sum(1 for rd in self._rows
                if rd["name_edit"].text().strip())
        self._import_btn.setText(f"✔  Import  ({n} classes)")

    def _load_preset(self):
        idx = self._preset_combo.currentIndex()
        if idx < 1:
            return
        preset = self._presets[idx - 1]
        self._clear_rows()
        for name, rgb in zip(preset.classes, preset.palette):
            self._add_row(
                name,
                QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))

    def _open_paste_dialog(self):
        dlg = _PasteDialog(self)
        if not dlg.exec():
            return
        names, palette = dlg.parsed_names, dlg.parsed_palette
        if not names:
            return
        self._clear_rows()
        for i, name in enumerate(names):
            if i < len(palette):
                color = QColor(int(palette[i][0]),
                               int(palette[i][1]),
                               int(palette[i][2]))
            else:
                hue   = int((i * 137.508) % 360)
                color = QColor.fromHsv(hue, 200, 210)
            self._add_row(name, color)

    def _on_import(self):
        result = []
        for i, rd in enumerate(self._rows):
            name = rd["name_edit"].text().strip()
            if not name:
                continue
            result.append(LabelClass(
                id=i, name=name, color=rd["color"]))
        if not result:
            QMessageBox.warning(
                self, "Nothing to import",
                "Add at least one class name before importing.")
            return
        self.result_classes = result
        self.merge          = self._merge_chk.isChecked()
        self.accept()

    # ── Style helpers ─────────────────────────────────────────────

    @staticmethod
    def _group_style() -> str:
        return (
            "QGroupBox {"
            "  color:#aaa; font-size:12px; font-weight:500;"
            "  border:1px solid " + BORDER_COLOR + ";"
            "  border-radius:6px; margin-top:8px;"
            "  padding-top:6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin:margin;"
            "  subcontrol-position:top left; padding:0 4px;"
            "}"
        )

    @staticmethod
    def _combo_style() -> str:
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
    def _btn_style(bg: str) -> str:
        return (
            "QPushButton {"
            "  background:" + bg + "; color:#fff; border:none;"
            "  border-radius:4px; padding:4px 12px;"
            "  font-size:12px;"
            "}"
            "QPushButton:hover { background:" + bg + "bb; }"
        )


# =============================================================================
# Paste sub-dialog
# =============================================================================

class _PasteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste names + palette")
        self.setMinimumSize(500, 360)
        self.parsed_names:   list = []
        self.parsed_palette: list = []

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self._lbl(
            "Class names — one per line, or Python list:"))
        self._names_edit = self._textedit(
            'road\nsidewalk\nbuilding')
        layout.addWidget(self._names_edit)

        layout.addWidget(self._lbl(
            "RGB palette — Python list of [R,G,B]  "
            "(leave blank for auto colors):"))
        self._palette_edit = self._textedit(
            "[[128,64,128],[244,35,232],...]")
        layout.addWidget(self._palette_edit)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(
            QDialogButtonBox.StandardButton.Ok).setText(
            "Parse & fill form")
        btns.accepted.connect(self._parse)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _parse(self):
        import ast
        raw_names   = self._names_edit.toPlainText().strip()
        raw_palette = self._palette_edit.toPlainText().strip()

        if raw_names.startswith("["):
            try:
                names = ast.literal_eval(raw_names)
                if not isinstance(names, list):
                    raise ValueError
            except Exception:
                QMessageBox.warning(
                    self, "Parse error",
                    "Could not parse class names.")
                return
        else:
            names = [n.strip().strip('"').strip("'")
                     for n in raw_names.splitlines()
                     if n.strip()]
        if not names:
            QMessageBox.warning(
                self, "Parse error", "No class names found.")
            return

        palette = []
        placeholder = "[[128,64,128],[244,35,232],...]"
        if raw_palette and raw_palette != placeholder:
            try:
                parsed = ast.literal_eval(raw_palette)
                if not isinstance(parsed, list):
                    raise ValueError
                palette = parsed
            except Exception:
                QMessageBox.warning(
                    self, "Parse error",
                    "Could not parse palette.")
                return

        self.parsed_names   = names
        self.parsed_palette = palette
        self.accept()

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#ccc; font-size:12px;")
        lbl.setWordWrap(True)
        return lbl

    @staticmethod
    def _textedit(placeholder: str) -> QPlainTextEdit:
        w = QPlainTextEdit()
        w.setPlaceholderText(placeholder)
        w.setMaximumHeight(110)
        w.setStyleSheet(
            "QPlainTextEdit {"
            "  background:#1e1e1e; color:#ccc;"
            "  border:1px solid " + BORDER_COLOR + ";"
            "  border-radius:4px; font-size:11px;"
            "  font-family:monospace; padding:4px;"
            "}"
        )
        return w