from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (QGraphicsRectItem, QGraphicsPolygonItem,
                              QGraphicsPixmapItem, QGraphicsItem)
from PyQt6.QtGui import (QPolygonF, QPen, QBrush, QColor, QFont,
                         QMouseEvent, QKeyEvent, QCursor, QUndoStack,
                         QPainter, QPixmap)
from PyQt6.QtCore import (Qt, QRectF, QPointF, QLineF,
                           QPoint, pyqtSignal)

from .canvas import AnnotationCanvas
from .annotations import (Annotation, AnnotationStore, LabelClass,
                           ToolMode, AddAnnotationCommand,
                           DeleteAnnotationCommand)
from .brush import BrushOverlay, BrushSettingsWidget
from .ai.suggestion import Suggestion


class DrawingCanvas(AnnotationCanvas):
    """
    Phase 3 + 4 + improvements canvas.
    Adds:
      - All drawing tools including SAM point/box
      - Polygon + mask move via Select tool drag
      - Mask edit mode (erase from existing annotation)
      - Suggestion overlays
      - Undo/redo
    """

    annotations_changed = pyqtSignal()
    sam_point_added     = pyqtSignal(list, list)
    sam_box_drawn       = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mode              = ToolMode.SELECT
        self._store             = AnnotationStore()
        self._undo_stack        = QUndoStack(self)
        self._classes: list[LabelClass] = []
        self._active_class_id   = 0

        self._drawing             = False
        self._draw_start          = QPointF()
        self._temp_item           = None
        self._poly_points: list[QPointF]  = []
        self._poly_items:  list           = []
        self._freehand_pts: list[QPointF] = []

        self._overlay_items: dict[str, list] = {}

        # ── Brush ─────────────────────────────────────────────────
        self._brush_overlay          = BrushOverlay()
        self._brush_painting         = False
        self._brush_last_pt: Optional[QPoint] = None
        self._brush_settings         = BrushSettingsWidget()
        self._brush_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._brush_settings._clear_btn.clicked.connect(
            self._brush_clear)

        # ── Mask edit state ───────────────────────────────────────
        # uid of annotation currently being edited via brush erase
        self._editing_uid: Optional[str] = None

        # ── AI suggestions ────────────────────────────────────────
        self._suggestions:      list[Suggestion] = []
        self._suggestion_items: list             = []
        self._sam_points:       list             = []
        self._sam_box_start:    Optional[QPointF]= None
        self._sam_temp_box                       = None

        # ── Move tracking ─────────────────────────────────────────
        # Maps QGraphicsItem -> annotation uid for move writeback
        self._item_to_uid: dict[int, str] = {}
        self._drag_start_pos: Optional[QPointF] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def set_tool(self, mode: ToolMode):
        self._mode = mode
        self._abort_drawing()
        if mode == ToolMode.BRUSH:
            self.setCursor(self._brush_cursor(
                self._brush_settings.brush_size))
        elif mode in (ToolMode.SAM_POINT, ToolMode.SAM_BOX):
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif mode == ToolMode.SELECT:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.setDragMode(
                self.DragMode.NoDrag)
        else:
            cursors = {
                ToolMode.BBOX:     Qt.CursorShape.CrossCursor,
                ToolMode.POLYGON:  Qt.CursorShape.CrossCursor,
                ToolMode.FREEHAND: Qt.CursorShape.CrossCursor,
            }
            self.setCursor(QCursor(
                cursors.get(mode, Qt.CursorShape.ArrowCursor)))

    def set_active_class(self, cid: int):
        self._active_class_id = cid

    def set_classes(self, classes: list[LabelClass]):
        self._classes = classes
        self.redraw_annotations()

    def load_image(self, path: str) -> bool:
        self._abort_drawing()
        self._safe_clear_overlays()
        self._clear_suggestion_overlays()
        self._brush_pixmap_item = None
        self._item_to_uid.clear()
        self._editing_uid = None

        ok = super().load_image(path)
        if ok:
            self._store.set_image(path)
            w, h = self.image_size()
            self._brush_overlay.set_image(path, w, h)
            self.redraw_annotations()
        return ok

    def delete_selected(self):
        for uid, items in list(self._overlay_items.items()):
            for item in items:
                try:
                    if item.isSelected():
                        self._undo_stack.push(
                            DeleteAnnotationCommand(
                                self._store, uid, self))
                        break
                except RuntimeError:
                    pass

    def redraw_annotations(self):
        self._safe_clear_overlays()
        self._item_to_uid.clear()
        for ann in self._store.current():
            cls   = self._get_class(ann.class_id)
            color = cls.color if cls else QColor("#888888")
            self._draw_annotation_overlay(ann, color)
        self.annotations_changed.emit()

    # =========================================================================
    # AI suggestion API
    # =========================================================================

    def set_suggestions(self, suggestions: list[Suggestion]):
        self._suggestions = suggestions
        self._draw_suggestion_overlays()
        self.annotations_changed.emit()

    def accept_suggestions(self):
        for sug in self._suggestions:
            ann = Annotation(**sug.to_annotation_kwargs())
            self._undo_stack.push(
                AddAnnotationCommand(self._store, ann, self))
        self.clear_suggestions()

    def reject_suggestions(self):
        self.clear_suggestions()

    def clear_suggestions(self):
        self._suggestions = []
        self._sam_points  = []
        self._clear_suggestion_overlays()
        self.annotations_changed.emit()

    def _clear_suggestion_overlays(self):
        for item in self._suggestion_items:
            try:
                item.scene()
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self._suggestion_items.clear()

    def _draw_suggestion_overlays(self):
        self._clear_suggestion_overlays()
        if not self._pixmap_item:
            return
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        W, H = br.width(), br.height()

        for sug in self._suggestions:
            color = (QColor("#00e5ff")
                     if sug.source == "sam"
                     else QColor("#ffb300"))
            pen   = QPen(color, 2, Qt.PenStyle.DashLine)
            fill  = QColor(color)
            fill.setAlpha(30)
            brush = QBrush(fill)

            if sug.ann_type == "polygon" and sug.points:
                poly = QPolygonF([
                    QPointF(p[0]*W, p[1]*H)
                    for p in sug.points])
                item = self._scene.addPolygon(poly, pen, brush)
                self._suggestion_items.append(item)
            elif sug.ann_type == "bbox":
                cx = sug.x_center*W;  cy = sug.y_center*H
                bw = sug.width*W;     bh = sug.height*H
                item = self._scene.addRect(
                    QRectF(cx-bw/2, cy-bh/2, bw, bh),
                    pen, brush)
                self._suggestion_items.append(item)

            lbl = self._scene.addText(
                f"{sug.source} {sug.score:.2f}",
                QFont("Arial", 9))
            lbl.setDefaultTextColor(color)
            if sug.ann_type == "polygon" and sug.points:
                lbl.setPos(
                    sug.points[0][0]*W,
                    sug.points[0][1]*H)
            else:
                lbl.setPos(
                    (sug.x_center - sug.width/2)*W,
                    (sug.y_center - sug.height/2)*H)
            self._suggestion_items.append(lbl)

    # =========================================================================
    # Mouse events
    # =========================================================================

    def mousePressEvent(self, event: QMouseEvent):
        if self._mode != ToolMode.SELECT and self._space_held:
            super().mousePressEvent(event)
            return

        pos = self.mapToScene(event.position().toPoint())
        btn = event.button()

        # ── BBox ──────────────────────────────────────────────────
        if (self._mode == ToolMode.BBOX
                and btn == Qt.MouseButton.LeftButton):
            self._drawing    = True
            self._draw_start = pos
            pen = QPen(QColor("#ffffff"), 1,
                       Qt.PenStyle.DashLine)
            self._temp_item = self._scene.addRect(
                QRectF(pos, pos), pen,
                QBrush(Qt.BrushStyle.NoBrush))

        # ── Polygon ───────────────────────────────────────────────
        elif (self._mode == ToolMode.POLYGON
              and btn == Qt.MouseButton.LeftButton):
            self._poly_points.append(pos)
            dot = self._scene.addEllipse(
                pos.x()-3, pos.y()-3, 6, 6,
                QPen(QColor("#ffff00")),
                QBrush(QColor("#ffff0088")))
            self._poly_items.append(dot)
            if len(self._poly_points) > 1:
                p, q = (self._poly_points[-2],
                        self._poly_points[-1])
                self._poly_items.append(
                    self._scene.addLine(
                        QLineF(p, q),
                        QPen(QColor("#ffff00"), 1)))

        elif (self._mode == ToolMode.POLYGON
              and btn == Qt.MouseButton.RightButton):
            self._finish_polygon()

        # ── Freehand ──────────────────────────────────────────────
        elif (self._mode == ToolMode.FREEHAND
              and btn == Qt.MouseButton.LeftButton):
            self._drawing      = True
            self._freehand_pts = [pos]

        # ── Brush ─────────────────────────────────────────────────
        elif (self._mode == ToolMode.BRUSH
              and btn == Qt.MouseButton.LeftButton):
            self._brush_painting = True
            self._brush_last_pt  = self._brush_scene_to_image(pos)
            self._brush_overlay.paint(
                self._brush_last_pt.x(),
                self._brush_last_pt.y(),
                self._brush_settings.brush_size,
                self._brush_settings.eraser)
            self._brush_update_overlay()

        # ── SAM point ─────────────────────────────────────────────
        elif (self._mode == ToolMode.SAM_POINT
              and btn == Qt.MouseButton.LeftButton):
            if not self._pixmap_item:
                return
            try:
                br = self._pixmap_item.boundingRect()
            except RuntimeError:
                return
            W = self._brush_overlay._w / br.width()
            H = self._brush_overlay._h / br.height()
            label = (0 if event.modifiers() &
                     Qt.KeyboardModifier.ShiftModifier else 1)
            img_x = int(pos.x() * W)
            img_y = int(pos.y() * H)
            self._sam_points.append((img_x, img_y, label))
            color = (QColor("#ff4444") if label == 0
                     else QColor("#00e5ff"))
            dot = self._scene.addEllipse(
                pos.x()-5, pos.y()-5, 10, 10,
                QPen(color, 2),
                QBrush(QColor(0, 0, 0, 0)))
            self._suggestion_items.append(dot)
            self.sam_point_added.emit(
                [(p[0], p[1]) for p in self._sam_points],
                [p[2]         for p in self._sam_points])

        # ── SAM box ───────────────────────────────────────────────
        elif (self._mode == ToolMode.SAM_BOX
              and btn == Qt.MouseButton.LeftButton):
            self._drawing       = True
            self._sam_box_start = pos
            pen = QPen(QColor("#00e5ff"), 2,
                       Qt.PenStyle.DashLine)
            self._sam_temp_box = self._scene.addRect(
                QRectF(pos, pos), pen,
                QBrush(Qt.BrushStyle.NoBrush))

        # ── Select (with move tracking) ───────────────────────────
        elif self._mode == ToolMode.SELECT:
            self._drag_start_pos = pos
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = self.mapToScene(event.position().toPoint())

        if (self._mode == ToolMode.BBOX
                and self._drawing and self._temp_item):
            self._temp_item.setRect(
                QRectF(self._draw_start, pos).normalized())

        elif self._mode == ToolMode.FREEHAND and self._drawing:
            self._freehand_pts.append(pos)
            if len(self._freehand_pts) > 1:
                p, q = (self._freehand_pts[-2],
                        self._freehand_pts[-1])
                self._poly_items.append(
                    self._scene.addLine(
                        QLineF(p, q),
                        QPen(QColor("#00ffff"), 1)))

        elif (self._mode == ToolMode.BRUSH
              and self._brush_painting):
            cur_pt = self._brush_scene_to_image(pos)
            if self._brush_last_pt:
                self._brush_overlay.paint_stroke(
                    self._brush_last_pt.x(),
                    self._brush_last_pt.y(),
                    cur_pt.x(), cur_pt.y(),
                    self._brush_settings.brush_size,
                    self._brush_settings.eraser)
            self._brush_last_pt = cur_pt
            self._brush_update_overlay()

        elif self._mode == ToolMode.BRUSH:
            self.setCursor(self._brush_cursor(
                self._brush_settings.brush_size))

        elif (self._mode == ToolMode.SAM_BOX
              and self._drawing and self._sam_temp_box):
            self._sam_temp_box.setRect(
                QRectF(self._sam_box_start, pos).normalized())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        pos = self.mapToScene(event.position().toPoint())

        if self._mode == ToolMode.BBOX and self._drawing:
            self._drawing = False
            rect = QRectF(self._draw_start, pos).normalized()
            if self._temp_item:
                self._scene.removeItem(self._temp_item)
                self._temp_item = None
            if rect.width() > 4 and rect.height() > 4:
                self._commit_bbox(rect)

        elif (self._mode == ToolMode.FREEHAND
              and self._drawing):
            self._drawing = False
            if len(self._freehand_pts) > 5:
                self._finish_freehand()
            else:
                self._abort_drawing()

        elif (self._mode == ToolMode.BRUSH
              and self._brush_painting):
            self._brush_painting = False
            self._brush_last_pt  = None

        elif (self._mode == ToolMode.SAM_BOX
              and self._drawing and self._sam_box_start):
            self._drawing = False
            rect = QRectF(
                self._sam_box_start, pos).normalized()
            if self._sam_temp_box:
                try:
                    self._scene.removeItem(self._sam_temp_box)
                except RuntimeError:
                    pass
                self._sam_temp_box = None
            if (rect.width() > 4 and rect.height() > 4
                    and self._pixmap_item):
                try:
                    br = self._pixmap_item.boundingRect()
                except RuntimeError:
                    return
                W = self._brush_overlay._w / br.width()
                H = self._brush_overlay._h / br.height()
                self.sam_box_drawn.emit((
                    int(rect.left()   * W),
                    int(rect.top()    * H),
                    int(rect.right()  * W),
                    int(rect.bottom() * H),
                ))

        elif self._mode == ToolMode.SELECT:
            # Write back moved annotation positions
            self._writeback_moved_items()
            super().mouseReleaseEvent(event)

        elif self._mode != ToolMode.POLYGON:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._mode == ToolMode.POLYGON:
            self._finish_polygon()
        elif self._mode == ToolMode.SELECT:
            # Double-click annotation to enter mask edit mode
            pos  = self.mapToScene(event.position().toPoint())
            item = self._scene.itemAt(
                pos, self.transform())
            uid  = self._uid_for_item(item)
            if uid:
                self._enter_mask_edit(uid)
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Escape:
            if self._suggestions:
                self.reject_suggestions()
            else:
                self._abort_drawing()
        elif event.key() in (Qt.Key.Key_Return,
                              Qt.Key.Key_Enter):
            if self._suggestions:
                self.accept_suggestions()
            elif self._mode == ToolMode.BRUSH:
                self._brush_commit()
            elif self._mode == ToolMode.POLYGON:
                self._finish_polygon()
        elif (event.key() == Qt.Key.Key_Z and
              event.modifiers() &
              Qt.KeyboardModifier.ControlModifier):
            self._undo_stack.undo()
        elif (event.key() == Qt.Key.Key_Y and
              event.modifiers() &
              Qt.KeyboardModifier.ControlModifier):
            self._undo_stack.redo()
        elif (event.key() == Qt.Key.Key_E
              and self._mode == ToolMode.BRUSH):
            self._brush_settings._erase_btn.toggle()
        else:
            super().keyPressEvent(event)

    # =========================================================================
    # Move writeback
    # =========================================================================

    def _writeback_moved_items(self):
        """
        After a drag in SELECT mode, update the AnnotationStore
        with the new normalised coordinates of any moved items.
        """
        if not self._pixmap_item:
            return
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        W, H = br.width(), br.height()
        if W == 0 or H == 0:
            return

        for uid, items in self._overlay_items.items():
            for item in items:
                try:
                    if not item.isSelected():
                        continue
                    ann = self._store.get(uid)
                    if ann is None:
                        continue

                    if isinstance(item, QGraphicsRectItem):
                        r  = item.mapToScene(
                            item.boundingRect())
                        xs = [r[i].x() for i in range(4)]
                        ys = [r[i].y() for i in range(4)]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        ann.x_center = ((x1+x2)/2) / W
                        ann.y_center = ((y1+y2)/2) / H
                        ann.width    = (x2-x1) / W
                        ann.height   = (y2-y1) / H

                    elif isinstance(item, QGraphicsPolygonItem):
                        scene_poly = item.mapToScene(
                            item.polygon())
                        ann.points = [
                            [scene_poly[i].x() / W,
                             scene_poly[i].y() / H]
                            for i in range(scene_poly.count())
                        ]
                except RuntimeError:
                    pass

    # =========================================================================
    # Mask edit (double-click to edit existing annotation)
    # =========================================================================

    def _uid_for_item(
            self, item) -> Optional[str]:
        """Find the annotation uid that owns a given scene item."""
        if item is None:
            return None
        for uid, items in self._overlay_items.items():
            if item in items:
                return uid
        return None

    def _enter_mask_edit(self, uid: str):
        """
        Load an existing polygon annotation into the brush overlay
        so the user can paint-erase parts of it.
        Switches to BRUSH tool automatically.
        """
        ann = self._store.get(uid)
        if ann is None or ann.ann_type != "polygon":
            return
        w, h = self.image_size()
        self._brush_overlay.load_from_polygon(
            ann.points, w, h)
        self._editing_uid = uid
        self.set_tool(ToolMode.BRUSH)
        # Switch eraser on so first strokes erase
        self._brush_settings._erase_btn.setChecked(True)
        self._brush_update_overlay()

    def _brush_commit(self):
        """
        If editing an existing annotation, replace its points.
        Otherwise create a new annotation.
        """
        polygons = self._brush_overlay.to_polygons()
        if not polygons:
            return

        if self._editing_uid:
            # Replace the existing annotation
            uid = self._editing_uid
            self._store.remove(uid)
            self._editing_uid = None
            # Add largest polygon as replacement
            pts = max(polygons, key=len)
            ann = Annotation(
                ann_type = "polygon",
                class_id = self._active_class_id,
                points   = pts)
            self._undo_stack.push(
                AddAnnotationCommand(self._store, ann, self))
        else:
            for pts in polygons:
                ann = Annotation(
                    ann_type = "polygon",
                    class_id = self._active_class_id,
                    points   = pts)
                self._undo_stack.push(
                    AddAnnotationCommand(self._store, ann, self))

        self._brush_overlay.clear()
        self._brush_update_overlay()

    # =========================================================================
    # Annotation drawing helpers
    # =========================================================================

    def _commit_bbox(self, rect: QRectF):
        if not self._pixmap_item:
            return
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        if br.width() == 0 or br.height() == 0:
            return
        ann = Annotation(
            ann_type = "bbox",
            class_id = self._active_class_id,
            x_center = rect.center().x() / br.width(),
            y_center = rect.center().y() / br.height(),
            width    = rect.width()       / br.width(),
            height   = rect.height()      / br.height(),
        )
        self._undo_stack.push(
            AddAnnotationCommand(self._store, ann, self))

    def _finish_polygon(self):
        if len(self._poly_points) < 3:
            self._abort_drawing()
            return
        try:
            br = self._pixmap_item.boundingRect()
        except (RuntimeError, AttributeError):
            self._abort_drawing()
            return
        pts = [[p.x() / br.width(), p.y() / br.height()]
               for p in self._poly_points]
        ann = Annotation(
            ann_type = "polygon",
            class_id = self._active_class_id,
            points   = pts)
        self._undo_stack.push(
            AddAnnotationCommand(self._store, ann, self))
        self._abort_drawing()

    def _finish_freehand(self):
        try:
            br = self._pixmap_item.boundingRect()
        except (RuntimeError, AttributeError):
            self._abort_drawing()
            return
        raw = np.array(
            [[p.x(), p.y()] for p in self._freehand_pts],
            dtype=np.float32)
        eps = max(2.0, cv2.arcLength(
            raw.reshape(-1, 1, 2), False) * 0.01)
        simplified = cv2.approxPolyDP(
            raw.reshape(-1, 1, 2), eps, False)
        pts = [[float(p[0][0]) / br.width(),
                float(p[0][1]) / br.height()]
               for p in simplified]
        ann = Annotation(
            ann_type = "polygon",
            class_id = self._active_class_id,
            points   = pts)
        self._undo_stack.push(
            AddAnnotationCommand(self._store, ann, self))
        self._abort_drawing()

    def _abort_drawing(self):
        self._drawing = False
        if self._temp_item:
            try:
                self._scene.removeItem(self._temp_item)
            except RuntimeError:
                pass
            self._temp_item = None
        if self._sam_temp_box:
            try:
                self._scene.removeItem(self._sam_temp_box)
            except RuntimeError:
                pass
            self._sam_temp_box = None
        for item in self._poly_items:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self._poly_items.clear()
        self._poly_points.clear()
        self._freehand_pts.clear()
        self._sam_box_start = None

    def _draw_annotation_overlay(
            self, ann: Annotation, color: QColor):
        if not self._pixmap_item:
            return
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        W, H = br.width(), br.height()
        if W == 0 or H == 0:
            return

        pen   = QPen(color, 1.5)
        fill  = QColor(color)
        fill.setAlpha(40)
        brush = QBrush(fill)
        items = []

        if ann.ann_type == "bbox":
            cx = ann.x_center * W;  cy = ann.y_center * H
            bw = ann.width * W;     bh = ann.height * H
            r  = self._scene.addRect(
                QRectF(cx-bw/2, cy-bh/2, bw, bh),
                pen, brush)
            r.setFlag(
                QGraphicsRectItem.GraphicsItemFlag
                .ItemIsSelectable)
            r.setFlag(
                QGraphicsRectItem.GraphicsItemFlag
                .ItemIsMovable)
            items.append(r)
            lbl = self._scene.addText(
                self._get_class_name(ann.class_id),
                QFont("Arial", 9))
            lbl.setDefaultTextColor(color)
            lbl.setPos(
                cx-bw/2,
                cy-bh/2 - lbl.boundingRect().height())
            items.append(lbl)

        elif ann.ann_type == "polygon" and ann.points:
            poly = QPolygonF([
                QPointF(p[0]*W, p[1]*H)
                for p in ann.points])
            pi = self._scene.addPolygon(poly, pen, brush)
            pi.setFlag(
                QGraphicsPolygonItem.GraphicsItemFlag
                .ItemIsSelectable)
            pi.setFlag(
                QGraphicsPolygonItem.GraphicsItemFlag
                .ItemIsMovable)
            items.append(pi)

        if items:
            self._overlay_items[ann.uid] = items

    # =========================================================================
    # Safe cleanup
    # =========================================================================

    def _safe_clear_overlays(self):
        for uid, items in list(self._overlay_items.items()):
            for it in items:
                try:
                    it.scene()
                    self._scene.removeItem(it)
                except RuntimeError:
                    pass
        self._overlay_items.clear()

    # =========================================================================
    # Brush helpers
    # =========================================================================

    def _brush_cursor(self, radius_scene: float) -> QCursor:
        r   = max(1, int(radius_scene * self._zoom_factor))
        dia = max(3, r * 2 + 2)
        pm  = QPixmap(dia, dia)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(1, 1, dia-2, dia-2)
        painter.end()
        return QCursor(pm, r, r)

    def _brush_scene_to_image(
            self, scene_pos: QPointF) -> QPoint:
        if not self._pixmap_item:
            return QPoint(0, 0)
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return QPoint(0, 0)
        w = self._brush_overlay._w
        h = self._brush_overlay._h
        if w == 0 or h == 0:
            return QPoint(0, 0)
        x = int(scene_pos.x() / br.width()  * w)
        y = int(scene_pos.y() / br.height() * h)
        return QPoint(max(0, min(w-1, x)),
                      max(0, min(h-1, y)))

    def _brush_update_overlay(self):
        if not self._pixmap_item:
            return
        try:
            br = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        vw, vh = int(br.width()), int(br.height())
        if vw <= 0 or vh <= 0:
            return
        cls   = self._get_class(self._active_class_id)
        color = cls.color if cls else QColor("#00ff00")
        pm    = self._brush_overlay.to_qpixmap(
            color, self._brush_settings.opacity, vw, vh)

        if self._brush_pixmap_item is None:
            try:
                self._brush_pixmap_item = (
                    self._scene.addPixmap(pm))
                self._brush_pixmap_item.setZValue(10)
                self._brush_pixmap_item.setPos(br.topLeft())
            except Exception:
                self._brush_pixmap_item = None
        else:
            try:
                self._brush_pixmap_item.setPixmap(pm)
                self._brush_pixmap_item.setPos(br.topLeft())
            except RuntimeError:
                self._brush_pixmap_item = None
                self._brush_update_overlay()

    def _brush_clear(self):
        self._editing_uid = None
        self._brush_overlay.clear()
        self._brush_update_overlay()
        self.annotations_changed.emit()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_class(self, cid: int) -> Optional[LabelClass]:
        return next(
            (c for c in self._classes if c.id == cid), None)

    def _get_class_name(self, cid: int) -> str:
        cls = self._get_class(cid)
        return cls.name if cls else f"class_{cid}"