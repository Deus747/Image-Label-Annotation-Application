from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum, auto

from PyQt6.QtGui import QColor, QUndoCommand, QUndoStack


class ToolMode(Enum):
    SELECT    = auto()
    BBOX      = auto()
    POLYGON   = auto()
    FREEHAND  = auto()
    BRUSH     = auto()
    SAM_POINT = auto()
    SAM_BOX   = auto()


class MaskType(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"


class MaskFormat(Enum):
    GRAYSCALE = "grayscale"
    COLOR     = "color"


@dataclass
class LabelClass:
    id:    int
    name:  str
    color: QColor


@dataclass
class Annotation:
    uid:      str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_id: int   = 0
    ann_type: str   = "bbox"
    x_center: float = 0.0
    y_center: float = 0.0
    width:    float = 0.0
    height:   float = 0.0
    points:   list  = field(default_factory=list)

    def to_yolo(self) -> str:
        if self.ann_type == "bbox":
            return (f"{self.class_id} {self.x_center:.6f} "
                    f"{self.y_center:.6f} {self.width:.6f} "
                    f"{self.height:.6f}")
        pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {pts}"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(**d)


class AnnotationStore:
    def __init__(self):
        self._data: dict[str, list[Annotation]] = {}
        self._current_path: str = ""

    def set_image(self, path: str):
        self._current_path = path
        if path not in self._data:
            self._data[path] = []

    def add(self, ann: Annotation):
        self._data[self._current_path].append(ann)

    def remove(self, uid: str):
        self._data[self._current_path] = [
            a for a in self._data[self._current_path]
            if a.uid != uid]

    def get(self, uid: str) -> Optional[Annotation]:
        return next(
            (a for a in self._data.get(self._current_path, [])
             if a.uid == uid), None)

    def current(self) -> list[Annotation]:
        return self._data.get(self._current_path, [])

    def save_json(self, path: str):
        out = {img: [a.to_dict() for a in anns]
               for img, anns in self._data.items()}
        Path(path).write_text(json.dumps(out, indent=2))

    def load_json(self, path: str):
        raw = json.loads(Path(path).read_text())
        self._data = {
            img: [Annotation.from_dict(d) for d in anns]
            for img, anns in raw.items()}


# ── Undo commands ─────────────────────────────────────────────────────────

class AddAnnotationCommand(QUndoCommand):
    def __init__(self, store: AnnotationStore,
                 ann: Annotation, canvas):
        super().__init__(f"Add {ann.ann_type}")
        self._store  = store
        self._ann    = ann
        self._canvas = canvas

    def redo(self):
        self._store.add(self._ann)
        self._canvas.redraw_annotations()

    def undo(self):
        self._store.remove(self._ann.uid)
        self._canvas.redraw_annotations()


class DeleteAnnotationCommand(QUndoCommand):
    def __init__(self, store: AnnotationStore,
                 uid: str, canvas):
        super().__init__("Delete annotation")
        self._store  = store
        self._uid    = uid
        self._canvas = canvas
        self._backup: Optional[Annotation] = store.get(uid)

    def redo(self):
        self._store.remove(self._uid)
        self._canvas.redraw_annotations()

    def undo(self):
        if self._backup:
            self._store.add(self._backup)
            self._canvas.redraw_annotations()