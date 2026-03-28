from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Suggestion:
    """
    A pending AI-generated annotation not yet committed
    to the AnnotationStore.

    source   : "sam" | "yolo"
    score    : confidence / IoU  0-1
    Coordinates use the same normalised 0-1 convention as Annotation.
    """
    ann_type:  str   = "polygon"
    source:    str   = "sam"
    score:     float = 1.0
    class_id:  int   = 0

    x_center: float = 0.0
    y_center: float = 0.0
    width:    float = 0.0
    height:   float = 0.0

    points: list = field(default_factory=list)
    mask:   Optional[np.ndarray] = field(default=None, repr=False)

    def to_annotation_kwargs(self) -> dict:
        return dict(
            ann_type = self.ann_type,
            class_id = self.class_id,
            x_center = self.x_center,
            y_center = self.y_center,
            width    = self.width,
            height   = self.height,
            points   = self.points,
        )