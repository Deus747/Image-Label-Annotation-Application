from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassPreset:
    name:    str
    classes: list[str]
    palette: np.ndarray   # shape (N, 3) RGB uint8


# ── IDD (India Driving Dataset) ───────────────────────────────────────────

IDD_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle", "parking",
    "rail track", "animal", "autorickshaw", "caravan",
    "trailer", "curb", "guard rail", "billboard", "bridge",
    "tunnel", "vehicle fallback", "obs-str-bar-fallback",
    "drivable fallback", "non-drivable fallback",
]

IDD_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32], [250, 170, 160],
    [230, 150, 140], [246, 198, 145], [255, 204,  54], [  0,   0,  90],
    [  0,   0, 110], [220, 190,  40], [180, 165, 180], [174,  64,  67],
    [150, 100, 100], [150, 120,  90], [136, 143, 153], [169, 187, 214],
    [ 81,   0,  81], [152, 251, 152],
], dtype=np.uint8)


# ── Cityscapes ────────────────────────────────────────────────────────────

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]

CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


# ── COCO (80 classes) ─────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _generate_palette(n: int) -> np.ndarray:
    """Generate n visually distinct colors using golden-angle hue spacing."""
    from PyQt6.QtGui import QColor
    palette = []
    for i in range(n):
        hue = int((i * 137.508) % 360)
        c   = QColor.fromHsv(hue, 200, 210)
        palette.append([c.red(), c.green(), c.blue()])
    return np.array(palette, dtype=np.uint8)


# ── Pascal VOC ────────────────────────────────────────────────────────────

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

VOC_PALETTE = np.array([
    [  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
    [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
    [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
    [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128], [192, 128, 128],
    [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
    [  0,  64, 128],
], dtype=np.uint8)


# ── ADE20K (150 classes, abbreviated) ────────────────────────────────────

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person",
    "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "storage tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower",
    "radiator", "glass", "clock", "flag",
]


# ── Registry ──────────────────────────────────────────────────────────────

def get_all_presets() -> list[ClassPreset]:
    return [
        ClassPreset("IDD",        IDD_CLASSES,        IDD_PALETTE),
        ClassPreset("Cityscapes", CITYSCAPES_CLASSES,  CITYSCAPES_PALETTE),
        ClassPreset("COCO",       COCO_CLASSES,
                    _generate_palette(len(COCO_CLASSES))),
        ClassPreset("Pascal VOC", VOC_CLASSES,         VOC_PALETTE),
        ClassPreset("ADE20K",     ADE20K_CLASSES,
                    _generate_palette(len(ADE20K_CLASSES))),
    ]


def preset_by_name(name: str) -> ClassPreset | None:
    return next((p for p in get_all_presets()
                 if p.name.lower() == name.lower()), None)


# ── Converters ────────────────────────────────────────────────────────────

def classes_from_preset(
        preset:   ClassPreset,
        start_id: int = 0) -> list:
    return classes_from_raw(preset.classes, preset.palette, start_id)


def classes_from_raw(
        names:    list[str],
        palette:  np.ndarray,
        start_id: int = 0) -> list:
    """
    Build a list of LabelClass from parallel names + Nx3 RGB palette.
    Falls back to golden-angle hue colors when palette is shorter than names.
    """
    from .annotations import LabelClass
    from PyQt6.QtGui import QColor

    result  = []
    n       = len(names)
    fallback = _generate_palette(n)

    for i, name in enumerate(names):
        if i < len(palette):
            r, g, b = int(palette[i][0]), int(palette[i][1]), int(palette[i][2])
        else:
            r, g, b = int(fallback[i][0]), int(fallback[i][1]), int(fallback[i][2])
        result.append(LabelClass(
            id    = start_id + i,
            name  = name,
            color = QColor(r, g, b),
        ))
    return result