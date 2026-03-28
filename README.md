# Image-Label-Annotation-Application

A Roboflow-style image annotation tool for Windows built with Python and PyQt6. Supports bounding box detection, polygon and brush-based segmentation, AI-assisted labeling via SAM 2 and YOLO, and exports to all major training formats.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-6.x-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Annotation tools](#annotation-tools)
- [AI-assisted labeling](#ai-assisted-labeling)
- [Mask preview](#mask-preview)
- [Export formats](#export-formats)
- [Session management](#session-management)
- [Class presets](#class-presets)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [License](#license)

---

## Features

### Image viewer
- Smooth zoom (5% to 4000%) via scroll wheel or `+` / `-`
- Pan via middle-click drag or Space + left-click drag
- Fit-to-view with `F`
- Thumbnail sidebar with lazy background loading — handles thousands of images without freezing
- Keyboard image navigation with arrow keys or `A` / `D`
- Live cursor coordinates in status bar
- Zoom slider in status bar

### Annotation tools
- **Bounding box** — click and drag, normalized YOLO-format storage
- **Polygon** — click-by-click point placement, right-click or double-click to close
- **Freehand** — draw and release, auto-simplified with Douglas-Peucker via OpenCV
- **Brush** — paint filled regions, erase overflows, commit to smooth polygon on Enter
  - 1px minimum brush size
  - Adjustable opacity
  - Eraser mode toggle
  - Gaussian-smoothed contour extraction on commit
- **Mask eraser** — double-click any existing polygon to load it into the brush overlay and paint-erase parts of it, then commit the corrected shape
- **Select and move** — drag bounding boxes and polygons to reposition, coordinates written back to the annotation store on release
- Full undo/redo stack (`Ctrl+Z` / `Ctrl+Y`)
- Per-class color coding with editable class manager
- Live mask preview dock (semantic or instance, grayscale or color)
- Export to YOLO, COCO JSON, Pascal VOC, and PNG masks

### AI-assisted labeling
- **SAM 2 point mode** — left-click to add a foreground point, Shift+click for background, SAM segments the object
- **SAM 2 box mode** — draw a box around an object, SAM segments the best match inside
- **YOLO auto-detection** — load any `.pt` model, run inference on the current image, detections appear as suggestions
- Suggestions shown as dashed cyan (SAM) or amber (YOLO) overlays
- Accept all suggestions with Enter or the Accept all button
- Reject all with Escape or the Reject all button
- SAM loads in a background thread with animated spinner and staged status messages
- Supports auto-download via ultralytics cache or manual `.pt` file selection
- Model combo shows ✔ cached / ⬇ needs download status for each variant
- Device auto-detection: OpenVINO GPU → OpenVINO CPU → CUDA → CPU

---

## Project structure

```
labelapp/
├── __init__.py
├── annotations.py      # Data models, AnnotationStore, undo commands, ToolMode
├── canvas.py           # Phase 2 — QGraphicsView image viewer, zoom, pan
├── constants.py        # Colors, sizes, defaults
├── brush.py            # BrushOverlay, BrushSettingsWidget, smooth contour extraction
├── tools.py            # Phase 3+4 — DrawingCanvas with all annotation tools
├── mask.py             # MaskGenerator, MaskPreviewWidget
├── widgets.py          # ThumbnailStrip, ZoomWidget, ClassManager, ToolbarWidget, PresetImportDialog
├── export.py           # YOLO, COCO, VOC, mask PNG export
├── session.py          # Session save/load with zlib compression and PNG-encoded brush masks
├── presets.py          # Built-in class presets: IDD, Cityscapes, COCO, Pascal VOC, ADE20K
├── main.py             # MainWindow, application-level shortcuts, AI signal bridges
└── ai/
    ├── __init__.py     # Sets YOLO_OFFLINE env vars before any ultralytics import
    ├── suggestion.py   # Suggestion dataclass (pending AI annotation)
    ├── sam_engine.py   # SamEngine — SAM 2 wrapper with background thread loading
    ├── yolo_engine.py  # YoloEngine — YOLO inference wrapper
    └── ai_toolbar.py   # AiToolbar dock widget with spinner, model status, mode buttons
```

---

## Installation

### Requirements

- Python 3.11+
- Windows 10/11 (developed and tested on Windows)
- Visual C++ Redistributables 2015-2022 ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe))

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/labelapp.git
cd labelapp

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
PyQt6
opencv-python
Pillow
numpy
ultralytics
openvino
torch
torchvision
```

> **NVIDIA GPU users:** Install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org) before running pip install.

---

## Quick start

```bash
# Activate venv
venv\Scripts\activate

# Run the app
python -m labelapp.run
```


### First steps

1. Click **📂 Open Folder** and select a folder of images
2. Images appear in the thumbnail strip on the left — click any to load it
3. Select a tool from the toolbar or press its shortcut key
4. Draw annotations on the canvas
5. Press `Ctrl+S` to save your session

---

## Annotation tools

### Bounding box `B`
Click and drag to draw a box. The box is stored in normalized YOLO format (x_center, y_center, width, height, all 0–1).

### Polygon `P`
Click to place points one by one. Right-click or double-click to close the polygon. Press Escape to cancel.

### Freehand `F2`
Click and drag to draw a free-form outline. Release the mouse to auto-simplify with Douglas-Peucker and commit.

### Brush `G`
Paint a filled region over the object. The brush overlay appears in the active class color.

- Use the **Brush Settings** dock (appears automatically) to adjust size (1–120px) and opacity
- Press `E` to toggle the eraser and remove overflow areas
- Press **Enter** to convert the painted region into a smooth polygon annotation
- Click **🗑 Clear brush mask** to start over

### Mask eraser (edit existing annotations)
1. Switch to **Select** tool (`V`)
2. **Double-click** any existing polygon or SAM mask
3. The annotation loads into the brush overlay with the eraser active
4. Paint over the parts you want to remove
5. Press **Enter** to commit the corrected polygon, or **Escape** to cancel

### Select and move `V`
Click any annotation to select it (highlighted outline). Drag to reposition. The new position is written back to the store on mouse release.

---

## AI-assisted labeling

### Loading SAM 2

1. Open the **AI Tools** dock on the left
2. Select a model variant from the dropdown:
   - SAM2-tiny (~40 MB) — fastest
   - SAM2-small (~180 MB) — recommended
   - SAM2-base (~310 MB)
   - SAM2-large (~900 MB) — most accurate
3. Click **⬇ Load / download** — weights download automatically on first run and are cached in `~/.cache/ultralytics/`
4. Or click **📁 Browse .pt…** to load a file you already have on disk
5. An animated spinner shows loading progress with staged status messages

### SAM 2 point mode `S`
1. Click **● Point** in the AI Tools dock or press `S`
2. Left-click the object to add a foreground point (cyan dot)
3. Shift+click to add a background point (red dot) to refine the mask
4. The segmentation suggestion appears as a cyan dashed overlay
5. Press **Enter** to accept, **Escape** to reject

### SAM 2 box mode `X`
1. Click **⬜ Box** in the AI Tools dock or press `X`
2. Draw a box around the object
3. SAM segments the best match inside the box
4. Press **Enter** to accept, **Escape** to reject

### YOLO auto-detection
1. Click **Browse .pt…** in the YOLO section and select a YOLO model file
2. Adjust the **Confidence** slider (default 0.25)
3. Click **▶ Run YOLO on image**
4. All detections appear as amber dashed overlays with confidence scores
5. Click **✔ Accept all** or press `Ctrl+A` to commit all suggestions
6. Click **✕ Reject all** or press `Ctrl+R` to discard them

### Refining AI masks with the brush eraser
```
SAM point click → cyan mask appears
→ Enter to accept → becomes polygon annotation
→ Double-click the polygon → enters brush edit mode
→ Erase the parts SAM got wrong
→ Enter to commit the correction
```

---

## Mask preview

The **Mask Preview** dock updates in real time as you annotate.

| Setting | Options | Description |
|---|---|---|
| Type | Semantic / Instance | Semantic: same class = same color. Instance: each annotation gets a unique color |
| Format | Color / Grayscale | Color: RGB class/instance colors. Grayscale: pixel value = class_id or instance index |
| Opacity | 20–100% | Controls overlay transparency in the preview |

Click **💾 Save mask PNG…** to save the current preview as a PNG file. When saving grayscale masks you can choose between raw label values (for training) or brightness-scaled values (for visualization).

---

## Export formats

Access via the **Export** dock (tabbed with Mask Preview).

| Format | Description |
|---|---|
| **YOLO .txt** | One file per image, normalized `class x_center y_center width height` |
| **COCO JSON** | Instance segmentation with polygon `segmentation` arrays and bounding boxes |
| **Mask PNG (current)** | PNG mask for the currently displayed image |
| **Mask PNG (all)** | Batch export masks for every annotated image in the session |

---

## Session management

LabelApp automatically saves your session every time you switch images. You can also save, load, and clear sessions manually.

| Action | How |
|---|---|
| Auto-save | Happens on every image switch |
| Manual save | `Ctrl+S` or **💾 Save Session** button |
| Load session | **📂 Load Session** button — restores folder, classes, all annotations, and brush masks |
| Clear session | **🗑 Clear Session** button |

### What is saved
- Last opened folder path
- All label classes with their names and colors
- All annotations (bounding boxes and polygons) for every image
- Brush mask overlays (stored as base64-encoded PNG, ~100× smaller than raw arrays)
- Last active image and class selection


---

## Class presets

Click **⬇ Import preset…** in the Classes panel to open the preset dialog.



### Custom import
Paste class names (one per line or as a Python list) and an optional RGB palette in the **Paste names + palette** dialog:

```python
# Names — one per line or Python list
road
sidewalk
building

# Palette — list of [R, G, B] rows (optional, auto-generated if omitted)
[[128, 64, 128], [244, 35, 232], [70, 70, 70]]
```

### Programmatic injection
```python
from labelapp.presets import classes_from_raw
import numpy as np

MY_CLASSES = ["cat", "dog", "bird"]
MY_PALETTE = np.array([[255,0,0],[0,255,0],[0,0,255]], dtype=np.uint8)

my_classes = classes_from_raw(MY_CLASSES, MY_PALETTE)
```

---

## Keyboard shortcuts

All shortcuts work application-wide regardless of which panel has focus.

### Tools

| Key | Tool |
|---|---|
| `V` | Select / move |
| `B` | Bounding box |
| `P` | Polygon |
| `F2` | Freehand |
| `G` | Brush |
| `S` | SAM 2 point |
| `X` | SAM 2 box |

### Canvas

| Key | Action |
|---|---|
| `F` | Fit image to view |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `Scroll wheel` | Zoom in / out |
| `Space + drag` | Pan canvas |
| `Middle-click drag` | Pan canvas |

### Annotation

| Key | Action |
|---|---|
| `Enter` | Accept suggestions / commit brush / close polygon |
| `Escape` | Reject suggestions / abort drawing |
| `Delete` | Delete selected annotation |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `E` | Toggle eraser (brush mode only) |
| `Double-click annotation` | Enter mask edit mode |

### Navigation and session

| Key | Action |
|---|---|
| `→` / `D` | Next image |
| `←` / `A` | Previous image |
| `Ctrl+S` | Save session |
| `Ctrl+A` | Accept all AI suggestions |
| `Ctrl+R` | Reject all AI suggestions |

---