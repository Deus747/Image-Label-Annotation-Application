# run.py
import os
import sys

# Disable ultralytics network calls
os.environ["YOLO_OFFLINE"]        = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

# ── Resolve bundled weights path ──────────────────────────────────────────
# When packaged with PyInstaller, sys._MEIPASS points to the
# temporary folder where bundled files are extracted.
# We copy bundled weights to ~/.cache/ultralytics/ so ultralytics
# finds them in the standard location without any code changes.

def _setup_bundled_weights():
    if not getattr(sys, "frozen", False):
        return   # running from source, weights already in cache

    bundle_dir  = sys._MEIPASS
    weights_src = os.path.join(bundle_dir, "weights")

    if not os.path.isdir(weights_src):
        return

    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "ultralytics")
    os.makedirs(cache_dir, exist_ok=True)

    import shutil
    for filename in os.listdir(weights_src):
        if filename.endswith(".pt"):
            src  = os.path.join(weights_src, filename)
            dest = os.path.join(cache_dir, filename)
            if not os.path.exists(dest):
                print(f"Installing weight: {filename}")
                shutil.copy2(src, dest)


_setup_bundled_weights()

# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Pre-import torch with friendly error ──────────────────────────────────
try:
    import torch
except OSError as e:
    import tkinter.messagebox as mb
    mb.showerror(
        "PyTorch DLL Error",
        f"Failed to load PyTorch.\n\n"
        f"Fix: Install Visual C++ Redistributables from\n"
        f"https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
        f"Detail: {e}")
    sys.exit(1)

from labelapp.main import main
main()