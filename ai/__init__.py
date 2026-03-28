"""
AI subpackage — SAM 2 + YOLO engines.

Sets ultralytics offline mode immediately on import so model
loading never hangs waiting for update checks or telemetry.
"""
import os

# Must be set before ultralytics is imported anywhere
os.environ["YOLO_OFFLINE"]        = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

# Also patch the settings object if ultralytics is already imported
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False, "api_key": ""})
except Exception:
    pass