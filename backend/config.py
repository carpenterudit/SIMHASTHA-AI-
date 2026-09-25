from pathlib import Path

API_PREFIX = "/api"
SYSTEM_NAME = "Simhastha AI"
BACKEND_ROOT = Path(__file__).resolve().parent
DATA_ROOT = BACKEND_ROOT / "data"
UPLOADS_DIR = DATA_ROOT / "uploads"
PROCESSED_DIR = DATA_ROOT / "processed"
RESULTS_DIR = DATA_ROOT / "results"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_CONFIDENCE = 0.40
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_INPUT_SIZE = 640
DEFAULT_FRAME_SKIP = 1

for directory in (UPLOADS_DIR, PROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
