"""Application-wide constants and configuration."""
import os

ALLOWED_EXTENSIONS = {
    "image": {"png", "jpg", "jpeg", "webp", "heic", "gif"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
MEDIA_EXTS = ALLOWED_EXTENSIONS["video"] | ALLOWED_EXTENSIONS["image"]
ALL_ALLOWED = MEDIA_EXTS | ALLOWED_EXTENSIONS["zip"]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_BASE_DIR, "uploads")
DATASET_FOLDER = os.path.join(_BASE_DIR, "dataset")
CHECKPOINT_FOLDER = os.path.join(_BASE_DIR, "checkpoints")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB
