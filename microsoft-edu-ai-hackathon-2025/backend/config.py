"""Application-wide constants and configuration."""
import os

ALLOWED_EXTENSIONS = {
    "image": {"png", "jpg", "jpeg", "webp", "heic", "gif"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
MEDIA_EXTS = ALLOWED_EXTENSIONS["video"] | ALLOWED_EXTENSIONS["image"]
ALL_ALLOWED = MEDIA_EXTS | ALLOWED_EXTENSIONS["zip"]

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
CHECKPOINT_FOLDER = "checkpoints"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
