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

# Maximum accepted upload size. Defaults to 2 GB (large video ZIPs); operators
# can lower it via MAX_CONTENT_LENGTH_MB to reduce disk-fill / DoS exposure.
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "2048")) * 1024 * 1024

# How many media samples Phase 1 (Feature Discovery) analyses before synthesising the feature spec.
# More samples → better feature spec, but longer discovery time.
DISCOVERY_MAX_SAMPLES = int(os.getenv("DISCOVERY_MAX_SAMPLES", "10"))

# Ollama client timeouts (seconds)
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5.0"))
