"""File handling utilities."""
import logging
import os
import zipfile

from services.processing import _is_media_file
from config import ALL_ALLOWED

logger = logging.getLogger(__name__)


def allowed_file(filename: str, allowed_exts: set | None = None) -> bool:
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in (allowed_exts or ALL_ALLOWED)


def extract_zip_contents(zip_path: str, extract_path: str):
    """Extract a ZIP archive and return (media_files, csv_file_path_or_None)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    csv_file = None
    media_files = []
    for root, _dirs, files in os.walk(extract_path):
        for f in files:
            if f.startswith("._") or "__MACOSX" in root:
                continue
            full = os.path.join(root, f)
            logger.info("ZIP contents: %s", full)
            if f.lower().endswith((".csv", ".xlsx")):
                csv_file = full
            elif _is_media_file(full):
                media_files.append(full)
    logger.info("Found %d media files, CSV: %s", len(media_files), csv_file)
    return media_files, csv_file
