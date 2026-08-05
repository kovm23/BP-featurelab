"""File handling utilities."""
import logging
import os
import zipfile

from config import ALL_ALLOWED
from services.processing import _is_media_file

logger = logging.getLogger(__name__)


def allowed_file(filename: str, allowed_exts: set | None = None) -> bool:
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in (allowed_exts or ALL_ALLOWED)


_MAX_UNZIPPED_BYTES = 5 * 1024 ** 3  # 5 GB
_S_IFLNK = 0o120000  # symlink mode bits (high 16 bits of external_attr)


def _is_within_directory(base_dir: str, target_path: str) -> bool:
    """Return True if *target_path* resolves to a location inside *base_dir*."""
    base = os.path.realpath(base_dir)
    target = os.path.realpath(target_path)
    return target == base or target.startswith(base + os.sep)


def _safe_extractall(zf: zipfile.ZipFile, extract_path: str) -> None:
    """Extract *zf* into *extract_path*, defending against Zip Slip.

    Rejects members whose resolved destination escapes *extract_path* (e.g.
    ``../../etc/passwd``) and symlink entries that could redirect later writes
    outside the target directory. Validation happens before any file is written.
    """
    for member in zf.infolist():
        if (member.external_attr >> 16) & 0o170000 == _S_IFLNK:
            raise ValueError(
                f"Refusing to extract symlink entry from ZIP: {member.filename!r}"
            )
        target = os.path.join(extract_path, member.filename)
        if not _is_within_directory(extract_path, target):
            raise ValueError(
                f"Unsafe path in ZIP archive (path traversal blocked): {member.filename!r}"
            )
    zf.extractall(extract_path)


def extract_zip_contents(zip_path: str, extract_path: str):
    """Extract a ZIP archive and return (media_files, csv_file_path_or_None)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        total_size = sum(m.file_size for m in zf.infolist())
        if total_size > _MAX_UNZIPPED_BYTES:
            raise ValueError(
                f"ZIP would exceed {_MAX_UNZIPPED_BYTES // 1024**3} GB when extracted "
                f"(estimated size: {total_size // 1024**3} GB)"
            )
        _safe_extractall(zf, extract_path)

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
