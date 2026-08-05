"""Security tests for ZIP extraction (Zip Slip / path traversal)."""
import os
import zipfile

import pytest

from utils.file_utils import extract_zip_contents


def _make_zip(zip_path: str, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_rejects_parent_traversal_entry(tmp_path):
    zip_path = os.path.join(tmp_path, "malicious.zip")
    extract_dir = os.path.join(tmp_path, "out")
    os.makedirs(extract_dir, exist_ok=True)
    _make_zip(zip_path, {"../escaped.txt": b"pwned"})

    with pytest.raises(ValueError):
        extract_zip_contents(zip_path, extract_dir)

    # Nothing must have been written outside the extraction directory.
    assert not os.path.exists(os.path.join(tmp_path, "escaped.txt"))


def test_rejects_absolute_path_entry(tmp_path):
    zip_path = os.path.join(tmp_path, "abs.zip")
    extract_dir = os.path.join(tmp_path, "out")
    os.makedirs(extract_dir, exist_ok=True)
    # Deep relative traversal that resolves outside the target tree.
    _make_zip(zip_path, {"a/../../../../evil.txt": b"pwned"})

    with pytest.raises(ValueError):
        extract_zip_contents(zip_path, extract_dir)


def test_extracts_benign_archive(tmp_path):
    zip_path = os.path.join(tmp_path, "ok.zip")
    extract_dir = os.path.join(tmp_path, "out")
    os.makedirs(extract_dir, exist_ok=True)
    _make_zip(zip_path, {"labels.csv": b"id,label\n1,a\n", "nested/clip.mp4": b"\x00\x00"})

    media_files, csv_path = extract_zip_contents(zip_path, extract_dir)

    assert os.path.exists(os.path.join(extract_dir, "labels.csv"))
    assert os.path.exists(os.path.join(extract_dir, "nested", "clip.mp4"))
    assert csv_path is not None and csv_path.endswith("labels.csv")
    assert any(p.endswith("clip.mp4") for p in media_files)
