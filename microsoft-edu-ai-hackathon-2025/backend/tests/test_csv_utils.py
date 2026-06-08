"""Unit tests for utils.csv_utils."""
import pandas as pd

from utils import csv_utils
from utils.csv_utils import normalize_media_name


def test_drops_extension():
    assert normalize_media_name("video.mp4") == "video"


def test_casefolds():
    assert normalize_media_name("Sample.MOV") == "sample"


def test_strips_whitespace_and_quotes():
    assert normalize_media_name('  "Sample.png"  ') == "sample"


def test_keeps_path_basename_only():
    assert normalize_media_name("folder/sub/clip.mp4") == "clip"
    assert normalize_media_name("folder\\sub\\clip.mp4") == "clip"


def test_collapses_internal_whitespace():
    assert normalize_media_name("Video   01.mov") == "video 01"


def test_none_returns_empty_string():
    assert normalize_media_name(None) == ""


def test_non_string_stringifies():
    assert normalize_media_name(42) == "42"


def test_empty_string_returns_empty():
    assert normalize_media_name("") == ""
    assert normalize_media_name("   ") == ""


def test_no_extension():
    assert normalize_media_name("clip") == "clip"


def test_load_labels_from_xlsx_uses_read_excel(monkeypatch, tmp_path):
    xlsx_path = tmp_path / "labels.xlsx"
    xlsx_path.write_bytes(b"not a real xlsx; read_excel is patched")
    expected = pd.DataFrame({"media_name": ["clip"], "label": ["CZ"]})
    calls = []

    def fake_read_excel(path):
        calls.append(path)
        return expected

    monkeypatch.setattr(csv_utils.pd, "read_excel", fake_read_excel)

    result = csv_utils.load_labels_from_path(str(xlsx_path))

    assert calls == [str(xlsx_path)]
    pd.testing.assert_frame_equal(result, expected)
