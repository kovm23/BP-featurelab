"""CSV/labels loading utilities – shared across routes."""
import logging
import os
import re

import pandas as pd
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Matches strings that look like filenames rather than column names:
# contains a digit, a path separator, or a known media extension.
_FILENAME_PATTERN = re.compile(r'\d|[/\\]|\.[a-zA-Z0-9]{2,4}$')


def _read_labels_csv(path: str) -> pd.DataFrame:
    """Read a labels CSV with automatic header detection.

    If the first column looks like a filename (contains digits, path separators
    or file extensions) rather than a descriptive column name, the file is
    re-read without a header row and generic column names are assigned.
    """
    df = pd.read_csv(path)
    if df.columns.size >= 1 and _FILENAME_PATTERN.search(str(df.columns[0])):
        logger.info("CSV '%s' appears to have no header row — re-reading with auto-assigned columns.", path)
        df = pd.read_csv(path, header=None)
        df.columns = pd.Index(["filename"] + [f"col_{i}" for i in range(1, len(df.columns))])
    return df


def load_labels_from_request(request, dest_folder: str) -> pd.DataFrame | None:
    """Load a labels CSV from the 'labels_file' field in a multipart request.

    Saves to *dest_folder* temporarily, reads into DataFrame, then deletes.
    Returns ``None`` when no file was provided or parsing failed.
    """
    if "labels_file" not in request.files:
        return None
    lf = request.files["labels_file"]
    if not lf.filename:
        return None

    labels_path = os.path.join(dest_folder, f"labels_{secure_filename(lf.filename)}")
    lf.save(labels_path)
    try:
        return _read_labels_csv(labels_path)
    except Exception as e:
        logger.warning("Cannot load labels CSV: %s", e)
        return None
    finally:
        if os.path.exists(labels_path):
            os.remove(labels_path)


def load_labels_from_path(path: str) -> pd.DataFrame | None:
    """Read a CSV from a local path. Returns ``None`` on failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        return _read_labels_csv(path)
    except Exception as e:
        logger.warning("Cannot load labels CSV from %s: %s", path, e)
        return None


def normalize_media_name(value: object) -> str:
    """Normalize media identifiers for tolerant joins.

    Examples:
    - ``video.mp4`` -> ``video``
    - ``folder/sub/Video 01.MOV`` -> ``video 01``
    - ``" sample.png "`` -> ``sample``
    """
    if value is None:
        return ""

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""

    text = text.replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(text)
    text = stem if ext else text
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()
