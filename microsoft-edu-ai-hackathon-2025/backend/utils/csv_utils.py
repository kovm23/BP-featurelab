"""CSV/labels loading utilities – shared across routes."""
import logging
import os

import pandas as pd
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


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
        return pd.read_csv(labels_path)
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
        return pd.read_csv(path)
    except Exception as e:
        logger.warning("Cannot load labels CSV from %s: %s", path, e)
        return None
