"""Phase 1: /discover endpoint (async)."""
import logging
import os
import shutil
import threading
import uuid

import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from jobs import set_job, update_job
from utils.file_utils import allowed_file, extract_zip_contents

logger = logging.getLogger(__name__)

discover_bp = Blueprint("discover", __name__)


@discover_bp.route("/discover", methods=["POST"])
def api_discover():
    """Phase 1: Feature discovery from sample media (async, returns job_id)."""
    from app import pipeline

    uploaded = request.files.getlist("files")
    if not uploaded:
        uploaded = request.files.getlist("file")
    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    target_var = request.form.get("target_variable", "target value")
    model_name = request.form.get("model", "qwen2.5vl:7b")

    labels_df = None
    if "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(UPLOAD_FOLDER, f"labels_{secure_filename(lf.filename)}")
            lf.save(labels_path)
            try:
                labels_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning("Cannot load labels CSV: %s", e)
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    media_paths = []
    extract_path = None
    saved_paths = []

    for f in uploaded:
        if not allowed_file(f.filename):
            continue
        path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(path)
        saved_paths.append(path)

    if not saved_paths:
        return jsonify({"error": "No supported files found."}), 400

    for path in saved_paths:
        if path.lower().endswith(".zip"):
            extract_path = os.path.join(UPLOAD_FOLDER, f"discover_{uuid.uuid4().hex[:8]}")
            os.makedirs(extract_path, exist_ok=True)
            media_files, csv_in_zip = extract_zip_contents(path, extract_path)
            media_paths.extend(media_files)
            if labels_df is None and csv_in_zip:
                try:
                    labels_df = pd.read_csv(csv_in_zip)
                except Exception:
                    pass
        else:
            media_paths.append(path)

    if not media_paths:
        return jsonify({"error": "No media files to analyse."}), 400

    job_id = str(uuid.uuid4())
    set_job(job_id, {"progress": 0, "stage": "Starting discovery...", "done": False})

    def _run():
        try:
            update_job(job_id, stage="Analysing samples...", progress=10)
            features = pipeline.discover_features(media_paths, target_var, model_name, labels_df)
            set_job(job_id, {
                "progress": 100,
                "stage": "Discovery complete!",
                "done": True,
                "suggested_features": features,
            })
        except Exception as e:
            logger.exception("Discovery failed: %s", e)
            set_job(job_id, {"progress": 0, "done": True, "error": str(e)})
        finally:
            if extract_path:
                shutil.rmtree(extract_path, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})
