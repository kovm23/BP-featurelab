"""Phase 2 & 4: /extract and /extract-local endpoints."""
import json
import logging
import os
import shutil
import threading
import uuid

import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import DATASET_FOLDER, ALLOWED_EXTENSIONS
from utils.file_utils import allowed_file, extract_zip_contents
import jobs as job_registry

logger = logging.getLogger(__name__)

extract_bp = Blueprint("extract", __name__)


@extract_bp.route("/extract", methods=["POST"])
def api_extract():
    """Phase 2 & 4: Async feature extraction from a ZIP dataset (upload)."""
    from app import pipeline

    if "file" not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS["zip"]):
        return jsonify({"error": "Allowed format: .zip"}), 400

    model_name = request.form.get("model", "qwen2.5vl:7b")
    feature_spec = json.loads(request.form.get("feature_spec", "{}"))
    dataset_type = request.form.get("dataset_type", "training")

    if not feature_spec:
        return jsonify({"error": "Missing feature_spec."}), 400

    labels_df = None
    if "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(
                DATASET_FOLDER, f"labels_{secure_filename(lf.filename)}"
            )
            lf.save(labels_path)
            try:
                labels_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning("Cannot load labels CSV: %s", e)
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    zip_path = os.path.join(DATASET_FOLDER, secure_filename(file.filename))
    file.save(zip_path)

    job_id = str(uuid.uuid4())
    job_registry.set(job_id, {"progress": 0, "stage": "Starting extraction...", "done": False})

    extract_path = os.path.join(DATASET_FOLDER, f"extract_{job_id}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Cannot extract ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP contains no media files."}), 400

    def _run():
        try:
            pipeline.extract_features_async(
                media_files, feature_spec, job_id, model_name,
                dataset_type, csv_path, labels_df
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    threading.Thread(target=_run).start()
    return jsonify({"job_id": job_id, "media_count": len(media_files)})


@extract_bp.route("/extract-local", methods=["POST"])
def api_extract_local():
    """Phase 2 & 4: Async feature extraction from a ZIP already on the server."""
    from app import pipeline

    data = request.get_json()
    if not data:
        return jsonify({"error": "Expected a JSON body."}), 400

    zip_path = data.get("zip_path")
    model_name = data.get("model", "qwen2.5vl:7b")
    feature_spec = data.get("feature_spec", {})
    dataset_type = data.get("dataset_type", "training")
    labels_df = None

    if not zip_path or not os.path.exists(zip_path):
        return jsonify({"error": f"File not found: {zip_path}"}), 400
    if not feature_spec:
        return jsonify({"error": "Missing feature_spec."}), 400

    labels_path = data.get("labels_path")
    if labels_path and os.path.exists(labels_path):
        try:
            labels_df = pd.read_csv(labels_path)
        except Exception as e:
            logger.warning("Cannot load labels CSV: %s", e)

    job_id = str(uuid.uuid4())
    job_registry.set(job_id, {"progress": 0, "stage": "Starting extraction...", "done": False})

    extract_path = os.path.join(DATASET_FOLDER, f"extract_{job_id}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Cannot extract ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP contains no media files."}), 400

    def _run():
        try:
            pipeline.extract_features_async(
                media_files, feature_spec, job_id, model_name,
                dataset_type, csv_path, labels_df
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)

    threading.Thread(target=_run).start()
    return jsonify({"job_id": job_id, "media_count": len(media_files)})
