"""Phase 2 & 4: /extract and /extract-local endpoints."""
import json
import logging
import os
import shutil
import threading
import uuid

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import DATASET_FOLDER, UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from pipeline.feature_schema import normalize_feature_spec
from utils.file_utils import allowed_file, extract_zip_contents
from utils.csv_utils import load_labels_from_request, load_labels_from_path
import jobs as job_registry

logger = logging.getLogger(__name__)

extract_bp = Blueprint("extract", __name__)

# Directories where /extract-local may read files from.
_SAFE_PREFIXES = [
    os.path.realpath(UPLOAD_FOLDER),
    os.path.realpath(DATASET_FOLDER),
    os.path.realpath("/tmp"),
    os.path.realpath(os.path.expanduser("~")),
]


def _start_extraction(pipeline, media_files, feature_spec, model_name,
                       dataset_type, csv_path, labels_df, extract_path,
                       cleanup_paths=None):
    """Common helper: create a job and run extraction in a background thread."""
    job_id = str(uuid.uuid4())
    job_registry.set(job_id, {"progress": 0, "stage": "Starting extraction...", "done": False})

    def _run():
        try:
            pipeline.extract_features_async(
                media_files, feature_spec, job_id, model_name,
                dataset_type, csv_path, labels_df
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)
            for p in (cleanup_paths or []):
                try:
                    os.remove(p)
                except OSError:
                    pass

    threading.Thread(target=_run, daemon=False).start()
    return job_id


@extract_bp.route("/extract", methods=["POST"])
def api_extract():
    """Phase 2 & 4: Async feature extraction from a ZIP dataset (upload)."""
    from app import get_pipeline
    pipeline = get_pipeline()

    if "file" not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS["zip"]):
        return jsonify({"error": "Allowed format: .zip"}), 400

    model_name = request.form.get("model", "qwen2.5vl:7b")
    try:
        feature_spec = json.loads(request.form.get("feature_spec", "{}"))
    except (json.JSONDecodeError, TypeError):
        return jsonify({"error": "Invalid feature_spec JSON."}), 400
    feature_spec = normalize_feature_spec(feature_spec)
    dataset_type = request.form.get("dataset_type", "training")

    if not feature_spec:
        return jsonify({"error": "Missing feature_spec."}), 400

    labels_df = load_labels_from_request(request, DATASET_FOLDER)

    zip_path = os.path.join(DATASET_FOLDER, secure_filename(file.filename))
    file.save(zip_path)

    extract_path = os.path.join(DATASET_FOLDER, f"extract_{uuid.uuid4()}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Cannot extract ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP contains no media files."}), 400

    job_id = _start_extraction(
        pipeline, media_files, feature_spec, model_name,
        dataset_type, csv_path, labels_df, extract_path,
        cleanup_paths=[zip_path],
    )
    return jsonify({"job_id": job_id, "media_count": len(media_files)})


@extract_bp.route("/extract-local", methods=["POST"])
def api_extract_local():
    """Phase 2 & 4: Async feature extraction from a ZIP already on the server."""
    from app import get_pipeline
    pipeline = get_pipeline()

    data = request.get_json()
    if not data:
        return jsonify({"error": "Expected a JSON body."}), 400

    zip_path = data.get("zip_path")
    model_name = data.get("model", "qwen2.5vl:7b")
    feature_spec = data.get("feature_spec", {})
    feature_spec = normalize_feature_spec(feature_spec)
    dataset_type = data.get("dataset_type", "training")

    if not zip_path:
        return jsonify({"error": "Missing zip_path."}), 400

    # Prevent path traversal
    real_path = os.path.realpath(zip_path)
    if not any(real_path.startswith(prefix) for prefix in _SAFE_PREFIXES):
        return jsonify({"error": "Path not allowed."}), 403

    if not os.path.exists(zip_path):
        return jsonify({"error": f"File not found: {zip_path}"}), 400
    if not feature_spec:
        return jsonify({"error": "Missing feature_spec."}), 400

    raw_labels_path = data.get("labels_path", "")
    if raw_labels_path:
        real_labels = os.path.realpath(raw_labels_path)
        if not any(real_labels.startswith(prefix) for prefix in _SAFE_PREFIXES):
            return jsonify({"error": "labels_path not allowed."}), 403
    labels_df = load_labels_from_path(raw_labels_path)

    extract_path = os.path.join(DATASET_FOLDER, f"extract_{uuid.uuid4()}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Cannot extract ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP contains no media files."}), 400

    job_id = _start_extraction(
        pipeline, media_files, feature_spec, model_name,
        dataset_type, csv_path, labels_df, extract_path,
    )
    return jsonify({"job_id": job_id, "media_count": len(media_files)})
