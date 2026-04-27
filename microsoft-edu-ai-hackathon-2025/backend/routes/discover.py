"""Phase 1: /discover endpoint (async)."""
import logging
import os
import shutil
import threading
import uuid

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from jobs import set_job, update_job
from services.openai_service import DEFAULT_MODEL
from utils.file_utils import allowed_file, extract_zip_contents
from utils.csv_utils import load_labels_from_request

logger = logging.getLogger(__name__)

discover_bp = Blueprint("discover", __name__)


@discover_bp.route("/discover", methods=["POST"])
def api_discover():
    """Phase 1: Feature discovery from sample media (async, returns job_id)."""
    from app import get_pipeline
    pipeline = get_pipeline()

    uploaded = request.files.getlist("files") or request.files.getlist("file")
    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    target_var = request.form.get("target_variable", "target value")
    target_mode = (request.form.get("target_mode", "regression") or "regression").strip().lower()
    if target_mode not in ("regression", "classification"):
        target_mode = "regression"
    pipeline.invalidate_from_phase(1)
    pipeline.target_mode = target_mode
    pipeline.target_variable = target_var
    pipeline.save_state()
    model_name = request.form.get("model", DEFAULT_MODEL)
    llm_base_url = request.form.get("llm_base_url", "").strip()
    llm_api_key = request.form.get("llm_api_key", "").strip()
    labels_df = load_labels_from_request(request, UPLOAD_FOLDER)

    media_paths = []
    extract_path = None
    saved_paths = []

    for f in uploaded:
        if not allowed_file(f.filename):
            continue
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}")
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
                from utils.csv_utils import load_labels_from_path
                labels_df = load_labels_from_path(csv_in_zip)
        else:
            media_paths.append(path)

    if not media_paths:
        return jsonify({"error": "No media files to analyse."}), 400

    job_id = str(uuid.uuid4())
    set_job(job_id, {"progress": 0, "stage": "Starting discovery...", "done": False})

    def _run():
        try:
            def _pcb(pct: int, msg: str) -> None:
                update_job(job_id, progress=pct, stage=msg)

            _pcb(3, "Preparing files...")
            features = pipeline.discover_features(
                media_paths, target_var, model_name, labels_df, progress_cb=_pcb,
                llm_base_url=llm_base_url, llm_api_key=llm_api_key,
            )
            pipeline.save_state()
            set_job(job_id, {
                "progress": 100,
                "stage": "Discovery complete!",
                "done": True,
                "suggested_features": features,
            })
        except Exception as e:
            logger.exception("Discovery failed: %s", e)
            set_job(job_id, {"progress": 100, "done": True, "error": str(e)})
        finally:
            if extract_path:
                shutil.rmtree(extract_path, ignore_errors=True)
            for p in saved_paths:
                try:
                    os.remove(p)
                except OSError as e:
                    logger.debug("Failed to remove temp file %s: %s", p, e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})
