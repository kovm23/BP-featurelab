"""Phase 3: /train endpoint (async)."""
import logging
import threading
import uuid

from flask import Blueprint, jsonify, request

from jobs import set_job, update_job

logger = logging.getLogger(__name__)

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def api_train():
    """Phase 3: Train RuleKit+XGBoost ensemble – async, returns job_id for polling."""
    from app import get_pipeline
    pipeline = get_pipeline()

    if request.is_json:
        target_col = (request.json or {}).get("target_column", "")
        target_mode = (request.json or {}).get("target_mode", pipeline.target_mode or "regression")
    else:
        target_col = request.form.get("target_column", "")
        target_mode = request.form.get("target_mode", pipeline.target_mode or "regression")

    target_mode = (target_mode or "regression").strip().lower()
    if target_mode not in ("regression", "classification"):
        target_mode = "regression"
    pipeline.target_mode = target_mode

    if not target_col:
        return jsonify({"error": "Missing target_column (name of the target column in the CSV)."}), 400

    pipeline.invalidate_from_phase(3)
    pipeline.target_mode = target_mode
    pipeline.save_state()

    job_id = str(uuid.uuid4())
    set_job(job_id, {"progress": 0, "stage": "Starting training...", "done": False})

    def _run():
        try:
            update_job(job_id, stage="Preprocessing features...", progress=10)
            result = pipeline.train_model(
                target_col,
                progress_cb=lambda pct, msg: update_job(job_id, progress=pct, stage=msg),
            )
            set_job(job_id, {
                "progress": 100,
                "stage": "Training complete!",
                "done": True,
                "details": result,
            })
        except Exception as e:
            logger.exception("Training failed: %s", e)
            set_job(job_id, {"progress": 100, "done": True, "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})
