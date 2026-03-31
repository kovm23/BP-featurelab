"""Phase 5: /predict endpoint (async)."""
import logging
import threading
import uuid

from flask import Blueprint, jsonify, request

from config import UPLOAD_FOLDER
from jobs import set_job, update_job
from utils.csv_utils import load_labels_from_request

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def api_predict():
    """Phase 5: Batch prediction – async, returns job_id for polling."""
    from app import get_pipeline
    pipeline = get_pipeline()

    testing_Y_df = load_labels_from_request(request, UPLOAD_FOLDER)

    job_id = str(uuid.uuid4())
    set_job(job_id, {"progress": 0, "stage": "Spouštím predikci...", "done": False})

    def _run():
        try:
            def _pcb(pct: int, msg: str) -> None:
                update_job(job_id, progress=pct, stage=msg)

            _pcb(5, "Připravuji predikci...")
            result = pipeline.predict_batch(testing_Y_df, progress_cb=_pcb)
            set_job(job_id, {
                "progress": 100,
                "stage": "Predikce dokončena!",
                "done": True,
                "details": {
                    "status": "success",
                    "predictions": result["predictions"],
                    "metrics": result["metrics"],
                    "count": len(result["predictions"]),
                },
            })
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            set_job(job_id, {"progress": 0, "done": True, "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})
