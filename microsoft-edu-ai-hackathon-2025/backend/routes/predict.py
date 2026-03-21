"""Phase 5: /predict endpoint."""
import logging
import os

import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def api_predict():
    """Phase 5: Batch prediction for all objects in the testing dataset."""
    from app import pipeline

    testing_Y_df = None
    if request.files and "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(
                UPLOAD_FOLDER, f"test_labels_{secure_filename(lf.filename)}"
            )
            lf.save(labels_path)
            try:
                testing_Y_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning("Cannot load testing labels CSV: %s", e)
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    try:
        result = pipeline.predict_batch(testing_Y_df)
        return jsonify({
            "status": "success",
            "predictions": result["predictions"],
            "metrics": result["metrics"],
            "count": len(result["predictions"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
