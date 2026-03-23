"""Phase 5: /predict endpoint."""
import logging

from flask import Blueprint, jsonify, request

from config import UPLOAD_FOLDER
from utils.csv_utils import load_labels_from_request

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def api_predict():
    """Phase 5: Batch prediction for all objects in the testing dataset."""
    from app import pipeline

    testing_Y_df = load_labels_from_request(request, UPLOAD_FOLDER)

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
