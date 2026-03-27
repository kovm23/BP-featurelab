"""Phase 3: /train endpoint."""
from flask import Blueprint, jsonify, request

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def api_train():
    """Phase 3: Train RuleKit model from stored dataset_X + dataset_Y."""
    from app import get_pipeline
    pipeline = get_pipeline()

    if request.is_json:
        target_col = (request.json or {}).get("target_column", "")
    else:
        target_col = request.form.get("target_column", "")

    if not target_col:
        return jsonify({"error": "Missing target_column (name of the target column in the CSV)."}), 400

    try:
        result = pipeline.train_model(target_col)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
