"""POST /load-demo — Load a pre-built demo session fixture (DEMO_MODE only).

Enable by setting environment variable DEMO_MODE=true.
"""
import json
import logging
import os

import pandas as pd
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)
demo_bp = Blueprint("demo", __name__)

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
_FIXTURE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "demo_session.json")
)


@demo_bp.route("/load-demo", methods=["POST"])
def api_load_demo():
    """Load demo fixture into session. Requires DEMO_MODE=true env var."""
    if not DEMO_MODE:
        return jsonify({"error": "Demo mode is not enabled on this server."}), 403

    from app import get_pipeline
    pipeline = get_pipeline()

    try:
        with open(_FIXTURE_PATH, "r", encoding="utf-8") as f:
            demo = json.load(f)
    except FileNotFoundError:
        logger.error("Demo fixture not found at %s", _FIXTURE_PATH)
        return jsonify({"error": "Demo fixture file not found."}), 500

    pipeline.invalidate_from_phase(1)

    pipeline.feature_spec = demo.get("feature_spec", {})
    pipeline.target_variable = demo.get("target_variable", "arousal_category")
    pipeline.target_mode = demo.get("target_mode", "classification")
    pipeline.training_Y_column = demo.get("training_Y_column", pipeline.target_variable)

    pipeline.is_trained = demo.get("is_trained", True)
    pipeline.train_accuracy = demo.get("train_accuracy")
    pipeline.train_balanced_accuracy = demo.get("train_balanced_accuracy")
    pipeline.train_f1_macro = demo.get("train_f1_macro")
    pipeline.train_mcc = demo.get("train_mcc")
    pipeline.train_baseline_accuracy = demo.get("train_baseline_accuracy")
    pipeline.train_majority_class = demo.get("train_majority_class")

    pipeline.cv_accuracy = demo.get("cv_accuracy")
    pipeline.cv_balanced_accuracy = demo.get("cv_balanced_accuracy")
    pipeline.cv_f1_macro = demo.get("cv_f1_macro")
    pipeline.cv_precision_macro = demo.get("cv_precision_macro")
    pipeline.cv_recall_macro = demo.get("cv_recall_macro")
    pipeline.cv_mcc = demo.get("cv_mcc")
    pipeline.cv_folds = demo.get("cv_folds")

    pipeline.rules = demo.get("rules", [])
    pipeline.feature_importance = demo.get("feature_importance", {})
    pipeline.warnings = demo.get("warnings", [])

    pipeline._label_classes = demo.get("_label_classes", [])
    pipeline._positive_label = demo.get("_positive_label")
    pipeline._training_columns = demo.get("_training_columns", list(demo.get("feature_spec", {}).keys()))
    pipeline._scaler_mean = demo.get("_scaler_mean", [])
    pipeline._scaler_scale = demo.get("_scaler_scale", [])

    pipeline.predictions = demo.get("predictions")
    pipeline.prediction_metrics = demo.get("prediction_metrics")

    training_rows = demo.get("training_X", [])
    pipeline.training_X = pd.DataFrame(training_rows) if training_rows else None

    testing_rows = demo.get("testing_X", [])
    pipeline.testing_X = pd.DataFrame(testing_rows) if testing_rows else None

    if pipeline.training_X is not None and pipeline.target_variable:
        pipeline.training_Y_df = pd.DataFrame({
            "media_name": pipeline.training_X["media_name"].tolist() if "media_name" in pipeline.training_X.columns else [],
            pipeline.target_variable: (
                ["1"] * 5 + ["2"] * 7 + ["3"] * 6 + ["4"] * 6
            )[:len(pipeline.training_X)],
        })

    logger.info("Demo session loaded for session.")
    return jsonify({"ok": True, "message": "Demo session loaded."})
