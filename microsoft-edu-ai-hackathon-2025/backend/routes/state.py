"""GET /state — Returns full pipeline state for frontend restore on page load."""
import json
import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

state_bp = Blueprint("state", __name__)


@state_bp.route("/state", methods=["GET"])
def api_state():
    """Return current pipeline state so the frontend can restore after refresh."""
    from app import get_pipeline
    pipeline = get_pipeline()

    # Determine which phases are complete
    completed_phases: list[int] = []
    if pipeline.feature_spec:
        completed_phases.append(1)
    has_training = pipeline.training_X is not None and not pipeline.training_X.empty
    if has_training:
        completed_phases.append(2)
    if pipeline.is_trained:
        completed_phases.append(3)
    has_testing = pipeline.testing_X is not None and not pipeline.testing_X.empty
    if has_testing:
        completed_phases.append(4)
    has_predictions = bool(pipeline.predictions)
    if has_predictions:
        completed_phases.append(5)

    # Suggest next step
    suggested_step = (max(completed_phases) + 1) if completed_phases else 1
    suggested_step = min(suggested_step, 5)

    # Training data (can be large — include as JSON records)
    training_data_X = None
    if has_training:
        try:
            training_data_X = json.loads(
                pipeline.training_X.to_json(orient="records")
            )
        except Exception:
            pass

    testing_data_X = None
    if has_testing:
        try:
            testing_data_X = json.loads(
                pipeline.testing_X.to_json(orient="records")
            )
        except Exception:
            pass

    # Dataset Y columns (for target column selector)
    dataset_Y_columns = None
    if pipeline.training_Y_df is not None:
        dataset_Y_columns = list(pipeline.training_Y_df.columns)

    # Build train result (mirrors POST /train response shape)
    train_result = None
    if pipeline.is_trained:
        train_result = {
            "status": "success",
            "target_mode": pipeline.target_mode,
            "mse": pipeline.mse,
            "rulekit_mse": pipeline.rulekit_mse,
            "xgb_mse": pipeline.xgb_mse,
            "cv_mse": pipeline.cv_mse,
            "cv_std": pipeline.cv_std,
            "cv_mae": pipeline.cv_mae,
            "train_accuracy": pipeline.train_accuracy,
            "train_balanced_accuracy": pipeline.train_balanced_accuracy,
            "train_f1_macro": pipeline.train_f1_macro,
            "train_mcc": pipeline.train_mcc,
            "cv_accuracy": pipeline.cv_accuracy,
            "cv_balanced_accuracy": pipeline.cv_balanced_accuracy,
            "cv_f1_macro": pipeline.cv_f1_macro,
            "cv_precision_macro": pipeline.cv_precision_macro,
            "cv_recall_macro": pipeline.cv_recall_macro,
            "cv_mcc": pipeline.cv_mcc,
            "cv_folds": pipeline.cv_folds,
            "warnings": pipeline.warnings,
            "rules_count": len(pipeline.rules),
            "rules": pipeline.rules,
            "feature_spec": pipeline.feature_spec,
            "feature_importance": pipeline.feature_importance,
            "training_data_X": training_data_X,
        }

    return jsonify({
        "feature_spec": pipeline.feature_spec,
        "target_variable": pipeline.target_variable,
        "target_mode": pipeline.target_mode,
        "is_trained": pipeline.is_trained,
        "completed_phases": completed_phases,
        "suggested_step": suggested_step,
        "training_rows": len(pipeline.training_X) if has_training else 0,
        "testing_rows": len(pipeline.testing_X) if has_testing else 0,
        "training_data_X": training_data_X,
        "testing_data_X": testing_data_X,
        "dataset_Y_columns": dataset_Y_columns,
        "train_result": train_result,
        "predictions": pipeline.predictions,
        "prediction_metrics": pipeline.prediction_metrics,
    })
