"""Reset endpoint – clears checkpoints and pipeline state."""
import glob
import logging
import os

from flask import Blueprint, jsonify

from config import CHECKPOINT_FOLDER

logger = logging.getLogger(__name__)

reset_bp = Blueprint("reset", __name__)


@reset_bp.route("/reset", methods=["POST"])
def api_reset():
    """Delete all checkpoint files and reset pipeline state."""
    from app import pipeline

    removed = []
    for pattern in ("*.json", "*.pkl", "*.csv"):
        for f in glob.glob(os.path.join(CHECKPOINT_FOLDER, pattern)):
            try:
                os.remove(f)
                removed.append(os.path.basename(f))
            except OSError as e:
                logger.warning("Failed to remove %s: %s", f, e)

    # Reset pipeline in-memory state (must match model.py __init__)
    pipeline.feature_spec = {}
    pipeline.target_variable = ""
    pipeline.training_X = None
    pipeline.training_Y = None
    pipeline.training_Y_df = None
    pipeline.training_Y_column = ""
    pipeline.model = None
    pipeline.xgb_model = None
    pipeline.scaler = None
    pipeline.rules = []
    pipeline.mse = None
    pipeline.rulekit_mse = None
    pipeline.xgb_mse = None
    pipeline.cv_mse = None
    pipeline.cv_std = None
    pipeline.cv_mae = None
    pipeline.feature_importance = {}
    pipeline.is_trained = False
    pipeline.testing_X = None
    pipeline._training_columns = []
    pipeline._scaler_mean = []
    pipeline._scaler_scale = []
    pipeline.save_state()

    logger.info("Reset: removed checkpoints %s", removed)
    return jsonify({"ok": True, "removed_checkpoints": removed})
