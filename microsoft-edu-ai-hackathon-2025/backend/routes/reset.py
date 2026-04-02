"""Reset endpoint – clears checkpoints and pipeline state."""
import glob
import logging
import os

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

reset_bp = Blueprint("reset", __name__)


@reset_bp.route("/reset", methods=["POST"])
def api_reset():
    """Delete all checkpoint files and reset pipeline state."""
    from app import get_pipeline
    pipeline = get_pipeline()

    removed = []
    for pattern in ("*.json", "*.pkl", "*.csv"):
        for f in glob.glob(os.path.join(pipeline._checkpoint_folder, pattern)):
            try:
                os.remove(f)
                removed.append(os.path.basename(f))
            except OSError as e:
                logger.warning("Failed to remove %s: %s", f, e)

    # Reset pipeline in-memory state (must match model.py __init__)
    pipeline.invalidate_from_phase(1)
    pipeline.target_variable = ""
    pipeline.target_mode = "regression"
    pipeline.save_state()

    logger.info("Reset: removed checkpoints %s", removed)
    return jsonify({"ok": True, "removed_checkpoints": removed})
