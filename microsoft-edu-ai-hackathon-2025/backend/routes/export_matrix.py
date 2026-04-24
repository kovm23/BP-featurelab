"""GET /export-confusion-matrix — Render confusion matrix as a PNG via matplotlib."""
import io
import logging

import numpy as np
from flask import Blueprint, jsonify, send_file

logger = logging.getLogger(__name__)
export_matrix_bp = Blueprint("export_matrix", __name__)


@export_matrix_bp.route("/export-confusion-matrix", methods=["GET"])
def export_confusion_matrix():
    """Return confusion matrix from the current session's prediction metrics as PNG."""
    from app import get_pipeline
    pipeline = get_pipeline()

    metrics = pipeline.prediction_metrics
    if not metrics or "confusion_matrix" not in metrics or "labels" not in metrics:
        return jsonify({"error": "No confusion matrix available. Run Phase 5 first."}), 400

    labels = metrics["labels"]
    cm = np.array(metrics["confusion_matrix"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return jsonify({"error": "matplotlib is not installed on the server."}), 500

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n)))

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=11,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="confusion_matrix.png",
    )
