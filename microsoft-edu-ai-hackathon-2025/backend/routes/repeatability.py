"""POST /repeatability-test — Run feature extraction N times on a single file.

Returns per-feature statistics (mean, std, CV%) to quantify LLM stochasticity.
Requires Phase 1 (feature discovery) to be completed first.
"""
import logging
import os
import statistics
import tempfile
import threading
import uuid
from collections import Counter

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

import jobs as job_registry
from config import UPLOAD_FOLDER
from pipeline.feature_extraction import _build_extraction_prompt, _extract_single_pass
from services.openai_service import DEFAULT_MODEL

logger = logging.getLogger(__name__)
repeatability_bp = Blueprint("repeatability", __name__)

_MAX_REPS = 10
_MIN_REPS = 2


def _compute_feature_stats(feature_name: str, values: list, feature_spec: dict) -> dict:
    """Return statistics for a single feature across N extraction runs."""
    spec_val = feature_spec.get(feature_name)
    is_numeric_spec = (
        isinstance(spec_val, list)
        and len(spec_val) == 2
        and all(isinstance(v, (int, float)) for v in spec_val)
    )

    numeric_vals = []
    for v in values:
        try:
            numeric_vals.append(float(v))
        except (ValueError, TypeError):
            pass

    if len(numeric_vals) >= max(1, len(values) * 0.5) or is_numeric_spec:
        mean_val = statistics.mean(numeric_vals) if numeric_vals else None
        std_val = statistics.stdev(numeric_vals) if len(numeric_vals) > 1 else 0.0
        cv_pct = (std_val / abs(mean_val) * 100) if mean_val else None
        return {
            "feature": feature_name,
            "type": "numeric",
            "values": [round(v, 4) for v in numeric_vals],
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "std": round(std_val, 4),
            "cv_pct": round(cv_pct, 2) if cv_pct is not None else None,
        }

    str_vals = [str(v) for v in values if v is not None]
    counter = Counter(str_vals)
    mode_val, mode_freq = counter.most_common(1)[0] if counter else (None, 0)
    return {
        "feature": feature_name,
        "type": "categorical",
        "values": str_vals,
        "mode": mode_val,
        "mode_frequency": mode_freq,
        "mode_frequency_pct": round(mode_freq / max(len(str_vals), 1) * 100, 1),
    }


@repeatability_bp.route("/repeatability-test", methods=["POST"])
def api_repeatability_test():
    """Start async repeatability test. Returns job_id for polling."""
    from app import get_pipeline
    pipeline = get_pipeline()

    feature_spec = pipeline.feature_spec
    if not feature_spec:
        return jsonify({"error": "Feature spec not set. Complete Phase 1 first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No media file uploaded."}), 400

    uploaded_file = request.files["file"]
    n_reps = int(request.form.get("n_repetitions", 5))
    n_reps = max(_MIN_REPS, min(n_reps, _MAX_REPS))
    model_name = request.form.get("model", DEFAULT_MODEL) or DEFAULT_MODEL

    safe_name = secure_filename(uploaded_file.filename or "sample_file")
    tmp_path = os.path.join(UPLOAD_FOLDER, f"repeat_{uuid.uuid4().hex[:8]}_{safe_name}")
    uploaded_file.save(tmp_path)

    job_id = str(uuid.uuid4())
    job_registry.set_job(job_id, {
        "progress": 0,
        "stage": f"Starting repeatability test ({n_reps} runs)...",
        "done": False,
    })

    session_feature_spec = dict(feature_spec)

    def _run():
        try:
            prompt = _build_extraction_prompt(session_feature_spec, "")
            per_feature_values: dict[str, list] = {k: [] for k in session_feature_spec}

            for rep in range(n_reps):
                job_registry.update_job(
                    job_id,
                    progress=int((rep / n_reps) * 90),
                    stage=f"Run {rep + 1} of {n_reps}...",
                )
                try:
                    attrs = _extract_single_pass(tmp_path, prompt, model_name)
                except Exception as e:
                    logger.warning("Repeatability run %d failed: %s", rep + 1, e)
                    attrs = {}
                for feat in session_feature_spec:
                    val = attrs.get(feat)
                    if val is not None:
                        per_feature_values[feat].append(val)

            feature_stats = [
                _compute_feature_stats(feat, vals, session_feature_spec)
                for feat, vals in per_feature_values.items()
            ]

            job_registry.set_job(job_id, {
                "progress": 100,
                "stage": "Done",
                "done": True,
                "details": {
                    "n_repetitions": n_reps,
                    "filename": safe_name,
                    "model": model_name,
                    "feature_stats": feature_stats,
                },
            })
        except Exception as e:
            logger.exception("Repeatability test failed: %s", e)
            job_registry.set_job(job_id, {
                "progress": 100,
                "stage": "Error",
                "done": True,
                "error": str(e),
            })
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})
