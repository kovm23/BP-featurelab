"""Phase 2 / 4: Async feature extraction from a list of media files.

Supports multi-pass extraction (configurable via EXTRACTION_PASSES env var)
with median aggregation for numeric features and majority vote for categorical.
Includes feature validation and clamping based on feature_spec descriptions.
"""
import json
import logging
import os
import statistics
from collections import Counter

import numpy as np
import pandas as pd

from pipeline.feature_validation import validate_row
from services.processing import process_single_media
from utils.target_context import build_labels_context
import jobs as job_registry

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning", "observation", "description"}

# For UI/user acceptance testing we default to 1 pass (faster).
# Production can raise this via EXTRACTION_PASSES env.
EXTRACTION_PASSES = int(os.environ.get("EXTRACTION_PASSES", "1"))


def _format_feature_constraint(spec_value) -> str:
    """Render a feature schema value into explicit prompt instructions."""
    if isinstance(spec_value, list):
        if len(spec_value) == 2 and all(isinstance(v, (int, float)) for v in spec_value):
            lo, hi = spec_value
            if lo > hi:
                lo, hi = hi, lo
            value_kind = "integer" if float(lo).is_integer() and float(hi).is_integer() else "number"
            return f"{value_kind} in [{lo}, {hi}]"

        if spec_value and all(isinstance(v, str) for v in spec_value):
            categories = json.dumps(spec_value, ensure_ascii=False)
            return f"one of {categories}"

        return f"match schema {json.dumps(spec_value, ensure_ascii=False)}"

    return str(spec_value)


def _build_extraction_prompt(feature_spec: dict, labels_context: str) -> str:
    """Build a chain-of-thought extraction prompt with explicit range enforcement."""
    # Build per-feature spec with explicit ranges
    feature_lines = []
    for name, desc in feature_spec.items():
        feature_lines.append(f"  - {name}: {_format_feature_constraint(desc)}")
    features_block = "\n".join(feature_lines)

    prompt = (
        "You are a feature extraction AI analyzing media content.\n\n"
        "First, briefly describe what you observe in this media "
        "(2-3 sentences about visual content, audio, mood, pacing).\n\n"
        "Then extract EXACTLY these features:\n"
        f"{features_block}\n\n"
    )

    if labels_context:
        prompt += f"{labels_context}\n"

    prompt += (
        "Output format: First your brief observation (2-3 sentences), "
        "then on a new line output ONLY a valid JSON object with the exact keys "
        "listed above. Each value MUST respect the specified range/type. "
        "No extra keys, no markdown formatting."
    )

    return prompt


def _aggregate_passes(pass_results: list[dict], feature_spec: dict) -> dict:
    """Aggregate multiple extraction passes into a single result row.

    Numeric features: median of valid values.
    Categorical features: majority vote.
    """
    aggregated = {}

    for feat_name in feature_spec:
        values = [p.get(feat_name) for p in pass_results if p.get(feat_name) is not None]

        if not values:
            aggregated[feat_name] = None
            continue

        # Try numeric aggregation
        numeric_vals = []
        for v in values:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_vals:
            aggregated[feat_name] = round(statistics.median(numeric_vals), 4)
        else:
            # Categorical: majority vote
            counter = Counter(str(v) for v in values)
            aggregated[feat_name] = counter.most_common(1)[0][0]

    return aggregated


def _extract_single_pass(media_path: str, prompt: str, model_name: str) -> dict:
    """Run a single extraction pass and return cleaned attrs dict."""
    result = process_single_media(media_path, prompt=prompt, model_name=model_name)
    analysis = result.get("analysis", {})
    if isinstance(analysis, dict):
        attrs = analysis.get("attributes", analysis)
        for key in _META_KEYS:
            attrs.pop(key, None)
    else:
        attrs = {}
    return attrs


def extract_features_async(
    pipeline,
    media_files: list[str],
    feature_spec: dict,
    job_id: str,
    model_name: str,
    dataset_type: str,
    csv_path: str | None = None,
    labels_df: pd.DataFrame | None = None,
) -> None:
    """Extract features from media files according to feature_spec.

    Runs inside a background thread. Updates the job registry with progress
    and writes results into pipeline.training_X / pipeline.testing_X.

    Uses multi-pass extraction (EXTRACTION_PASSES) with median aggregation
    and validates/clamps values against feature_spec ranges.
    """
    try:
        pipeline.feature_spec = feature_spec
        total = len(media_files)
        n_passes = max(1, EXTRACTION_PASSES)

        # Build labels context for calibration
        target_mode = getattr(pipeline, "target_mode", "regression")
        labels_context = build_labels_context(labels_df, pipeline.target_variable, target_mode).strip()
        if labels_context:
            labels_context = labels_context.replace("\n\n", "\n").strip()

        prompt = _build_extraction_prompt(feature_spec, labels_context)

        # Checkpoint for resume
        checkpoint_file = os.path.join(pipeline._checkpoint_folder, f"extract_{dataset_type}.json")
        features_data = []
        done_names: set[str] = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as cf:
                    features_data = json.load(cf)
                done_names = {row["media_name"] for row in features_data}
                logger.info("Resume: loaded %d records from checkpoint.", len(features_data))
            except Exception as e:
                logger.warning("Cannot load checkpoint: %s", e)
                features_data = []

        job_registry.set_job(job_id, {
            "progress": 5,
            "stage": f"Starting extraction ({n_passes} pass/es)... "
                     f"({len(done_names)} already done)",
            "done": False,
        })

        total_clamped = 0

        for i, media_path in enumerate(media_files):
            file_name = os.path.basename(media_path)
            media_name = os.path.splitext(file_name)[0]

            if media_name in done_names:
                progress = 5 + int((i / total) * 90)
                job_registry.set_job(job_id, {
                    "progress": progress,
                    "stage": f"Skipping ({i + 1}/{total}): {file_name} (already done)",
                    "done": False,
                })
                continue

            progress = 5 + int((i / total) * 90)

            # Multi-pass extraction
            pass_results = []
            for p in range(n_passes):
                pass_label = f" [pass {p + 1}/{n_passes}]" if n_passes > 1 else ""
                job_registry.set_job(job_id, {
                    "progress": progress,
                    "stage": f"Extracting ({i + 1}/{total}): {file_name}{pass_label}",
                    "done": False,
                })
                attrs = _extract_single_pass(media_path, prompt, model_name)
                if attrs:
                    pass_results.append(attrs)

            # Aggregate passes
            if pass_results:
                if len(pass_results) == 1:
                    aggregated = pass_results[0]
                else:
                    aggregated = _aggregate_passes(pass_results, feature_spec)
                    logger.info(
                        "%s: aggregated %d passes", file_name, len(pass_results)
                    )
            else:
                aggregated = {}

            # Validate and clamp
            validated, stats = validate_row(aggregated, feature_spec)
            if stats["clamped_count"] > 0:
                total_clamped += stats["clamped_count"]
                logger.info(
                    "%s: clamped %d features: %s",
                    file_name, stats["clamped_count"], stats["clamped_features"],
                )

            # Build row
            row = {"media_name": media_name}
            missing = []
            for feat_key in feature_spec:
                val = validated.get(feat_key)
                if val is None:
                    missing.append(feat_key)
                    # Leave as None — will be imputed after all rows are collected
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val) if val else ""
                elif isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False)
                row[feat_key] = val
            if missing:
                logger.warning("%s: missing features %s", file_name, missing)
            features_data.append(row)

            # Checkpoint
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as cf:
                    json.dump(features_data, cf, ensure_ascii=False)
            except Exception as e:
                logger.warning("Cannot write checkpoint: %s", e)

        df_X = pd.DataFrame(features_data)

        # Impute missing cells immediately so the UI never shows empty rows.
        # We use column-median computed from this extraction batch (no train/test
        # leakage because each batch is imputed independently). The training
        # pipeline still re-imputes per CV fold for methodological correctness.
        feature_cols = [c for c in df_X.columns if c != "media_name"]
        missing_cells = int(df_X[feature_cols].isna().sum().sum())
        if missing_cells > 0:
            logger.info("Imputing %d missing feature cells with column medians.", missing_cells)
            for col in feature_cols:
                numeric_series = pd.to_numeric(df_X[col], errors="coerce")
                col_median = numeric_series.median()  # NaN if all missing
                if pd.isna(col_median):
                    col_median = 0.0
                df_X[col] = numeric_series.fillna(col_median)

        logger.info(
            "Extraction complete: %d rows, %d total clamped values",
            len(df_X), total_clamped,
        )

        df_Y = None
        if csv_path:
            try:
                df_Y = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning("Cannot load CSV labels: %s", e)

        if dataset_type == "training":
            pipeline.training_X = df_X
            if df_Y is not None:
                pipeline.training_Y_df = df_Y
        elif dataset_type == "testing":
            pipeline.testing_X = df_X

        pipeline.save_state()
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        except Exception:
            pass

        result_payload = {
            "progress": 100,
            "stage": "Extraction complete!",
            "done": True,
            "details": {
                "status": "success",
                "dataset_type": dataset_type,
                "dataset_X": json.loads(df_X.to_json(orient="records")),
                "feature_spec": feature_spec,
                "rows_count": len(df_X),
                "clamped_count": total_clamped,
            },
        }
        if df_Y is not None:
            result_payload["details"]["dataset_Y_columns"] = list(df_Y.columns)
        job_registry.set_job(job_id, result_payload)

    except Exception as e:
        logger.exception("Error during extraction job %s", job_id)
        job_registry.set_job(job_id, {"progress": 100, "stage": "Error", "done": True, "error": str(e)})
