"""Phase 2 / 4: Async feature extraction from a list of media files."""
import json
import logging
import os

import pandas as pd

from config import CHECKPOINT_FOLDER
from services.processing import process_single_media
import jobs as job_registry

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning"}


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

    Runs inside a background thread.  Updates the job registry with progress
    and writes results into pipeline.training_X / pipeline.testing_X.
    """
    try:
        pipeline.feature_spec = feature_spec
        spec_string = json.dumps(feature_spec, ensure_ascii=False)
        total = len(media_files)

        labels_context = ""
        if labels_df is not None:
            numeric_cols = labels_df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
                labels_context = (
                    f"\nThe target variable '{target_col}' has range "
                    f"[{labels_df[target_col].min()}, {labels_df[target_col].max()}]. "
                    f"Use this context to calibrate your feature value estimates.\n"
                )

        prompt = (
            f"You are a feature extraction AI.\n"
            f"Extract EXACTLY these features from the provided media:\n{spec_string}\n\n"
            f"{labels_context}"
            f"Output STRICTLY a valid JSON object with these exact keys "
            f"and their corresponding numerical or categorical values. "
            f"No extra keys, no explanations."
        )

        checkpoint_file = os.path.join(CHECKPOINT_FOLDER, f"extract_{dataset_type}.json")
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

        job_registry.set(job_id, {
            "progress": 5,
            "stage": f"Starting feature extraction... ({len(done_names)} already done)",
            "done": False,
        })

        for i, media_path in enumerate(media_files):
            file_name = os.path.basename(media_path)
            media_name = os.path.splitext(file_name)[0]

            if media_name in done_names:
                progress = 5 + int((i / total) * 90)
                job_registry.set(job_id, {
                    "progress": progress,
                    "stage": f"Skipping ({i + 1}/{total}): {file_name} (already done)",
                    "done": False,
                })
                continue

            progress = 5 + int((i / total) * 90)
            job_registry.set(job_id, {
                "progress": progress,
                "stage": f"Extracting ({i + 1}/{total}): {file_name}",
                "done": False,
            })

            result = process_single_media(media_path, prompt=prompt, model_name=model_name)
            analysis = result.get("analysis", {})
            if isinstance(analysis, dict):
                attrs = analysis.get("attributes", analysis)
                for key in _META_KEYS:
                    attrs.pop(key, None)
            else:
                attrs = {}

            row = {"media_name": media_name}
            missing = []
            for feat_key in feature_spec:
                val = attrs.get(feat_key)
                if val is None:
                    missing.append(feat_key)
                    val = 0
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val) if val else ""
                elif isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False)
                row[feat_key] = val
            if missing:
                logger.warning("%s: missing features %s", file_name, missing)
            features_data.append(row)

            try:
                with open(checkpoint_file, "w", encoding="utf-8") as cf:
                    json.dump(features_data, cf, ensure_ascii=False)
            except Exception as e:
                logger.warning("Cannot write checkpoint: %s", e)

        df_X = pd.DataFrame(features_data)

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
            },
        }
        if df_Y is not None:
            result_payload["details"]["dataset_Y_columns"] = list(df_Y.columns)
        job_registry.set(job_id, result_payload)

    except Exception as e:
        logger.exception("Error during extraction job %s", job_id)
        job_registry.set(job_id, {"progress": 100, "stage": "Error", "done": True, "error": str(e)})
