"""Phase 1: Feature discovery from sample media files."""
import logging

import pandas as pd

from services.processing import process_single_media

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning"}


def discover_features(
    pipeline,
    media_paths: list[str],
    target_variable: str,
    model_name: str,
    labels_df: pd.DataFrame | None = None,
) -> dict:
    """Analyse sample media and suggest a feature definition spec.

    Updates pipeline.target_variable and pipeline.feature_spec in-place.
    Returns the suggested feature dict.
    """
    pipeline.target_variable = target_variable

    labels_context = ""
    if labels_df is not None:
        target_col = None
        for c in labels_df.columns:
            if c.lower().replace(" ", "_") == target_variable.lower().replace(" ", "_"):
                target_col = c
                break
        if target_col is None:
            numeric_cols = labels_df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]

        if target_col is not None:
            col_data = labels_df[target_col]
            labels_context = (
                f"\n\nYou also have access to the target variable '{target_col}' "
                f"from the training labels:\n"
                f"- Min: {col_data.min()}, Max: {col_data.max()}, "
                f"Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}\n"
                f"- Sample values: {list(col_data.head(10).values)}\n"
                f"Use this information to suggest features that would correlate "
                f"well with these target values.\n"
            )

    prompt = (
        f"You are a machine learning feature engineer.\n"
        f"The goal is to build a model to predict: '{target_variable}'.\n"
        f"Analyze the provided media sample(s) and suggest 3 to 8 features "
        f"that are highly relevant for predicting the target.\n\n"
        f"For each feature, provide:\n"
        f"- A descriptive name (lowercase_with_underscores)\n"
        f"- Expected value range, units, or categories\n\n"
        f"{labels_context}"
        f"Output STRICTLY a JSON object where keys are feature names "
        f"and values are descriptions with ranges/units.\n"
        f"Example: {{\"movie_length\": \"duration in seconds (0-7200)\", "
        f"\"extreme_language\": \"score 0-10\"}}"
    )

    all_features = {}
    for path in media_paths[:3]:  # max 3 samples
        result = process_single_media(path, prompt=prompt, model_name=model_name)
        analysis = result.get("analysis")
        if isinstance(analysis, dict):
            features = analysis.get("attributes", analysis)
            for key in _META_KEYS:
                features.pop(key, None)
            all_features.update(features)

    if all_features:
        pipeline.feature_spec = all_features
        return all_features

    # Fallback
    return {"visual_complexity": "score 1-10", "action_intensity": "score 1-10"}
