"""Phase 1: Feature discovery from sample media files."""
import json
import logging
import os
import re

import pandas as pd

from services.openai_service import local_client, _tracked_ollama_lock
from services.processing import process_single_media

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning"}


def discover_features(
    pipeline,
    media_paths: list[str],
    target_variable: str,
    model_name: str,
    labels_df: pd.DataFrame | None = None,
    progress_cb=None,
) -> dict:
    """Analyse sample media and suggest a feature definition spec.

    Updates pipeline.target_variable and pipeline.feature_spec in-place.
    Returns the suggested feature dict.
    """
    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    pipeline.target_variable = target_variable
    target_mode = getattr(pipeline, "target_mode", "regression")

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

    # Step 1: analyse each sample independently to gather observations
    observations = []
    sample_paths = media_paths[:5]
    n_samples = len(sample_paths)
    _cb(5, f"Připravuji analýzu {n_samples} vzorků...")
    for idx, path in enumerate(sample_paths):
        file_name = os.path.basename(path)
        pct = 5 + int((idx / n_samples) * 55)
        _cb(pct, f"Analyzuji vzorek {idx + 1}/{n_samples}: {file_name}...")
        obs_prompt = (
            "You are a media analysis AI.\n"
            "Carefully observe this media clip and describe what you perceive — "
            "visual content, motion, audio characteristics, mood, pacing, people, "
            "objects, environment, and any other notable properties.\n"
            "Be objective and specific. Output a concise bullet-point list of observations."
        )
        result = process_single_media(path, prompt=obs_prompt, model_name=model_name)
        raw = result.get("analysis") or result.get("description") or str(result)
        if raw:
            observations.append(str(raw))

    observations_text = "\n\n---\n\n".join(
        f"Sample {i+1}:\n{obs}" for i, obs in enumerate(observations)
    )

    # Step 2: ask LLM to derive a universal feature spec from the observations
    _cb(65, f"LLM navrhuje feature spec z {len(observations)} vzorků...")
    mode_hint = (
        "Target type: regression (continuous numeric value). "
        "Prefer features that can be quantified on continuous scales."
        if target_mode == "regression"
        else "Target type: classification (categorical label). "
        "Prefer discriminative features that separate classes clearly."
    )

    synthesis_prompt = (
        f"You are a machine learning feature engineer.\n"
        f"Your goal is to predict: '{target_variable}'.\n\n"
        f"{mode_hint}\n\n"
        f"Below are observations from {len(observations)} media sample(s):\n\n"
        f"{observations_text}\n\n"
        f"{labels_context}"
        f"Based on these observations, define EXACTLY 5 to 8 measurable features that:\n"
        f"- Can be extracted from ANY media clip of this type (not just these samples)\n"
        f"- Are likely to correlate with '{target_variable}'\n"
        f"- Cover DIVERSE perceptual dimensions (visual, audio, temporal, semantic) — do NOT repeat the same dimension multiple times\n"
        f"- Have clear, unambiguous measurement criteria\n"
        f"- Are independent from each other (avoid redundant or highly correlated features)\n\n"
        f"Output STRICTLY a JSON object with 5–8 keys. Keys are feature names "
        f"(lowercase_with_underscores). Values describe the measurement scale or "
        f"categories. DO NOT output more than 8 features.\n"
        f"Example: {{\"action_intensity\": \"score 0-10, how dynamic/fast-paced the clip is\", "
        f"\"speech_presence\": \"binary 0 or 1, whether speech is audible\"}}"
    )

    with _tracked_ollama_lock():
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3,
        )
    raw_content = response.choices[0].message.content or ""
    # Use json decoder to find the first valid JSON object
    _cb(90, "Parsování feature specifikace...")
    all_features = {}
    start = raw_content.find("{")
    if start != -1:
        try:
            decoder = json.JSONDecoder()
            all_features, _ = decoder.raw_decode(raw_content, start)
            for key in _META_KEYS:
                all_features.pop(key, None)
            # Cap at 8 features
            if len(all_features) > 8:
                all_features = dict(list(all_features.items())[:8])
        except json.JSONDecodeError:
            logger.warning("Could not parse feature spec JSON from synthesis step")

    if all_features:
        pipeline.feature_spec = all_features
        return all_features

    # Fallback
    return {"visual_complexity": "score 1-10", "action_intensity": "score 1-10"}
