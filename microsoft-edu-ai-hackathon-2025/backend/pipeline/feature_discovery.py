"""Phase 1: Feature discovery from sample media files."""
import json
import logging
import os
import re

import pandas as pd

from pipeline.feature_schema import normalize_feature_spec
from services.openai_service import local_client, _tracked_ollama_lock
from services.processing import process_single_media
from utils.target_context import build_labels_context

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning"}


def _is_transient_ollama_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_tokens = (
        "eof",
        "load request",
        "connection reset",
        "remoteprotocolerror",
        "timed out",
        "connection refused",
    )
    return any(token in msg for token in transient_tokens)


def _warm_up_model(model_name: str, progress_cb=None) -> None:
    """Send a minimal request to ensure the model is fully loaded before real work.

    Ollama spawns a new runner process on the first request and can return an EOF
    while that runner is still initialising. Retrying here with generous backoff
    absorbs the load latency so downstream calls succeed on the first try.
    """
    import time
    max_wait = 120
    deadline = time.monotonic() + max_wait
    attempt = 0
    while True:
        try:
            with _tracked_ollama_lock():
                local_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "1"}],
                    max_tokens=1,
                    temperature=0.0,
                )
            return
        except Exception as exc:
            if not _is_transient_ollama_error(exc):
                raise
            attempt += 1
            wait_s = min(15 * attempt, 45)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Model '{model_name}' failed to load within {max_wait}s: {exc}"
                ) from exc
            actual_wait = min(wait_s, remaining)
            logger.warning(
                "Ollama model load not ready (attempt %s), retrying in %ss: %s",
                attempt,
                actual_wait,
                exc,
            )
            if progress_cb:
                try:
                    progress_cb(3, f"Model se načítá ({attempt}. pokus)...")
                except Exception:
                    pass
            time.sleep(actual_wait)


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

    labels_context = build_labels_context(labels_df, target_variable, target_mode)

    # Pre-warm the model so Ollama finishes loading before the observation loop.
    # Without this, the first inference call often returns EOF while the runner starts.
    _cb(3, "Načítám model...")
    _warm_up_model(model_name, progress_cb=progress_cb)

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
        f"(lowercase_with_underscores). Values MUST be one of:\n"
        f"- numeric range as [min, max] (prefer integer ranges when possible)\n"
        f"- categorical domain as [\"category_a\", \"category_b\", ...]\n"
        f"DO NOT output more than 8 features.\n"
        f"Example: {{\"action_intensity\": [0, 10], \"speech_presence\": [0, 1], "
        f"\"scene_type\": [\"indoor\", \"outdoor\", \"mixed\"]}}"
    )

    response = None
    max_retries = 4
    backoff_s = 20  # model cold-load takes 20-30 s; each wait must exceed that
    import time
    for attempt in range(max_retries):
        try:
            with _tracked_ollama_lock():
                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3,
                )
            break
        except Exception as exc:
            is_last = attempt >= max_retries - 1
            if is_last or not _is_transient_ollama_error(exc):
                raise
            wait_s = backoff_s * (attempt + 1)
            logger.warning(
                "Transient Ollama failure during feature synthesis (attempt %s/%s): %s. Retrying in %ss",
                attempt + 1,
                max_retries,
                exc,
                wait_s,
            )
            _cb(70, f"Model se načítá, opakuji požadavek ({attempt + 2}/{max_retries})...")
            time.sleep(wait_s)

    if response is None:
        raise RuntimeError("LLM feature synthesis failed without response.")
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
            all_features = normalize_feature_spec(all_features)
            # Cap at 8 features
            if len(all_features) > 8:
                all_features = dict(list(all_features.items())[:8])
        except json.JSONDecodeError:
            logger.warning("Could not parse feature spec JSON from synthesis step")

    if all_features:
        pipeline.feature_spec = all_features
        return all_features

    # Fallback
    return {"visual_complexity": [0, 10], "action_intensity": [0, 10]}
