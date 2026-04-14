"""Phase 1: Feature discovery from sample media files."""
import json
import logging
import os
import re
import time

import pandas as pd

from config import DISCOVERY_MAX_SAMPLES
from pipeline.feature_schema import normalize_feature_spec
from services.openai_service import (
    _tracked_ollama_lock,
    local_client,
    ollama_request_options,
)
from services.processing import process_single_media
from utils.target_context import build_labels_context

logger = logging.getLogger(__name__)

_META_KEYS = {"summary", "classification", "reasoning"}

_AUDIO_KEYWORDS = frozenset({
    "audio", "sound", "speech", "voice", "music", "noise",
    "loudness", "pitch", "tone", "tempo", "silence", "transcript",
    "emotion",  # audio_emotion is the concrete offender
})


def _any_has_audio(paths: list[str]) -> bool:
    """Return True if at least one file has an audio stream (uses ffprobe)."""
    try:
        import ffmpeg  # ffmpeg-python — already a project dependency
        for p in paths:
            try:
                probe = ffmpeg.probe(p)
                if any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
                    return True
            except Exception:
                pass
    except ImportError:
        logger.debug("ffmpeg-python not available; skipping audio stream check")
    return False


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


def _is_gpu_load_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "unable to allocate cuda",
            "cuda0 buffer",
            "do load request",
            "/load\": eof",
        )
    )


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
    use_cpu_fallback = False
    while True:
        try:
            options = ollama_request_options()
            if use_cpu_fallback:
                options["num_gpu"] = 0
            with _tracked_ollama_lock():
                local_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "1"}],
                    max_tokens=1,
                    temperature=0.0,
                    extra_body={"options": options},
                )
            return
        except Exception as exc:
            if not use_cpu_fallback and _is_gpu_load_error(exc):
                use_cpu_fallback = True
                logger.warning("GPU load failed during warm-up, switching to CPU fallback: %s", exc)
                continue
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
                    progress_cb(3, f"Model loading (attempt {attempt})...")
                except Exception as cb_exc:
                    logger.debug("progress_cb failed: %s", cb_exc)
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
            except Exception as cb_exc:
                logger.debug("progress_cb failed: %s", cb_exc)

    pipeline.target_variable = target_variable
    target_mode = getattr(pipeline, "target_mode", "regression")

    labels_context = build_labels_context(labels_df, target_variable, target_mode)

    # Pre-warm the model so Ollama finishes loading before the observation loop.
    # Without this, the first inference call often returns EOF while the runner starts.
    _cb(3, "Loading model...")
    _warm_up_model(model_name, progress_cb=progress_cb)

    # Step 1: analyse each sample independently to gather observations
    observations = []
    sample_paths = media_paths[:DISCOVERY_MAX_SAMPLES]
    n_samples = len(sample_paths)

    media_has_audio = _any_has_audio(sample_paths)
    if not media_has_audio:
        logger.info("No audio streams detected in sample media — audio-based features will be suppressed.")
    audio_note = (
        ""
        if media_has_audio
        else (
            "\nNOTE: This media clip does NOT contain an audio track. "
            "Do not describe or infer any audio characteristics."
        )
    )
    dim_note = (
        "Cover DIVERSE perceptual dimensions (visual, temporal, semantic)"
        if not media_has_audio
        else "Cover DIVERSE perceptual dimensions (visual, audio, temporal, semantic)"
    )

    _cb(5, f"Preparing analysis of {n_samples} samples...")
    for idx, path in enumerate(sample_paths):
        file_name = os.path.basename(path)
        pct = 5 + int((idx / n_samples) * 55)
        _cb(pct, f"Analyzuji vzorek {idx + 1}/{n_samples}: {file_name}...")
        obs_prompt = (
            "You are a media analysis AI.\n"
            "Carefully observe this media clip and describe what you perceive — "
            "visual content, motion, mood, pacing, people, "
            "objects, environment, and any other notable properties."
            + audio_note
            + "\nBe objective and specific. Output a concise bullet-point list of observations."
        )
        result = process_single_media(path, prompt=obs_prompt, model_name=model_name)
        raw = result.get("analysis") or result.get("description") or str(result)
        if raw:
            observations.append(str(raw))

    observations_text = "\n\n---\n\n".join(
        f"Sample {i+1}:\n{obs}" for i, obs in enumerate(observations)
    )

    # Step 2: ask LLM to derive a universal feature spec from the observations
    _cb(65, f"LLM synthesising feature spec from {len(observations)} samples...")
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
        f"- {dim_note} — do NOT repeat the same dimension multiple times\n"
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
    use_cpu_fallback = False
    for attempt in range(max_retries):
        try:
            options = ollama_request_options()
            if use_cpu_fallback:
                options["num_gpu"] = 0
            with _tracked_ollama_lock():
                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3,
                    extra_body={"options": options},
                )
            break
        except Exception as exc:
            if not use_cpu_fallback and _is_gpu_load_error(exc):
                use_cpu_fallback = True
                logger.warning("GPU load failed during synthesis, switching to CPU fallback: %s", exc)
                continue
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
            _cb(70, f"Model loading, retrying request ({attempt + 2}/{max_retries})...")
            time.sleep(wait_s)

    if response is None:
        raise RuntimeError("LLM feature synthesis failed without response.")
    raw_content = response.choices[0].message.content or ""
    # Use json decoder to find the first valid JSON object
    _cb(90, "Parsing feature specification...")
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

    if not media_has_audio and all_features:
        audio_feats = [
            k for k in all_features
            if any(kw in k.lower() for kw in _AUDIO_KEYWORDS)
        ]
        if audio_feats:
            logger.warning(
                "Feature discovery proposed audio-based features for silent media — removing: %s",
                audio_feats,
            )
            for k in audio_feats:
                del all_features[k]

    if all_features:
        pipeline.feature_spec = all_features
        return all_features

    # Fallback
    return {"visual_complexity": [0, 10], "action_intensity": [0, 10]}
