# processing.py - Modular file processing service

import os
import json
import base64
import io
import logging
from typing import List, Optional, Any, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .openai_service import DEFAULT_MODEL, extract_image_features_with_llm
from .speech_service import extract_audio_from_video, transcribe_with_timestamps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_KEY_FRAME_LIMIT = 10

MEDIA_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".gif"}


def _is_image_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _is_media_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in MEDIA_EXTENSIONS


def _convert_frame_to_base64(frame) -> str:
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _image_to_base64(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalised HSV histogram for scene comparison."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def extract_key_frames_with_timestamps(
    video_path: str, frame_limit: int = 10
) -> List[Tuple[np.ndarray, float]]:
    """Extract keyframes using scene-change detection.

    Selects frames at points where the visual content changes significantly
    (based on histogram Bhattacharyya distance). Falls back to uniform
    sampling when too few scene changes are detected.
    Always includes the first and last frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total == 0 or fps == 0:
        cap.release()
        return []

    # Pass 1: sample every N-th frame and compute histogram distances
    sample_step = max(1, total // 200)  # ~200 samples for efficiency
    frame_indices = list(range(0, total, sample_step))

    histograms: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (idx, frame, hist)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            hist = _compute_histogram(frame)
            histograms.append((idx, frame, hist))

    if len(histograms) < 2:
        cap.release()
        return [(h[1], round(h[0] / fps, 2)) for h in histograms]

    # Compute distances between consecutive frames
    distances = []
    for i in range(1, len(histograms)):
        dist = cv2.compareHist(
            histograms[i - 1][2], histograms[i][2], cv2.HISTCMP_BHATTACHARYYA
        )
        distances.append((i, dist))

    # Sort by distance descending — top peaks are scene changes
    distances.sort(key=lambda x: x[1], reverse=True)

    # Threshold: take frames with distance > 0.35 (significant change)
    SCENE_THRESHOLD = 0.35
    scene_change_indices = {0, len(histograms) - 1}  # always include first & last
    for frame_i, dist in distances:
        if dist < SCENE_THRESHOLD and len(scene_change_indices) >= 3:
            break
        scene_change_indices.add(frame_i)
        if len(scene_change_indices) >= frame_limit:
            break

    # If too few scene changes, fill with uniformly distributed frames
    if len(scene_change_indices) < frame_limit:
        uniform_step = max(1, len(histograms) // frame_limit)
        for j in range(0, len(histograms), uniform_step):
            scene_change_indices.add(j)
            if len(scene_change_indices) >= frame_limit:
                break

    # Collect selected frames in temporal order
    selected = sorted(scene_change_indices)[:frame_limit]
    result = []
    for si in selected:
        idx, frame, _ = histograms[si]
        result.append((frame, round(idx / fps, 2)))

    cap.release()
    logger.info(
        "Keyframe selection for %s: %d scene-based frames from %d total",
        os.path.basename(video_path), len(result), total,
    )
    return result


# ================================================================
# CORE: Process a single media file with a custom prompt
# ================================================================

def process_single_media(
    media_path: str,
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    custom_base_url: str = "",
    custom_api_key: str = "",
) -> Dict[str, Any]:
    """Process a single media file (video or image) and send it to the LLM with the given prompt.

    For video: extracts keyframes + audio transcript and appends them to the prompt.
    For image: sends the image directly.

    Returns a dict with keys: filename, transcript, analysis (JSON from LLM), error (if any).
    """
    filename = os.path.basename(media_path)
    result: Dict[str, Any] = {
        "filename": filename,
        "transcript": "",
        "analysis": None,
    }

    try:
        if _is_image_file(media_path):
            # --- IMAGE ---
            img_b64 = _image_to_base64(media_path)

            llm_resp = extract_image_features_with_llm(
                [img_b64], prompt=prompt, deployment_name=model_name, feature_gen=True,
                custom_base_url=custom_base_url, custom_api_key=custom_api_key,
            )
            result["analysis"] = llm_resp[0] if isinstance(llm_resp, list) and llm_resp else llm_resp

        else:
            # --- VIDEO ---
            # 1. Audio extraction + transcription
            audio_filename = os.path.splitext(filename)[0] + ".mp3"
            audio_path = os.path.join(os.path.dirname(media_path), audio_filename)
            transcript_text = ""
            has_audio = extract_audio_from_video(media_path, audio_path)
            if has_audio:
                t_data = transcribe_with_timestamps(audio_path)
                if isinstance(t_data, dict):
                    transcript_text = t_data.get("full_text", "")
            result["transcript"] = transcript_text

            # 2. Keyframes
            frames = extract_key_frames_with_timestamps(media_path, VIDEO_KEY_FRAME_LIMIT)
            frame_b64 = [_convert_frame_to_base64(f) for f, t in frames]
            timestamps = [f"{t}s" for f, t in frames]

            # 3. Build full prompt with transcript + timestamps context
            full_prompt = prompt
            if transcript_text:
                if len(transcript_text) > 12000:
                    logger.warning("Transcript truncated to 12000 chars for %s", media_path)
                full_prompt += f"\n\nAudio transcript:\n{transcript_text[:12000]}"
            if timestamps:
                full_prompt += f"\n\nVisual frame timestamps: {', '.join(timestamps)}"

            # 4. LLM call
            llm_resp = extract_image_features_with_llm(
                frame_b64, prompt=full_prompt, deployment_name=model_name, feature_gen=True,
                custom_base_url=custom_base_url, custom_api_key=custom_api_key,
            )
            result["analysis"] = llm_resp[0] if isinstance(llm_resp, list) and llm_resp else llm_resp

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {filename}: {e}")

    return result
