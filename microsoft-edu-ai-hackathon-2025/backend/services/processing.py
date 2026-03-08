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

from .openai_service import extract_image_features_with_llm
from .speech_service import extract_audio_from_video, transcribe_with_timestamps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt", "md", "csv"},
    "image": {"png", "jpg", "jpeg"},
    "video": {"mp4", "avi", "mov", "mkv"},
}
VIDEO_KEY_FRAME_LIMIT = 8

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


def extract_key_frames_with_timestamps(
    video_path: str, frame_limit: int = 8
) -> List[Tuple[np.ndarray, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total == 0 or fps == 0:
        return []
    step = max(1, total // frame_limit)
    res = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            res.append((frame, round(i / fps, 2)))
        if len(res) >= frame_limit:
            break
    cap.release()
    return res


# ================================================================
# CORE: Zpracuj jedno médium s custom promptem
# ================================================================

def process_single_media(
    media_path: str,
    prompt: str,
    model_name: str = "qwen2.5vl:7b",
) -> Dict[str, Any]:
    """
    Zpracuje jedno médium (video/obrázek) a pošle na LLM s daným promptem.

    Pro video: extrahuje keyframes + audio transcript, přiloží je k promptu.
    Pro obrázek: pošle obrázek přímo.

    Vrací dict s klíči: filename, transcript, analysis (JSON z LLM), error (pokud nastala).
    """
    filename = os.path.basename(media_path)
    result: Dict[str, Any] = {
        "filename": filename,
        "transcript": "",
        "analysis": None,
    }

    try:
        if _is_image_file(media_path):
            # --- OBRÁZEK ---
            img_b64 = _image_to_base64(media_path)

            llm_resp = extract_image_features_with_llm(
                [img_b64], prompt=prompt, deployment_name=model_name, feature_gen=True
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
                full_prompt += f"\n\nAudio transcript:\n{transcript_text[:12000]}"
            if timestamps:
                full_prompt += f"\n\nVisual frame timestamps: {', '.join(timestamps)}"

            # 4. LLM call
            llm_resp = extract_image_features_with_llm(
                frame_b64, prompt=full_prompt, deployment_name=model_name, feature_gen=True
            )
            result["analysis"] = llm_resp[0] if isinstance(llm_resp, list) and llm_resp else llm_resp

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {filename}: {e}")

    return result
