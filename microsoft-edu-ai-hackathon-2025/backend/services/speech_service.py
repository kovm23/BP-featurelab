import logging
import os
import ffmpeg
from typing import Dict, Any, Union
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Global variable holding the loaded model instance (Singleton)
local_whisper_model = None


def get_local_whisper():
    """Load the Whisper model into VRAM once (lazy singleton)."""
    global local_whisper_model
    if local_whisper_model is None:
        logger.info("Loading Local Whisper Large-v3 model to GPU (this may take a moment)...")
        local_whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return local_whisper_model


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract the audio track from a video file and save it as MP3."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='libmp3lame', **{'qscale:a': 2}, loglevel="quiet")
            .overwrite_output()
            .run()
        )
        return True
    except ffmpeg.Error as e:
        logger.error("ffmpeg audio extraction failed: %s", e)
        return False
    except Exception:
        logger.exception("Unexpected error during audio extraction")
        return False


def transcribe_with_timestamps(file_path: str) -> Dict[str, Any]:
    """Run local transcription and return structured data with timestamps."""
    try:
        model = get_local_whisper()
        segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True)

        transcript_result = []
        full_text = []

        for segment in segments:
            transcript_result.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())

        return {
            "full_text": " ".join(full_text),
            "segments": transcript_result,
            "language": info.language
        }
    except Exception as e:
        return {"error": str(e), "full_text": "", "segments": []}


def transcribe_video_file(file_path: str, model_choice: str = "local") -> Union[str, Dict[str, Any]]:
    """Transcribe an audio/video file using the local Whisper model."""
    return transcribe_with_timestamps(file_path)
