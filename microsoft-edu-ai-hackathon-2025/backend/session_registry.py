"""Per-session pipeline registry.

Each browser session gets its own MachineLearningPipeline instance so that
concurrent users do not overwrite each other's state.
"""
import logging
import os
import threading
import time

from config import CHECKPOINT_FOLDER
from pipeline import MachineLearningPipeline

logger = logging.getLogger(__name__)

_SESSION_TTL = 6 * 3600  # seconds — evict idle sessions after 6 hours
_registry: dict[str, dict] = {}
_lock = threading.Lock()


def get_pipeline(session_id: str) -> MachineLearningPipeline:
    """Return (creating if needed) the pipeline for *session_id*."""
    with _lock:
        _cleanup_stale()
        if session_id not in _registry:
            folder = os.path.join(CHECKPOINT_FOLDER, "sessions", session_id)
            os.makedirs(folder, exist_ok=True)
            pl = MachineLearningPipeline(checkpoint_folder=folder)
            pl.load_state()
            _registry[session_id] = {"pipeline": pl, "last_used": time.time()}
            logger.info("Created new pipeline for session %s", session_id)
        else:
            _registry[session_id]["last_used"] = time.time()
        return _registry[session_id]["pipeline"]


def delete_session(session_id: str) -> None:
    """Remove a session's pipeline from the in-memory registry."""
    with _lock:
        _registry.pop(session_id, None)


def _cleanup_stale() -> None:
    """Evict sessions that have been idle longer than _SESSION_TTL.

    Must be called with *_lock* held.
    """
    now = time.time()
    stale = [k for k, v in _registry.items() if now - v["last_used"] > _SESSION_TTL]
    for k in stale:
        logger.info("Evicting idle session %s", k)
        del _registry[k]
