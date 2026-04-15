"""Health check endpoint — reports Ollama availability."""
import logging

import requests
from flask import Blueprint, jsonify

from services.openai_service import get_ollama_healthcheck_url, get_ollama_queue_info

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def api_health():
    """Check if the backend and Ollama are reachable."""
    try:
        r = requests.get(get_ollama_healthcheck_url(), timeout=2)
        ollama_ok = r.status_code == 200
    except requests.RequestException as e:
        logger.debug("Ollama health check failed: %s", e)
        ollama_ok = False

    return jsonify({"ok": True, "ollama": ollama_ok})


@health_bp.route("/queue-info")
def api_queue_info():
    """Return current Ollama processing queue status."""
    return jsonify(get_ollama_queue_info())
