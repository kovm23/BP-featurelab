"""Health check endpoint — reports Ollama availability."""
import logging

import requests
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def api_health():
    """Check if the backend and Ollama are reachable."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return jsonify({"ok": True, "ollama": ollama_ok})
