"""Flask application factory."""
import logging
import os
import secrets

from flask import Flask, request
from flask_cors import CORS

import session_registry
from env_loader import load_backend_env
from routes import (
    discover_bp,
    extract_bp,
    health_bp,
    predict_bp,
    reset_bp,
    session_transfer_bp,
    state_bp,
    status_bp,
    train_bp,
)

load_backend_env()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_allowed_origins() -> list[str]:
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://bpfeaturelab.kovm23.workers.dev",
    )
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)

ALLOWED_ORIGINS = _parse_allowed_origins()
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Session-ID"],
)
logger.info("Allowed CORS origins: %s", ", ".join(ALLOWED_ORIGINS))

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

app.register_blueprint(discover_bp)
app.register_blueprint(extract_bp)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(status_bp)
app.register_blueprint(reset_bp)
app.register_blueprint(state_bp)
app.register_blueprint(health_bp)
app.register_blueprint(session_transfer_bp)


def get_pipeline():
    """Return the MachineLearningPipeline for the current request's session.

    Session is identified by the *X-Session-ID* request header (a UUID
    generated and persisted in the browser's localStorage). Falls back to
    "default" so existing single-user deployments keep working without changes.
    """
    session_id = request.headers.get("X-Session-ID", "default")
    return session_registry.get_pipeline(session_id)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=debug,
        use_reloader=debug,
    )
