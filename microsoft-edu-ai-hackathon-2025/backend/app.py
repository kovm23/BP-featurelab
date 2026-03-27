"""Flask application factory."""
import logging
import os
import secrets

from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS

import session_registry
from routes import (
    analyze_bp,
    discover_bp,
    extract_bp,
    health_bp,
    predict_bp,
    reset_bp,
    state_bp,
    status_bp,
    train_bp,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Session-ID"],
)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

app.register_blueprint(discover_bp)
app.register_blueprint(extract_bp)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(analyze_bp)
app.register_blueprint(status_bp)
app.register_blueprint(reset_bp)
app.register_blueprint(state_bp)
app.register_blueprint(health_bp)


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


