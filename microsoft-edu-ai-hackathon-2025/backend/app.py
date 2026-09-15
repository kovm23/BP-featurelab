"""Flask application factory."""
import logging
import os
import secrets
import time
import uuid

from flask import Flask, g, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import session_registry
from config import MAX_CONTENT_LENGTH
from env_loader import load_backend_env
from extensions import limiter
from routes import (
    discover_bp,
    export_matrix_bp,
    extract_bp,
    health_bp,
    predict_bp,
    repeatability_bp,
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
        "http://localhost:5173,http://127.0.0.1:5173,https://llmfeatures.vse.cz",
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

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

limiter.init_app(app)


@app.before_request
def _start_request_logging():
    """Attach an ID so a browser error can be matched to server logs."""
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    g.request_started_at = time.perf_counter()


@app.after_request
def _log_request(response):
    response.headers["X-Request-ID"] = g.get("request_id", "unknown")
    elapsed_ms = (time.perf_counter() - g.get("request_started_at", time.perf_counter())) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        g.get("request_id", "unknown"),
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.errorhandler(Exception)
def _unhandled_error(err):
    """Log the full failure while returning a traceable JSON error to the SPA."""
    if isinstance(err, HTTPException):
        return err
    request_id = g.get("request_id", "unknown")
    logger.exception(
        "Unhandled request error: request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.path,
    )
    return jsonify({
        "error": "Backend request failed. See server logs with the request ID.",
        "request_id": request_id,
    }), 500


@app.errorhandler(429)
def _rate_limited(err):
    """Return a JSON body for rate-limited requests so the SPA can display it."""
    return (
        jsonify({"error": "Too many requests. Please wait a moment and try again."}),
        429,
    )


@app.errorhandler(413)
def _payload_too_large(err):
    """Return a JSON body when an upload exceeds MAX_CONTENT_LENGTH."""
    limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return (
        jsonify({"error": f"Upload too large. The maximum allowed size is {limit_mb} MB."}),
        413,
    )


app.register_blueprint(discover_bp)
app.register_blueprint(extract_bp)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(status_bp)
app.register_blueprint(reset_bp)
app.register_blueprint(state_bp)
app.register_blueprint(health_bp)
app.register_blueprint(session_transfer_bp)
app.register_blueprint(export_matrix_bp)
app.register_blueprint(repeatability_bp)


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
