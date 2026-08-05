"""Shared Flask extensions.

Kept in a dedicated module so blueprints can import the limiter without
creating a circular import with ``app.py``.
"""
import os

from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Limits for the heavy, abuse-prone endpoints (LLM jobs, uploads, imports).
# Overridable via environment so operators can tune per deployment.
JOB_RATE_LIMIT = os.getenv("RATE_LIMIT_JOBS", "20 per minute")
IMPORT_RATE_LIMIT = os.getenv("RATE_LIMIT_IMPORT", "10 per minute")


def rate_limit_key() -> str:
    """Rate-limit bucket key.

    Prefer the per-browser session id so one user gets their own budget and
    cannot starve others; fall back to the remote address for requests that do
    not carry the ``X-Session-ID`` header.
    """
    return request.headers.get("X-Session-ID") or get_remote_address()


# No global default limits: polling endpoints (/status, /state, /health,
# /queue-info) are called frequently and must stay unthrottled. Heavy endpoints
# opt in explicitly via ``@limiter.limit(JOB_RATE_LIMIT)``.
limiter = Limiter(
    key_func=rate_limit_key,
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
    headers_enabled=True,
)

# Allow tests and specific deployments to disable rate limiting entirely
# (e.g. RATELIMIT_ENABLED=false), which also avoids spawning the storage
# backend's background cleanup thread.
if os.getenv("RATELIMIT_ENABLED", "true").strip().lower() in ("0", "false", "no"):
    limiter.enabled = False
