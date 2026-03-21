"""Job status endpoint."""
from flask import Blueprint, jsonify

import jobs as job_registry

status_bp = Blueprint("status", __name__)


@status_bp.route("/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    """Return the current status of an async job."""
    return jsonify(job_registry.get_or_default(job_id, {"error": "Job not found"}))
