"""Async job registry with thread-safe access and TTL-based cleanup."""
import threading
import time

_lock = threading.Lock()
_registry: dict = {}

# Completed jobs are eligible for cleanup this long after their last update
_JOB_TTL = 8 * 3600  # 8 hours after completion


def _cleanup_stale() -> None:
    """Remove completed jobs whose last update is older than _JOB_TTL."""
    cutoff = time.time() - _JOB_TTL
    with _lock:
        stale = [
            jid for jid, job in _registry.items()
            if job.get("done") and job.get("_updated_at", job.get("_created_at", 0)) < cutoff
        ]
        for jid in stale:
            del _registry[jid]


def get(job_id: str) -> dict | None:
    with _lock:
        return _registry.get(job_id)


def set_job(job_id: str, value: dict) -> None:
    value.setdefault("_created_at", time.time())
    with _lock:
        _registry[job_id] = value
    # Opportunistic cleanup on each new job (amortised cost)
    _cleanup_stale()


def get_or_default(job_id: str, default: dict) -> dict:
    with _lock:
        return _registry.get(job_id, default)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _registry:
            kwargs["_updated_at"] = time.time()
            _registry[job_id].update(kwargs)

