"""Async job registry with thread-safe access and TTL-based cleanup."""
import threading
import time

_lock = threading.Lock()
_registry: dict = {}

# Jobs older than this (seconds) are eligible for cleanup
_JOB_TTL = 2 * 3600  # 2 hours


def _cleanup_stale() -> None:
    """Remove completed jobs older than _JOB_TTL. Call without holding _lock."""
    cutoff = time.time() - _JOB_TTL
    with _lock:
        stale = [
            jid for jid, job in _registry.items()
            if job.get("done") and job.get("_created_at", 0) < cutoff
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


# Keep `set` as an alias for backward compatibility
set = set_job  # noqa: A001


def get_or_default(job_id: str, default: dict) -> dict:
    with _lock:
        return _registry.get(job_id, default)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _registry:
            _registry[job_id].update(kwargs)
