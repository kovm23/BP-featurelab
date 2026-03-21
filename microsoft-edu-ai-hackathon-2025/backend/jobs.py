"""Async job registry with thread-safe access."""
import threading

_lock = threading.Lock()
_registry: dict = {}


def get(job_id: str) -> dict | None:
    with _lock:
        return _registry.get(job_id)


def set(job_id: str, value: dict) -> None:  # noqa: A001
    with _lock:
        _registry[job_id] = value


def get_or_default(job_id: str, default: dict) -> dict:
    with _lock:
        return _registry.get(job_id, default)
