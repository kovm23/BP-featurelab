"""Shared predicates for detecting transient/GPU-related Ollama errors.

Used by feature_discovery and openai_service to decide whether a failed
Ollama call is worth retrying (and whether to fall back to CPU).
"""
from __future__ import annotations

_TRANSIENT_TOKENS = (
    "eof",
    "load request",
    "connection reset",
    "remoteprotocolerror",
    "timed out",
    "connection refused",
)

_GPU_LOAD_TOKENS = (
    "unable to allocate cuda",
    "cuda0 buffer",
    "do load request",
    "/load\": eof",
    "connection reset",
)


def is_transient_ollama_error(exc: BaseException) -> bool:
    """True if the error looks like a retriable Ollama hiccup (cold-load, reset, timeout)."""
    msg = str(exc).lower()
    return any(token in msg for token in _TRANSIENT_TOKENS)


def is_gpu_load_error(exc: BaseException) -> bool:
    """True if the error indicates Ollama failed to load the model on GPU.

    CPU fallback is warranted in that case.
    """
    msg = str(exc).lower()
    return any(token in msg for token in _GPU_LOAD_TOKENS)
