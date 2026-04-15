"""Retry helper with exponential backoff for flaky external calls (Ollama).

Deliberately tiny — no dependency on tenacity. Handles only the minimal case
we need: synchronous call, configurable attempts, exponential delay capped
by ``max_delay``, optional predicate that decides if an exception is retriable.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> T:
    """Call ``fn()`` up to ``max_attempts`` times with exponential backoff.

    - Delay doubles each failure: ``base_delay``, ``base_delay*2``, ``base_delay*4`` …
      clamped by ``max_delay``.
    - If ``should_retry`` is provided and returns False for an exception,
      the exception is re-raised immediately (no further attempts).
    - ``on_retry(attempt, delay, exc)`` is called before each sleep (useful
      for progress callbacks or structured logging).

    Raises the last encountered exception if all attempts fail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if on_retry is not None:
                try:
                    on_retry(attempt, delay, exc)
                except Exception as cb_exc:
                    logger.debug("retry on_retry callback failed: %s", cb_exc)
            else:
                logger.warning(
                    "Attempt %s/%s failed: %s. Retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
            time.sleep(delay)
