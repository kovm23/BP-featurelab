"""Unit tests for utils.retry.retry_with_backoff."""
import pytest

from utils.retry import retry_with_backoff


class Boom(Exception):
    pass


class Other(Exception):
    pass


def test_returns_on_first_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert retry_with_backoff(fn, max_attempts=3, base_delay=0) == "ok"
    assert calls["n"] == 1


def test_retries_then_succeeds(monkeypatch):
    attempts = iter([Boom("transient"), Boom("transient"), "finally"])

    def fn():
        v = next(attempts)
        if isinstance(v, Exception):
            raise v
        return v

    result = retry_with_backoff(fn, max_attempts=3, base_delay=0)
    assert result == "finally"


def test_raises_last_exception_after_exhaustion():
    def fn():
        raise Boom("nope")

    with pytest.raises(Boom):
        retry_with_backoff(fn, max_attempts=3, base_delay=0)


def test_should_retry_false_skips_retry():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise Other("not retriable")

    with pytest.raises(Other):
        retry_with_backoff(
            fn,
            max_attempts=5,
            base_delay=0,
            should_retry=lambda e: isinstance(e, Boom),
        )
    assert calls["n"] == 1


def test_on_retry_callback_is_called(monkeypatch):
    seen = []

    def fn():
        if len(seen) < 2:
            raise Boom("x")
        return "done"

    def on_retry(attempt, delay, exc):
        seen.append((attempt, delay))

    result = retry_with_backoff(fn, max_attempts=5, base_delay=0, on_retry=on_retry)
    assert result == "done"
    assert len(seen) == 2
    assert seen[0][0] == 1 and seen[1][0] == 2


def test_exponential_delay_doubles(monkeypatch):
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("utils.retry.time.sleep", fake_sleep)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 4:
            raise Boom("x")
        return "ok"

    retry_with_backoff(fn, max_attempts=4, base_delay=1.0, max_delay=100.0)
    assert sleeps == [1.0, 2.0, 4.0]


def test_max_delay_clamps_exponential(monkeypatch):
    sleeps = []
    monkeypatch.setattr("utils.retry.time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 5:
            raise Boom("x")
        return "ok"

    retry_with_backoff(fn, max_attempts=5, base_delay=10.0, max_delay=20.0)
    assert sleeps == [10.0, 20.0, 20.0, 20.0]


def test_invalid_max_attempts():
    with pytest.raises(ValueError):
        retry_with_backoff(lambda: None, max_attempts=0)
