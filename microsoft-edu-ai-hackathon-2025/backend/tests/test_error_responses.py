"""Tests for API error diagnostics."""

import app as app_module


def test_unhandled_error_returns_traceable_json(monkeypatch):
    def fail_pipeline():
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(app_module, "get_pipeline", fail_pipeline)

    response = app_module.app.test_client().get("/state")

    assert response.status_code == 500
    assert response.json["error"] == "Backend request failed. See server logs with the request ID."
    assert response.json["request_id"] == response.headers["X-Request-ID"]