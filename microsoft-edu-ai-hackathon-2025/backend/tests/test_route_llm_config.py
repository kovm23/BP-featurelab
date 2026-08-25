"""Route tests for forwarding LLM endpoint configuration into background jobs."""

import io
from types import SimpleNamespace

import app as app_module
from routes import extract as extract_routes
from routes import repeatability as repeatability_routes


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.feature_spec = {"f1": [0, 1]}
        self.target_variable = "target"

    def invalidate_from_phase(self, phase):
        self.invalidated_phase = phase

    def save_state(self):
        pass

    def extract_features_async(
        self,
        media_files,
        feature_spec,
        job_id,
        model_name,
        dataset_type,
        csv_path,
        labels_df,
        llm_base_url="",
        llm_api_key="",
        llm_temperature=None,
    ):
        self.calls.append({
            "media_files": media_files,
            "feature_spec": feature_spec,
            "model_name": model_name,
            "dataset_type": dataset_type,
            "csv_path": csv_path,
            "labels_df": labels_df,
            "llm_base_url": llm_base_url,
            "llm_api_key": llm_api_key,
            "llm_temperature": llm_temperature,
        })


def test_extract_upload_forwards_llm_temperature(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(extract_routes.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        extract_routes,
        "extract_zip_contents",
        lambda *_args, **_kwargs: (["clip.mp4"], None),
    )

    response = app_module.app.test_client().post(
        "/extract",
        data={
            "file": (io.BytesIO(b"zip"), "dataset.zip"),
            "model": "gpt-test",
            "feature_spec": '{"f1": [0, 1]}',
            "dataset_type": "training",
            "use_custom_llm": "1",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-test",
            "llm_temperature": "0.37",
        },
        content_type="multipart/form-data",
        headers={"X-Session-ID": "route-test"},
    )

    assert response.status_code == 200
    assert pipeline.calls[0]["llm_base_url"] == "https://api.openai.com/v1"
    assert pipeline.calls[0]["llm_api_key"] == "sk-test"
    assert pipeline.calls[0]["llm_temperature"] == 0.37


def test_extract_upload_ignores_custom_credentials_without_opt_in(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(extract_routes.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        extract_routes,
        "extract_zip_contents",
        lambda *_args, **_kwargs: (["clip.mp4"], None),
    )

    response = app_module.app.test_client().post(
        "/extract",
        data={
            "file": (io.BytesIO(b"zip"), "dataset.zip"),
            "model": "qwen2.5vl:7b",
            "feature_spec": '{"f1": [0, 1]}',
            "dataset_type": "training",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-stale",
            "llm_temperature": "0.37",
        },
        content_type="multipart/form-data",
        headers={"X-Session-ID": "route-test"},
    )

    assert response.status_code == 200
    assert pipeline.calls[0]["llm_base_url"] == ""
    assert pipeline.calls[0]["llm_api_key"] == ""


def test_extract_local_forwards_llm_temperature(monkeypatch, tmp_path):
    zip_path = tmp_path / "dataset.zip"
    zip_path.write_bytes(b"zip")
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(extract_routes.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        extract_routes,
        "extract_zip_contents",
        lambda *_args, **_kwargs: (["clip.mp4"], None),
    )

    response = app_module.app.test_client().post(
        "/extract-local",
        json={
            "zip_path": str(zip_path),
            "model": "gpt-test",
            "feature_spec": {"f1": [0, 1]},
            "dataset_type": "testing",
            "use_custom_llm": True,
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-test",
            "llm_temperature": 0.42,
        },
        headers={"X-Session-ID": "route-test"},
    )

    assert response.status_code == 200
    assert pipeline.calls[0]["dataset_type"] == "testing"
    assert pipeline.calls[0]["llm_base_url"] == "https://api.openai.com/v1"
    assert pipeline.calls[0]["llm_api_key"] == "sk-test"
    assert pipeline.calls[0]["llm_temperature"] == 0.42


def test_repeatability_forwards_custom_endpoint(monkeypatch):
    captured = []
    monkeypatch.setattr(
        app_module,
        "get_pipeline",
        lambda: SimpleNamespace(feature_spec={"f1": [0, 1]}),
    )
    monkeypatch.setattr(repeatability_routes.threading, "Thread", ImmediateThread)

    def fake_extract_single_pass(media_path, prompt, model_name, **kwargs):
        captured.append({
            "media_path": media_path,
            "prompt": prompt,
            "model_name": model_name,
            **kwargs,
        })
        return {"f1": 1}

    monkeypatch.setattr(repeatability_routes, "_extract_single_pass", fake_extract_single_pass)

    response = app_module.app.test_client().post(
        "/repeatability-test",
        data={
            "file": (io.BytesIO(b"media"), "clip.mp4"),
            "n_repetitions": "2",
            "model": "gpt-test",
            "use_custom_llm": "1",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-test",
            "llm_temperature": "0.55",
        },
        content_type="multipart/form-data",
        headers={"X-Session-ID": "route-test"},
    )

    assert response.status_code == 200
    assert len(captured) == 2
    assert captured[0]["model_name"] == "gpt-test"
    assert captured[0]["custom_base_url"] == "https://api.openai.com/v1"
    assert captured[0]["custom_api_key"] == "sk-test"
    assert captured[0]["custom_temperature"] == 0.55
