"""Unit tests for OpenAI-compatible token-limit parameter handling."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pipeline import feature_discovery
from services import openai_service


class FakeCompletions:
    def __init__(self, failures=None, content="{}"):
        self.calls = []
        self.failures = list(failures or [])
        self.content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, failures=None, content="{}"):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(failures=failures, content=content)
        )

    @property
    def calls(self):
        return self.chat.completions.calls


def test_custom_endpoint_uses_max_completion_tokens():
    client = FakeClient()

    openai_service.create_chat_completion_with_token_limit(
        client,
        is_custom=True,
        token_limit=2048,
        model="gpt-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert client.calls[0]["max_completion_tokens"] == 2048
    assert "max_tokens" not in client.calls[0]


def test_custom_endpoint_falls_back_to_max_tokens_when_modern_param_is_unsupported():
    client = FakeClient(
        failures=[
            RuntimeError("Unsupported parameter: 'max_completion_tokens' is not supported")
        ]
    )

    openai_service.create_chat_completion_with_token_limit(
        client,
        is_custom=True,
        token_limit=2048,
        model="legacy-compatible-model",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(client.calls) == 2
    assert client.calls[0]["max_completion_tokens"] == 2048
    assert "max_tokens" not in client.calls[0]
    assert client.calls[1]["max_tokens"] == 2048
    assert "max_completion_tokens" not in client.calls[1]


def test_custom_endpoint_does_not_retry_unrelated_errors():
    client = FakeClient(failures=[RuntimeError("boom")])

    with pytest.raises(RuntimeError, match="boom"):
        openai_service.create_chat_completion_with_token_limit(
            client,
            is_custom=True,
            token_limit=2048,
            model="gpt-test",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert len(client.calls) == 1
    assert client.calls[0]["max_completion_tokens"] == 2048


def test_local_ollama_keeps_max_tokens(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(openai_service, "_tracked_ollama_lock", lambda: nullcontext())

    openai_service.create_chat_completion_with_token_limit(
        client,
        is_custom=False,
        token_limit=2048,
        model="qwen-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert client.calls[0]["max_tokens"] == 2048
    assert "max_completion_tokens" not in client.calls[0]


def test_warm_up_model_uses_token_limit_one_for_custom_endpoint(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(feature_discovery, "get_client", lambda *_: (client, True))

    feature_discovery._warm_up_model(
        "gpt-test",
        custom_base_url="https://api.openai.com/v1",
        custom_api_key="sk-test",
    )

    assert client.calls[0]["max_completion_tokens"] == 1
    assert "max_tokens" not in client.calls[0]


def test_text_extraction_uses_token_limit_2048_for_custom_endpoint(monkeypatch):
    client = FakeClient(content='{"feature": 1}')
    monkeypatch.setattr(openai_service, "get_client", lambda *_: (client, True))

    result = openai_service.extract_text_features_with_llm(
        ["sample text"],
        prompt="Extract one feature.",
        deployment_name="gpt-test",
        custom_base_url="https://api.openai.com/v1",
        custom_api_key="sk-test",
    )

    assert result == [{"feature": 1}]
    assert client.calls[0]["max_completion_tokens"] == 2048
    assert "max_tokens" not in client.calls[0]
