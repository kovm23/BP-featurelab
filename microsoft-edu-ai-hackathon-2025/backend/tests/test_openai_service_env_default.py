"""Tests for the env-configured LLM provider switch (LLM_PROVIDER + per-provider keys)."""
import pytest
from types import SimpleNamespace

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


_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --- LLM_PROVIDER presets ----------------------------------------------------

def test_provider_anthropic_uses_preset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    client, is_custom = openai_service.get_client()

    assert is_custom is True
    assert str(client.base_url).startswith("https://api.anthropic.com/v1")
    assert openai_service.resolve_model(openai_service.DEFAULT_MODEL) == "claude-haiku-4-5"


def test_provider_gemini_uses_preset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")

    client, is_custom = openai_service.get_client()

    assert is_custom is True
    assert str(client.base_url).startswith(
        "https://generativelanguage.googleapis.com/v1beta/openai/v1"
    ) or str(client.base_url).startswith(
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert openai_service.resolve_model("") == "gemini-2.5-flash"


def test_provider_model_env_overrides_preset_default(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")

    assert openai_service.resolve_model(openai_service.DEFAULT_MODEL) == "claude-sonnet-5"


def test_provider_missing_key_raises_clear_error(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        openai_service.get_client()


def test_unknown_provider_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "chatgpt")

    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        openai_service.get_client()


def test_provider_ollama_is_local(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-unused")  # other keys may be set

    _client, is_custom = openai_service.get_client()

    assert is_custom is False


# --- Generic custom endpoint (advanced) --------------------------------------

def test_generic_custom_endpoint_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("LLM_API_KEY", "sk-env")

    client, is_custom = openai_service.get_client()

    assert is_custom is True
    assert str(client.base_url).startswith("https://api.openai.com/v1")


def test_get_client_prefers_request_override(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    client, is_custom = openai_service.get_client("https://request.example.com", "sk-req")

    assert is_custom is True
    assert str(client.base_url).startswith("https://request.example.com/v1")


def test_get_client_falls_back_to_local_ollama(monkeypatch):
    _clear_env(monkeypatch)

    _client, is_custom = openai_service.get_client()

    assert is_custom is False


# --- resolve_model semantics --------------------------------------------------

def test_resolve_model_keeps_explicit_model(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")

    # Explicitly requested non-default models stay untouched.
    assert openai_service.resolve_model("gemini-2.5-pro") == "gemini-2.5-pro"


def test_resolve_model_keeps_request_override_untouched(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    resolved = openai_service.resolve_model(
        openai_service.DEFAULT_MODEL, has_request_override=True
    )

    assert resolved == openai_service.DEFAULT_MODEL


def test_resolve_model_without_env_returns_default(monkeypatch):
    _clear_env(monkeypatch)

    assert openai_service.resolve_model("") == openai_service.DEFAULT_MODEL
    assert openai_service.resolve_model("qwen2.5vl:7b") == "qwen2.5vl:7b"


# --- Parameter fallbacks for external endpoints -------------------------------

def test_custom_endpoint_drops_unsupported_temperature():
    client = FakeClient(
        failures=[
            RuntimeError(
                "Unsupported value: 'temperature' does not support 0.1 with this model. "
                "Only the default (1) value is supported."
            )
        ]
    )

    openai_service.create_chat_completion_with_token_limit(
        client,
        is_custom=True,
        token_limit=8192,
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
    )

    assert len(client.calls) == 2
    assert client.calls[0]["temperature"] == 0.1
    assert "temperature" not in client.calls[1]
    assert client.calls[1]["max_completion_tokens"] == 8192


def test_custom_endpoint_survives_both_param_fallbacks():
    client = FakeClient(
        failures=[
            RuntimeError("Unknown name \"max_completion_tokens\": Cannot find field."),
            RuntimeError("Unsupported value: 'temperature' is not supported"),
        ]
    )

    openai_service.create_chat_completion_with_token_limit(
        client,
        is_custom=True,
        token_limit=8192,
        model="legacy-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
    )

    assert len(client.calls) == 3
    assert client.calls[2]["max_tokens"] == 8192
    assert "temperature" not in client.calls[2]
