"""Provider selection for live model calls."""

import pytest
from langchain_core.messages import AIMessage

from support_triage.context import Context
from support_triage.judge import judge_reply
from support_triage.providers import load_chat_model, provider_key, redact, resolve_provider
from support_triage.scripted import ScriptedChatModel

KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "SUPPORT_TRIAGE_PROVIDER",
    "SUPPORT_TRIAGE_MODEL",
    "SUPPORT_TRIAGE_LIVE_JUDGE",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


def test_default_provider_is_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    spec = resolve_provider()
    assert spec.name == "gemini"
    assert spec.model_name == "google_genai/gemini-2.5-flash"
    assert spec.key_envs == ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def test_anthropic_and_codex_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SUPPORT_TRIAGE_PROVIDER", "anthropic")
    assert resolve_provider().model_name == "anthropic/claude-sonnet-4-5-20250929"
    monkeypatch.setenv("SUPPORT_TRIAGE_PROVIDER", "codex")
    assert resolve_provider().model_name == "openai/gpt-5.3-codex"
    assert resolve_provider().key_envs == ("OPENAI_API_KEY",)


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SUPPORT_TRIAGE_PROVIDER", "bedrock")
    with pytest.raises(ValueError, match="bedrock"):
        resolve_provider()


def test_provider_key_uses_the_selected_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert provider_key() is None
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert provider_key() == "google-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert provider_key() == "gemini-key"
    monkeypatch.setenv("SUPPORT_TRIAGE_PROVIDER", "codex")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert provider_key() == "openai-key"


def test_context_model_follows_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert Context().model == "google_genai/gemini-2.5-flash"
    monkeypatch.setenv("SUPPORT_TRIAGE_PROVIDER", "codex")
    assert Context().model == "openai/gpt-5.3-codex"
    monkeypatch.setenv("SUPPORT_TRIAGE_MODEL", "openai/gpt-5.4")
    assert Context().model == "openai/gpt-5.4"


def test_load_requires_the_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        load_chat_model("google_genai/gemini-2.5-flash")


def test_codex_load_uses_the_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    seen: dict[str, object] = {}

    def fake_init(model: str, *, model_provider: str, **kwargs: object) -> ScriptedChatModel:
        seen["model"] = model
        seen["provider"] = model_provider
        seen["kwargs"] = kwargs
        return ScriptedChatModel(responses=[])

    monkeypatch.setattr("support_triage.providers.init_chat_model", fake_init)
    load_chat_model("openai/gpt-5.3-codex")
    assert seen == {
        "model": "gpt-5.3-codex",
        "provider": "openai",
        "kwargs": {"api_key": "openai-key", "use_responses_api": True},
    }


def test_load_chat_model_passes_the_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    seen: dict[str, object] = {}

    def fake_init(model: str, *, model_provider: str, **kwargs: object) -> ScriptedChatModel:
        seen["model"] = model
        seen["provider"] = model_provider
        seen["kwargs"] = kwargs
        return ScriptedChatModel(responses=[])

    monkeypatch.setattr("support_triage.providers.init_chat_model", fake_init)
    load_chat_model("google_genai/gemini-2.5-flash")
    assert seen["model"] == "gemini-2.5-flash"
    assert seen["provider"] == "google_genai"
    assert seen["kwargs"] == {
        "api_key": "gemini-key",
        "api_version": "v1beta",
        "temperature": 0,
    }


def test_judge_reply_reads_json() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content='{"grounded": true, "helpful": false}')])
    score = judge_reply(
        model,
        ticket="Why was I charged twice for order 1001?",
        reply="Citations: billing-001",
        citation_ids=["billing-001"],
        articles="[billing-001]\nA duplicate capture is reversed.",
    )
    assert score == {"grounded": True, "helpful": False}


def test_redact_strips_secret_values_and_key_fields() -> None:
    cleaned = redact(
        {
            "note": "prefix gemini-key suffix",
            "api_key": "gemini-key",
            "nested": {"authorization": "Bearer gemini-key", "queue": "billing"},
        },
        ["gemini-key"],
    )
    assert cleaned == {"note": "prefix [redacted] suffix", "nested": {"queue": "billing"}}
