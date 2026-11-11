"""Choose Gemini, Anthropic, or Codex for a live model call.

The offline tests never call this loader. They install a scripted model on the
graph. A live run reads ``SUPPORT_TRIAGE_PROVIDER`` and that provider's key.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "authorization",
        "google_api_key",
        "openai_api_key",
        "token",
        "x-goog-api-key",
    }
)


@dataclass(frozen=True)
class ProviderSpec:
    """One configured chat provider."""

    name: str
    langchain_provider: str
    model: str
    key_envs: tuple[str, ...]

    @property
    def model_name(self) -> str:
        """Return the ``provider/model`` string the graph stores."""
        return f"{self.langchain_provider}/{self.model}"


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        "gemini",
        "google_genai",
        "gemini-2.5-flash",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "anthropic": ProviderSpec(
        "anthropic",
        "anthropic",
        "claude-sonnet-4-5-20250929",
        ("ANTHROPIC_API_KEY",),
    ),
    "codex": ProviderSpec(
        "codex",
        "openai",
        "gpt-5.3-codex",
        ("OPENAI_API_KEY",),
    ),
}


def resolve_provider(name: str | None = None) -> ProviderSpec:
    """Return the provider selected by name or by the environment.

    Args:
        name: A provider name. When omitted, ``SUPPORT_TRIAGE_PROVIDER`` is
            read and defaults to ``gemini``.

    Returns:
        The matching provider.

    Raises:
        ValueError: The name is not gemini, anthropic, or codex.
    """
    chosen = name if name is not None else os.environ.get("SUPPORT_TRIAGE_PROVIDER", "gemini")
    spec = PROVIDERS.get(chosen)
    if spec is None:
        message = f"Unknown provider {chosen}. Choose gemini, anthropic, or codex."
        raise ValueError(message)
    return spec


def provider_key(spec: ProviderSpec | None = None) -> str | None:
    """Return the first configured key for a provider.

    Args:
        spec: The provider to read. Defaults to the selected provider.

    Returns:
        The key text, or ``None`` when every listed variable is empty.
    """
    chosen = resolve_provider() if spec is None else spec
    for env_name in chosen.key_envs:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model and pass the key for its provider.

    Args:
        fully_specified_name: A ``langchain-provider/model`` string.

    Returns:
        The initialized chat model.

    Raises:
        ValueError: The provider prefix is unknown, or its key is missing.
    """
    langchain_provider, model = fully_specified_name.split("/", maxsplit=1)
    spec = _spec_for_langchain(langchain_provider)
    key = provider_key(spec)
    if key is None:
        names = " or ".join(spec.key_envs)
        message = f"{spec.name} needs {names}"
        raise ValueError(message)
    kwargs: dict[str, Any] = {"api_key": key}
    if spec.name == "gemini":
        kwargs["api_version"] = "v1beta"
        kwargs["temperature"] = 0
    elif spec.name == "codex":
        kwargs["use_responses_api"] = True
    else:
        kwargs["temperature"] = 0
    loaded = init_chat_model(model, model_provider=langchain_provider, **kwargs)
    return cast("BaseChatModel", loaded)


def redact(value: Any, secrets: Sequence[str]) -> Any:
    """Remove key fields and replace secret text.

    Args:
        value: A JSON-like value from a model run.
        secrets: Secret strings to replace with ``[redacted]``.

    Returns:
        A copy safe to check in.
    """
    secret_values = [secret for secret in secrets if secret]
    if isinstance(value, str):
        for secret in secret_values:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, list):
        return [redact(item, secret_values) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).casefold() in _SECRET_FIELDS:
                continue
            cleaned[str(key)] = redact(item, secret_values)
        return cleaned
    return value


def _spec_for_langchain(langchain_provider: str) -> ProviderSpec:
    for spec in PROVIDERS.values():
        if spec.langchain_provider == langchain_provider:
            return spec
    message = f"Unknown model provider {langchain_provider}."
    raise ValueError(message)
