"""Scripted chat model for offline graph runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Return a fixed list of assistant messages, in order.

    The last message repeats if the graph asks for another turn. Tests use this
    so the suite never calls a model provider.
    """

    responses: list[AIMessage]
    seen: list[list[str]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    @override
    def _llm_type(self) -> str:
        return "scripted-support-triage"

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        """Record the bound tool names and return this model.

        Args:
            tools: Tools the graph binds for this turn.
            tool_choice: Ignored. The script chooses every message.
            **kwargs: Ignored provider options.

        Returns:
            This model, so later invocations stay on the script.
        """
        del tool_choice, kwargs
        self.bound_tool_names = [_tool_name(tool) for tool in tools]
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.seen.append([_message_text(message) for message in messages])
        index = min(len(self.seen) - 1, len(self.responses) - 1)
        message = self.responses[index].model_copy(update={"id": str(uuid4())})
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    fallback = getattr(tool, "__name__", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    return type(tool).__name__


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)
