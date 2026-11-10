"""Utility and helper functions."""

from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Return the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        text = part.get("text", "")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a provider/model name.

    Args:
        fully_specified_name: String in the form ``provider/model``.

    Returns:
        The initialized chat model.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    loaded = init_chat_model(model, model_provider=provider)
    return cast("BaseChatModel", loaded)
