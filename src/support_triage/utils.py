"""Utility and helper functions."""

from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def _load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a provider/model name.

    Args:
        fully_specified_name: String in the form ``provider/model``.

    Returns:
        The initialized chat model.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    loaded = init_chat_model(model, model_provider=provider)
    return cast("BaseChatModel", loaded)
