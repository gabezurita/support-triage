"""Utility and helper functions."""

from langchain_core.language_models import BaseChatModel

from .providers import load_chat_model as _load_provider_model


def _load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a provider/model name.

    Args:
        fully_specified_name: String in the form ``provider/model``.

    Returns:
        The initialized chat model.
    """
    return _load_provider_model(fully_specified_name)
