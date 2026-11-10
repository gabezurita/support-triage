"""Trajectory checks over model tool calls."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage


def model_called_route_ticket(messages: Sequence[AnyMessage]) -> bool:
    """Return whether a model message called ``route_ticket``.

    Args:
        messages: The conversation, including tool calls and tool results.

    Returns:
        True when an assistant message contains a ``route_ticket`` tool call.
        A graph fallback that calls the function directly does not count.
    """
    return any(isinstance(message, AIMessage) and _has_route_call(message) for message in messages)


def _has_route_call(message: AIMessage) -> bool:
    return any(call.get("name") == "route_ticket" for call in message.tool_calls)
