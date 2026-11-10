"""Retrieve the help center, then run the tool loop.

The first node calls ``support_triage.corpus.retrieve``. The following nodes are
the model-and-tools cycle, bound only to ``lookup_order`` and ``route_ticket``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from . import corpus
from .context import Context
from .state import InputState, State
from .tools import lookup_order, route_ticket
from .trajectory import model_called_route_ticket
from .utils import load_chat_model

FALLBACK_REASON = (
    "The model omitted route_ticket or hit the step limit. The ticket was routed to human."
)

_model_override: BaseChatModel | None = None


@contextmanager
def use_chat_model(model: BaseChatModel) -> Iterator[None]:
    """Point the graph at a scripted model for one offline run.

    Args:
        model: The model ``call_model`` should bind and invoke.

    Yields:
        Control back to the caller while the override is installed.
    """
    global _model_override
    previous = _model_override
    _model_override = model
    try:
        yield
    finally:
        _model_override = previous


def _chat_model(name: str) -> BaseChatModel:
    if _model_override is not None:
        return _model_override
    return load_chat_model(name)


def ticket_text(messages: Sequence[AnyMessage]) -> str:
    """Join the human messages into the ticket string retrieval sees.

    Args:
        messages: The conversation so far.

    Returns:
        The ticket text, or an empty string when there is no human message.
    """
    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            parts.append(message.content)
    return "\n".join(parts)


def render_passages(passages: Sequence[dict[str, str]]) -> str:
    """Render retrieved passages as model input.

    Args:
        passages: Passage records with ``id`` and ``text``.

    Returns:
        A single text block the model call prepends to the conversation.
    """
    if not passages:
        return "Retrieved help-center passages:\n(none)"
    blocks = [f"[{passage['id']}]\n{passage['text']}" for passage in passages]
    return "Retrieved help-center passages:\n\n" + "\n\n".join(blocks)


def retrieve(state: State) -> dict[str, list[dict[str, str]]]:
    """Retrieve help-center passages and store them on state.

    Args:
        state: The current conversation state.

    Returns:
        The passage records for this ticket.
    """
    passages = [
        {"id": passage.id, "text": passage.text}
        for passage in corpus.retrieve(ticket_text(state.messages))
    ]
    return {"passages": passages}


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Call the model with retrieved passages and the triage tools.

    On the last allowed step, the route decision is applied here. Scheduling
    another node would cross the recursion limit.

    Args:
        state: The current conversation state.
        runtime: Runtime context, including the model name and system prompt.

    Returns:
        The model message, and on the last step the route outcome as well.
    """
    model = _chat_model(runtime.context.model).bind_tools([lookup_order, route_ticket])
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    response = cast(
        "AIMessage",
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                {"role": "system", "content": render_passages(state.passages)},
                *state.messages,
            ]
        ),
    )
    # Stop inside this node once fewer than three steps remain. A tool round
    # trip would otherwise be cut off before the fallback can run.
    must_stop = state.is_last_step or state.remaining_steps <= 2
    if must_stop and response.tool_calls:
        response = AIMessage(
            content="Sorry, I could not finish the ticket in the specified number of steps."
        )
    if not must_stop:
        return {"messages": [response]}
    outcome = finish(State(messages=[*state.messages, response]))
    extra = outcome.pop("messages", [])
    return {**outcome, "messages": [response, *extra]}


def _tool_payload(message: ToolMessage) -> dict[str, Any]:
    content = message.content
    if not isinstance(content, str):
        return {}
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return cast("dict[str, Any]", loaded)
    return {}


def _route_was_accepted(messages: Sequence[AnyMessage]) -> bool:
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and message.name == "route_ticket"
            and _tool_payload(message).get("accepted") is True
        ):
            return True
    return False


def finish(state: State) -> dict[str, Any]:
    """Accept a completed route, or fall back when ``route_ticket`` was skipped.

    A rejected queue such as ``executive`` records nothing. A missing
    ``route_ticket`` call, including a step-limit stop, routes to ``human`` and
    fails the trajectory check.

    Args:
        state: The conversation after the model has stopped calling tools.

    Returns:
        Acceptance, trajectory, and an optional fallback message.
    """
    called = model_called_route_ticket(state.messages)
    if _route_was_accepted(state.messages):
        return {"accepted": True, "trajectory_ok": True, "fallback_reason": ""}
    if called:
        return {"accepted": False, "trajectory_ok": True, "fallback_reason": ""}
    route_ticket("human", [])
    return {
        "accepted": False,
        "trajectory_ok": False,
        "fallback_reason": FALLBACK_REASON,
        "messages": [AIMessage(content=FALLBACK_REASON)],
    }


builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node("retrieve", retrieve)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode([lookup_order, route_ticket]))
builder.add_node("finish", finish)
builder.add_edge("__start__", "retrieve")
builder.add_edge("retrieve", "call_model")


def route_model_output(state: State) -> Literal["__end__", "finish", "tools"]:
    """Choose the next node from the model's last message.

    Args:
        state: The current conversation state.

    Returns:
        ``tools`` when the model called a tool and steps remain, ``__end__`` when
        this step already applied the route decision, otherwise ``finish``.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        message = f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        raise ValueError(message)
    if state.remaining_steps <= 2:
        return "__end__"
    if not last_message.tool_calls:
        return "finish"
    return "tools"


builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
builder.add_edge("finish", "__end__")
graph = builder.compile(name="support-triage")
