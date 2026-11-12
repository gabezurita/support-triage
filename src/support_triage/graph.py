"""Retrieve the help center, then run the tool loop.

The first node calls ``support_triage.corpus.retrieve``. The following nodes are
the model-and-tools cycle, bound only to ``lookup_order`` and ``route_ticket``.
A citation gate then rejects a reply whose article ids were not retrieved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from support_triage import corpus
from support_triage.context import Context
from support_triage.evaluators import citation_subset_ok, model_reply_text
from support_triage.state import InputState, State
from support_triage.tools import clear_recorded_routes, lookup_order, route_ticket
from support_triage.trajectory import model_called_route_ticket
from support_triage.utils import _load_chat_model

FALLBACK_REASON = (
    "The model omitted route_ticket or hit the step limit. The ticket was routed to human."
)
CITATION_REJECTION = "Rejected: the reply cites an article that was not retrieved."

_model_override: ContextVar[BaseChatModel | None] = ContextVar(
    "support_triage_model_override", default=None
)


@contextmanager
def use_chat_model(model: BaseChatModel) -> Iterator[None]:
    """Point the graph at a scripted model for one offline run.

    Args:
        model: The scripted model the call_model node binds and invokes.

    Yields:
        Control back to the caller while the override is installed.
    """
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)


def _chat_model(name: str) -> BaseChatModel:
    override = _model_override.get()
    if override is not None:
        return override
    return _load_chat_model(name)


def _ticket_text(messages: Sequence[AnyMessage]) -> str:
    """Join the human messages into the ticket string retrieval sees.

    Args:
        messages: The conversation so far.

    Returns:
        The ticket text, or an empty string when there is no human message.
    """
    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
    return "\n".join(parts)


def _render_passages(passages: Sequence[dict[str, str]]) -> str:
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


def _current_turn(messages: Sequence[AnyMessage]) -> Sequence[AnyMessage]:
    """Return the messages from the latest human message onward.

    Args:
        messages: The conversation so far, including older turns.

    Returns:
        The current turn. The whole list when it has no human message.
    """
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index
    return messages[start:]


def _earlier_turn_removals(messages: Sequence[AnyMessage]) -> list[RemoveMessage]:
    """Mark messages from earlier tickets for removal.

    Args:
        messages: The thread, including the latest human message.

    Returns:
        Removals for every message before the current turn.
    """
    current_ids = {message.id for message in _current_turn(messages) if message.id}
    return [
        RemoveMessage(id=message.id)
        for message in messages
        if message.id and message.id not in current_ids
    ]


def _retrieve(state: State) -> dict[str, Any]:
    """Retrieve help-center passages for the latest ticket.

    Earlier tickets are removed so this run's state and the model see one ticket.
    The previous route fields are cleared until this ticket finishes.

    Args:
        state: The current conversation state.

    Returns:
        Passages for this ticket, cleared route fields, and removals for earlier tickets.
    """
    clear_recorded_routes()
    turn = _current_turn(state.messages)
    passages = [
        {"id": passage.id, "text": passage.text} for passage in corpus.retrieve(_ticket_text(turn))
    ]
    update: dict[str, Any] = {
        "passages": passages,
        "accepted": False,
        "trajectory_ok": False,
        "fallback_reason": "",
        "citation_rejection": "",
    }
    expired = _earlier_turn_removals(state.messages)
    if expired:
        update["messages"] = expired
    return update


async def _call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
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
                {"role": "system", "content": _render_passages(state.passages)},
                *_current_turn(state.messages),
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
    return _finalize_stop(state, response)


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


def _finish(state: State) -> dict[str, Any]:
    """Accept a completed route, or fall back when ``route_ticket`` was skipped.

    A rejected queue such as ``executive`` records nothing. A missing
    ``route_ticket`` call in the current turn, including a step-limit stop,
    routes to ``human`` and fails the trajectory check. Older turns are ignored.

    Args:
        state: The conversation after the model has stopped calling tools.

    Returns:
        Acceptance, trajectory, and an optional fallback message.
    """
    turn = _current_turn(state.messages)
    called = model_called_route_ticket(turn)
    if _route_was_accepted(turn):
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


def _accepted_citation_ids(messages: Sequence[AnyMessage]) -> list[str]:
    results = {
        message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)
    }
    chosen: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call.get("name") != "route_ticket":
                continue
            call_id = call.get("id")
            if not call_id:
                continue
            result = results.get(str(call_id))
            if result is None or _tool_payload(result).get("accepted") is not True:
                continue
            raw_ids = call.get("args", {}).get("citation_ids", [])
            if isinstance(raw_ids, list):
                chosen = [str(item) for item in raw_ids]
    return chosen


def _citation_gate(state: State) -> dict[str, Any]:
    """Reject a reply whose citations were not retrieved.

    The ids passed to ``route_ticket`` must match the ids in the reply and be
    a subset of the retrieved passage ids. A failure leaves the recorded route
    in place and does not accept the reply.

    Args:
        state: The conversation after a route was accepted.

    Returns:
        A cleared rejection, or a rejected reply.
    """
    turn = _current_turn(state.messages)
    retrieved_ids = [passage["id"] for passage in state.passages]
    if citation_subset_ok(
        _accepted_citation_ids(turn),
        retrieved_ids,
        model_reply_text(turn),
    ):
        return {"accepted": True, "citation_rejection": ""}
    return {
        "accepted": False,
        "citation_rejection": CITATION_REJECTION,
        "messages": [
            AIMessage(content=CITATION_REJECTION, response_metadata={"source": "citation_gate"})
        ],
    }


def _finalize_stop(state: State, response: AIMessage) -> dict[str, Any]:
    """Apply the route decision and citation gate inside the last allowed step.

    Args:
        state: The state before the final model message is appended.
        response: The model message for this step.

    Returns:
        The combined state update, including any gate rejection.
    """
    outcome = _finish(State(messages=[*state.messages, response]))
    extra = outcome.pop("messages", [])
    messages: list[AnyMessage] = [response]
    if isinstance(extra, list):
        messages.extend(message for message in extra if isinstance(message, AIMessage))
    if not outcome.get("accepted"):
        return {**outcome, "messages": messages}
    preview = State(messages=[*state.messages, *messages], passages=list(state.passages))
    gated = _citation_gate(preview)
    gate_extra = gated.pop("messages", [])
    if isinstance(gate_extra, list):
        messages.extend(message for message in gate_extra if isinstance(message, AIMessage))
    return {**outcome, **gated, "messages": messages}


builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node("retrieve", _retrieve)
builder.add_node("call_model", _call_model)
builder.add_node("tools", ToolNode([lookup_order, route_ticket]))
builder.add_node("finish", _finish)
builder.add_node("citation_gate", _citation_gate)
builder.add_edge("__start__", "retrieve")
builder.add_edge("retrieve", "call_model")


def _route_model_output(state: State) -> Literal["__end__", "finish", "tools"]:
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


def _route_after_finish(state: State) -> Literal["citation_gate", "__end__"]:
    """Send an accepted route through the citation gate.

    Args:
        state: The conversation after finish.

    Returns:
        ``citation_gate`` when a route was accepted, otherwise ``__end__``.
    """
    if state.accepted:
        return "citation_gate"
    return "__end__"


builder.add_conditional_edges("call_model", _route_model_output)
builder.add_edge("tools", "call_model")
builder.add_conditional_edges("finish", _route_after_finish)
builder.add_edge("citation_gate", "__end__")
graph = builder.compile(name="support-triage")
