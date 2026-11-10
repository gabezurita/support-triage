"""Scripted graph runs. No model provider is called."""

import asyncio
from collections.abc import Generator
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from support_triage.context import Context
from support_triage.graph import FALLBACK_REASON, graph, use_chat_model
from support_triage.scripted import ScriptedChatModel
from support_triage.state import InputState
from support_triage.tools import clear_recorded_routes, recorded_routes
from support_triage.trajectory import model_called_route_ticket

BILLING_TICKET = "Why was I charged twice for order 1001?"
MISSING_ORDER_TICKET = "Where is order 9999?"


@pytest.fixture(autouse=True)
def _isolate_routes() -> Generator[None, None, None]:
    clear_recorded_routes()
    yield
    clear_recorded_routes()


def _tool_message(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _run(model: ScriptedChatModel, ticket: str, *, recursion_limit: int = 25) -> dict[str, Any]:
    config: RunnableConfig = {"recursion_limit": recursion_limit}

    async def _invoke() -> object:
        with use_chat_model(model):
            return await graph.ainvoke(
                InputState(messages=[HumanMessage(content=ticket)]),
                config,
                context=Context(),
            )

    result = asyncio.run(_invoke())
    assert isinstance(result, dict)
    return cast("dict[str, Any]", result)


def _reply_text(result: dict[str, Any]) -> str:
    messages = result["messages"]
    assert isinstance(messages, list)
    texts: list[str] = []
    for message in messages:
        if (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content
            and not message.tool_calls
        ):
            texts.append(message.content)
    return texts[-1] if texts else ""


def test_billing_ticket_retrieves_injects_and_routes() -> None:
    reply = (
        "A second charge is a pending authorization or a duplicate capture. Citations: billing-001"
    )
    model = ScriptedChatModel(
        responses=[
            _tool_message("lookup_order", {"order_id": "1001"}, "lookup-1001"),
            _tool_message(
                "route_ticket",
                {"queue": "billing", "citation_ids": ["billing-001"]},
                "route-billing",
            ),
            AIMessage(content=reply),
        ]
    )

    async def _stream() -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        with use_chat_model(model):
            async for update in graph.astream(
                InputState(messages=[HumanMessage(content=BILLING_TICKET)]),
                context=Context(),
                stream_mode="updates",
            ):
                assert isinstance(update, dict)
                collected.append(cast("dict[str, Any]", update))
        return collected

    updates = asyncio.run(_stream())
    assert next(iter(updates[0])) == "retrieve"
    retrieve_update = updates[0]["retrieve"]
    assert isinstance(retrieve_update, dict)
    passages = retrieve_update["passages"]
    assert isinstance(passages, list)
    assert [passage["id"] for passage in passages] == ["billing-001"]
    assert "duplicate capture" in passages[0]["text"]
    assert model.seen
    assert "duplicate capture" in model.seen[0][1]
    assert model.bound_tool_names == ["lookup_order", "route_ticket"]
    assert recorded_routes() == [{"queue": "billing", "citation_ids": ["billing-001"]}]
    final = updates[-1]["finish"]
    assert isinstance(final, dict)
    assert final["accepted"] is True
    assert final["trajectory_ok"] is True


def test_order_9999_is_not_found_and_reply_has_no_tracking_number() -> None:
    reply = "Order 9999 was not found, so there is no shipment to report."
    model = ScriptedChatModel(
        responses=[
            _tool_message("lookup_order", {"order_id": "9999"}, "lookup-9999"),
            _tool_message(
                "route_ticket",
                {"queue": "human", "citation_ids": ["shipping-001"]},
                "route-missing",
            ),
            AIMessage(content=reply),
        ]
    )
    result = _run(model, MISSING_ORDER_TICKET)
    messages = result["messages"]
    assert isinstance(messages, list)
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    lookup = next(message for message in tool_messages if message.name == "lookup_order")
    assert '"found": false' in str(lookup.content)
    assert "9999" in str(lookup.content)
    assert "tracking" not in _reply_text(result).casefold()
    assert recorded_routes() == [{"queue": "human", "citation_ids": ["shipping-001"]}]


def test_missing_route_ticket_defaults_to_human_and_fails_trajectory() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="I can answer without routing.")])
    result = _run(model, BILLING_TICKET)
    messages = result["messages"]
    assert isinstance(messages, list)
    assert result["accepted"] is False
    assert result["trajectory_ok"] is False
    assert result["fallback_reason"] == FALLBACK_REASON
    assert not model_called_route_ticket(messages)
    assert recorded_routes() == [{"queue": "human", "citation_ids": []}]
    assert FALLBACK_REASON in _reply_text(result)


def test_executive_queue_is_rejected_and_records_nothing() -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_message(
                "route_ticket",
                {"queue": "executive", "citation_ids": ["billing-001"]},
                "route-executive",
            ),
            AIMessage(content="Sent to the executive queue."),
        ]
    )
    result = _run(model, BILLING_TICKET)
    assert result["accepted"] is False
    assert result["fallback_reason"] == ""
    assert recorded_routes() == []


def test_step_limit_without_route_ticket_falls_back_to_human() -> None:
    model = ScriptedChatModel(
        responses=[_tool_message("lookup_order", {"order_id": "1001"}, "lookup-loop")]
    )
    result = _run(model, BILLING_TICKET, recursion_limit=4)
    messages = result["messages"]
    assert isinstance(messages, list)
    assert result["accepted"] is False
    assert result["trajectory_ok"] is False
    assert result["fallback_reason"] == FALLBACK_REASON
    assert not model_called_route_ticket(messages)
    assert recorded_routes() == [{"queue": "human", "citation_ids": []}]


def test_ticket_text_handles_content_blocks() -> None:
    from support_triage.graph import ticket_text

    msg = HumanMessage(content=[{"type": "text", "text": "Why was I charged twice?"}])
    assert ticket_text([msg]) == "Why was I charged twice?"
