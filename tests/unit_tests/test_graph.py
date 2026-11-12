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
from support_triage.tools import clear_recorded_routes, recorded_routes, route_ticket
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
    finish_update = next(update["finish"] for update in updates if "finish" in update)
    assert isinstance(finish_update, dict)
    assert finish_update["accepted"] is True
    gate_update = updates[-1]["citation_gate"]
    assert isinstance(gate_update, dict)
    assert gate_update["accepted"] is True


def test_order_9999_is_not_found_and_reply_has_no_tracking_number() -> None:
    reply = "Order 9999 was not found, so there is no shipment to report. Citations: shipping-001"
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
    assert result["accepted"] is True
    assert result["citation_rejection"] == ""


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


def test_shipping_citation_after_billing_retrieval_is_rejected() -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_message(
                "route_ticket",
                {"queue": "billing", "citation_ids": ["shipping-001"]},
                "route-shipping",
            ),
            AIMessage(content="The shipment is still at the warehouse. Citations: shipping-001"),
        ]
    )
    result = _run(model, BILLING_TICKET)
    messages = result["messages"]
    assert isinstance(messages, list)
    assert result["accepted"] is False
    assert result["citation_rejection"]
    assert any(
        isinstance(message, AIMessage)
        and message.response_metadata.get("source") == "citation_gate"
        for message in messages
    )


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


def test_follow_up_without_route_ticket_falls_back_to_human() -> None:
    prior_route = _tool_message(
        "route_ticket",
        {"queue": "billing", "citation_ids": ["billing-001"]},
        "route-old",
    )
    accepted = ToolMessage(
        content='{"accepted": true, "queue": "billing", "citation_ids": ["billing-001"]}',
        name="route_ticket",
        tool_call_id="route-old",
    )
    history = [
        HumanMessage(content=BILLING_TICKET),
        prior_route,
        accepted,
        AIMessage(content="Citations: billing-001"),
        HumanMessage(content="Thanks, nothing else."),
    ]
    model = ScriptedChatModel(responses=[AIMessage(content="Glad that helped.")])
    config: RunnableConfig = {"recursion_limit": 25}

    async def _invoke() -> object:
        with use_chat_model(model):
            return await graph.ainvoke(
                InputState(messages=history),
                config,
                context=Context(),
            )

    result = asyncio.run(_invoke())
    assert isinstance(result, dict)
    assert result["accepted"] is False
    assert result["trajectory_ok"] is False
    assert result["fallback_reason"] == FALLBACK_REASON
    assert recorded_routes() == [{"queue": "human", "citation_ids": []}]


def test_later_ticket_drops_the_earlier_ticket() -> None:
    history = [
        HumanMessage(content=BILLING_TICKET, id="turn-1-human"),
        AIMessage(content="Citations: billing-001", id="turn-1-reply"),
        HumanMessage(content=MISSING_ORDER_TICKET, id="turn-2-human"),
    ]
    model = ScriptedChatModel(responses=[AIMessage(content="Order 9999 was not found.")])
    config: RunnableConfig = {"recursion_limit": 25}

    async def _invoke() -> object:
        with use_chat_model(model):
            return await graph.ainvoke(
                InputState(messages=history),
                config,
                context=Context(),
            )

    result = asyncio.run(_invoke())
    assert isinstance(result, dict)
    messages = result["messages"]
    assert isinstance(messages, list)
    texts = [message.content for message in messages if isinstance(message.content, str)]
    assert BILLING_TICKET not in texts
    assert "Citations: billing-001" not in texts
    assert MISSING_ORDER_TICKET in texts
    assert model.seen
    assert all(BILLING_TICKET not in text for text in model.seen[0])
    assert any(MISSING_ORDER_TICKET in text for text in model.seen[0])


def test_split_model_reply_is_saved_as_one_string() -> None:
    reply = AIMessage(
        content=[
            {
                "type": "text",
                "text": "I have routed your ticket to a",
                "index": 0,
            },
            " human agent for further assistance.",
        ]
    )
    result = _run(ScriptedChatModel(responses=[reply]), BILLING_TICKET)
    messages = result["messages"]
    assert isinstance(messages, list)
    reply_message = next(
        message
        for message in messages
        if isinstance(message, AIMessage)
        and not message.tool_calls
        and message.content != FALLBACK_REASON
    )
    assert reply_message.content == (
        "I have routed your ticket to a human agent for further assistance."
    )


def test_graph_run_drops_a_route_recorded_before_it_starts() -> None:
    route_ticket("shipping", ["shipping-001"])
    model = ScriptedChatModel(responses=[AIMessage(content="I can answer without routing.")])
    result = _run(model, BILLING_TICKET)
    assert result["fallback_reason"] == FALLBACK_REASON
    assert recorded_routes() == [{"queue": "human", "citation_ids": []}]


def test_content_block_ticket_retrieves_the_billing_article() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="I can answer without routing.")])
    config: RunnableConfig = {"recursion_limit": 25}

    async def _invoke() -> object:
        with use_chat_model(model):
            return await graph.ainvoke(
                InputState(
                    messages=[HumanMessage(content=[{"type": "text", "text": BILLING_TICKET}])]
                ),
                config,
                context=Context(),
            )

    result = asyncio.run(_invoke())
    assert isinstance(result, dict)
    passages = result["passages"]
    assert isinstance(passages, list)
    assert [passage["id"] for passage in passages] == ["billing-001"]


def test_step_limit_with_accepted_route_runs_citation_gate() -> None:
    reply = (
        "A second charge is a pending authorization or a duplicate capture. Citations: billing-001"
    )
    model = ScriptedChatModel(
        responses=[
            _tool_message(
                "route_ticket",
                {"queue": "billing", "citation_ids": ["billing-001"]},
                "route-billing",
            ),
            AIMessage(content=reply),
        ]
    )
    result = _run(model, BILLING_TICKET, recursion_limit=5)
    messages = result["messages"]
    assert isinstance(messages, list)
    assert result["accepted"] is True
    assert result["trajectory_ok"] is True
    assert result["citation_rejection"] == ""
    assert recorded_routes() == [{"queue": "billing", "citation_ids": ["billing-001"]}]
