"""Order lookup and ticket routing without a model."""

from collections.abc import Generator
from datetime import date

import pytest

from support_triage.tools import (
    AS_OF,
    clear_recorded_routes,
    lookup_order,
    recorded_routes,
    route_ticket,
)


@pytest.fixture(autouse=True)
def _isolate_routes() -> Generator[None, None, None]:
    clear_recorded_routes()
    yield
    clear_recorded_routes()


def test_lookup_order_1001_returns_a_recent_fixture() -> None:
    result = lookup_order("1001")
    assert result["found"] is True
    assert result["order_id"] == "1001"
    delivery = date.fromisoformat(str(result["delivery_date"]))
    assert (AS_OF - delivery).days < 30
    result["status"] = "cancelled"
    assert lookup_order("1001")["status"] != "cancelled"


def test_lookup_order_1002_was_delivered_90_days_before_as_of() -> None:
    result = lookup_order("1002")
    assert result["found"] is True
    delivery = date.fromisoformat(str(result["delivery_date"]))
    order = date.fromisoformat(str(result["order_date"]))
    assert (AS_OF - delivery).days == 90
    assert order <= delivery


def test_lookup_order_9999_is_not_found_without_shipment_fields() -> None:
    result = lookup_order("9999")
    assert result == {"order_id": "9999", "found": False}


@pytest.mark.parametrize("queue", ["billing", "shipping", "refunds", "human"])
def test_route_ticket_records_each_valid_queue(queue: str) -> None:
    result = route_ticket(queue, ["billing-001"])
    assert result["accepted"] is True
    assert recorded_routes() == [{"queue": queue, "citation_ids": ["billing-001"]}]


def test_route_ticket_rejects_unknown_queue_and_records_nothing() -> None:
    route_ticket("billing", ["billing-001"])
    before = recorded_routes()
    result = route_ticket("executive", ["billing-001"])
    assert result["accepted"] is False
    assert recorded_routes() == before


def test_route_ticket_does_not_change_the_order_fixture() -> None:
    before = lookup_order("1001")
    route_ticket("refunds", ["refunds-001"])
    assert lookup_order("1001") == before


def test_async_task_inherits_recorded_routes() -> None:
    import asyncio

    async def _simulate_task() -> None:
        route_ticket("shipping", ["shipping-001"])

    asyncio.run(_simulate_task())
    assert recorded_routes() == [{"queue": "shipping", "citation_ids": ["shipping-001"]}]

