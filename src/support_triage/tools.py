"""Order lookup and ticket routing.

These are the only tools the agent may call. ``route_ticket`` records a route.
It does not send mail or issue a refund.
"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, TypedDict

AS_OF = date(2026, 11, 9)
QUEUES = frozenset({"billing", "shipping", "refunds", "human"})

_ORDERS: dict[str, dict[str, str | bool]] = {
    "1001": {
        "order_id": "1001",
        "found": True,
        "order_date": (AS_OF - timedelta(days=5)).isoformat(),
        "delivery_date": (AS_OF - timedelta(days=2)).isoformat(),
        "status": "delivered",
    },
    "1002": {
        "order_id": "1002",
        "found": True,
        "order_date": (AS_OF - timedelta(days=100)).isoformat(),
        "delivery_date": (AS_OF - timedelta(days=90)).isoformat(),
        "status": "delivered",
    },
}


class RouteRecord(TypedDict):
    """A route that was accepted."""

    queue: str
    citation_ids: list[str]


_routes: list[RouteRecord] = []


def lookup_order(order_id: str) -> dict[str, str | bool]:
    """Return a copy of an order fixture, or a not-found result.

    Args:
        order_id: The order id from the ticket.

    Returns:
        The fixture for ``1001`` or ``1002``, otherwise ``found`` false and no
        shipment fields.
    """
    row = _ORDERS.get(order_id)
    if row is None:
        return {"order_id": order_id, "found": False}
    return dict(row)


def route_ticket(queue: str, citation_ids: list[str]) -> dict[str, str | bool | list[str]]:
    """Record a route into an allowed queue.

    Args:
        queue: One of billing, shipping, refunds, or human.
        citation_ids: Article ids the reply cites.

    Returns:
        Whether the queue was accepted, and the recorded citation ids.
    """
    if queue not in QUEUES:
        return {"accepted": False, "queue": queue}
    record: RouteRecord = {"queue": queue, "citation_ids": list(citation_ids)}
    _routes.append(record)
    return {"accepted": True, "queue": queue, "citation_ids": list(citation_ids)}


def recorded_routes() -> list[RouteRecord]:
    """Return copies of the routes recorded in this process."""
    return [
        {"queue": route["queue"], "citation_ids": list(route["citation_ids"])} for route in _routes
    ]


def clear_recorded_routes() -> None:
    """Drop recorded routes. Tests use this to isolate cases."""
    _routes.clear()


TOOLS: list[Callable[..., Any]] = [lookup_order, route_ticket]
