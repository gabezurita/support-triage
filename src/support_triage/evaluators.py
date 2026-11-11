"""Deterministic checks for queue, citations, and trajectory."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage

ARTICLE_ID = re.compile(r"\b(?:billing|shipping|refunds)-\d+\b")

_WITHHELD = (
    "not approved",
    "not be approved",
    "cannot approve",
    "can't approve",
    "withhold",
    "outside the 30",
    "outside that window",
)


def queue_correct(actual: str, expected: str) -> bool:
    """Return whether the recorded queue matches the expected queue.

    Args:
        actual: Queue recorded for the ticket.
        expected: Queue the dataset row requires.

    Returns:
        True when the two queues are the same string.
    """
    return actual == expected


def citation_subset_ok(
    citation_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    reply: str,
) -> bool:
    """Return whether the route citations are grounded and match the reply.

    Args:
        citation_ids: Ids passed to ``route_ticket``.
        retrieved_ids: Ids retrieved for the ticket.
        reply: The user-facing reply text.

    Returns:
        True when the route ids are a subset of the retrieved ids and the reply
        cites that same set.
    """
    cited = set(citation_ids)
    reply_ids = set(ARTICLE_ID.findall(reply))
    return cited <= set(retrieved_ids) and reply_ids == cited


def trajectory_ok(
    messages: Sequence[AnyMessage],
    *,
    lookup_required: bool,
    passages_present: bool,
) -> bool:
    """Return whether the model trajectory meets the dataset row.

    Args:
        messages: The conversation, including model tool calls.
        lookup_required: Whether this row must call ``lookup_order``.
        passages_present: Whether retrieval already stored passages on state.

    Returns:
        True when ``route_ticket`` was called, lookup was called if required,
        and passages are already on state.
    """
    names = _tool_names(messages)
    if "route_ticket" not in names:
        return False
    if lookup_required and "lookup_order" not in names:
        return False
    return passages_present


def reply_approves_refund(reply: str) -> bool:
    """Return whether the reply approves or issues a refund.

    Args:
        reply: The user-facing reply text.

    Returns:
        True when the reply grants the refund. A withheld decision is false.
    """
    text = reply.casefold()
    if any(phrase in text for phrase in _WITHHELD):
        return False
    return "approv" in text or "refund has been issued" in text or "was refunded" in text


def late_refund_ok(queue: str, reply: str) -> bool:
    """Return whether a late refund was withheld and sent to human.

    Args:
        queue: The recorded queue.
        reply: The user-facing reply text.

    Returns:
        True only when the queue is human and the reply does not approve the refund.
    """
    return queue == "human" and not reply_approves_refund(reply)


def grounded_outcome(accepted: bool, citations_ok: bool) -> bool:
    """Return whether the run avoided accepting an ungrounded citation.

    Args:
        accepted: Whether the graph accepted the reply.
        citations_ok: Whether the citation subset check passed.

    Returns:
        True when the citations are grounded or the ungrounded reply was not accepted.
    """
    if citations_ok:
        return True
    return not accepted


def model_reply_text(messages: Sequence[AnyMessage]) -> str:
    """Return the last model reply, skipping a citation-gate rejection.

    Args:
        messages: The conversation.

    Returns:
        The reply text, or an empty string when the model never wrote one.
    """
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        if not isinstance(message.content, str) or not message.content:
            continue
        if message.response_metadata.get("source") == "citation_gate":
            continue
        texts.append(message.content)
    return texts[-1] if texts else ""


def _tool_names(messages: Sequence[AnyMessage]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            names.extend(str(call.get("name", "")) for call in message.tool_calls)
    return names
