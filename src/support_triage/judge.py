"""Groundedness agreement and the live-judge switch."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from .providers import provider_key


def groundedness_agreement(human: Sequence[bool], judge: Sequence[bool]) -> float:
    """Return the fraction of groundedness labels the judge matched.

    Args:
        human: Hand-labeled groundedness values.
        judge: Judge labels in the same order.

    Returns:
        Matches divided by the number of labels.

    Raises:
        ValueError: The two sequences differ in length, or either one is empty.
    """
    if not human or len(human) != len(judge):
        message = "human and judge labels must be the same non-empty length"
        raise ValueError(message)
    matches = sum(left == right for left, right in zip(human, judge, strict=True))
    return matches / len(human)


def live_judge_status() -> str:
    """Return whether a live groundedness judge would run.

    The offline suite does not call a provider. A live run needs the selected
    provider's key and ``SUPPORT_TRIAGE_LIVE_JUDGE=1``.

    Returns:
        ``skipped`` or ``enabled``.
    """
    if provider_key() is None:
        return "skipped"
    if os.environ.get("SUPPORT_TRIAGE_LIVE_JUDGE") != "1":
        return "skipped"
    return "enabled"


def judge_reply(
    model: BaseChatModel,
    *,
    ticket: str,
    reply: str,
    citation_ids: Sequence[str],
    articles: str,
) -> dict[str, bool]:
    """Score one reply for groundedness and helpfulness.

    Args:
        model: The chat model that returns the score. Tests pass a scripted model.
        ticket: The customer ticket.
        reply: The reply being judged.
        citation_ids: Article ids named by the reply.
        articles: Help-center text the reply was allowed to use.

    Returns:
        ``grounded`` and ``helpful`` booleans.

    Raises:
        ValueError: The model did not return those two booleans as JSON.
    """
    cited = ", ".join(citation_ids) if citation_ids else "(none)"
    prompt = (
        "Score this support reply. Grounded means every claim and citation is "
        "supported by the help-center text. Helpful means the reply answers the "
        "ticket without inventing a tracking number, a refund, or a charge reversal.\n"
        "Return JSON only, with boolean fields grounded and helpful.\n\n"
        f"Help center:\n{articles}\n\n"
        f"Ticket:\n{ticket}\n\n"
        f"Reply:\n{reply}\n\n"
        f"Citation ids: {cited}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return _parse_score(_message_text(response))


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def _parse_score(text: str) -> dict[str, bool]:
    body = text.strip()
    if body.startswith("```"):
        lines = [line for line in body.splitlines() if not line.startswith("```")]
        body = "\n".join(lines).strip()
    parsed: Any = json.loads(body)
    if not isinstance(parsed, dict):
        message = "The judge response must be a JSON object."
        raise ValueError(message)
    score = cast("dict[str, Any]", parsed)
    grounded = score.get("grounded")
    helpful = score.get("helpful")
    if not isinstance(grounded, bool) or not isinstance(helpful, bool):
        message = "The judge response must include boolean grounded and helpful fields."
        raise ValueError(message)
    return {"grounded": grounded, "helpful": helpful}
