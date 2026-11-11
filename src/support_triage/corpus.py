"""Deterministic retrieval over the checked-in help center."""

import importlib.resources
from dataclasses import dataclass

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing-001": ("charged", "charge", "billing", "invoice", "payment"),
    "shipping-001": ("shipping", "tracking", "track", "where is"),
    "refunds-001": ("refund", "return"),
}


@dataclass(frozen=True)
class Passage:
    """One retrieved help-center article."""

    id: str
    text: str


def _read_article(article_id: str) -> str:
    """Read article text from package resources."""
    ref = importlib.resources.files("support_triage.help_center").joinpath(f"{article_id}.md")
    return ref.read_text(encoding="utf-8")


def retrieve(ticket: str) -> list[Passage]:
    """Return help-center passages whose keywords appear in the ticket.

    Args:
        ticket: The support ticket text. An empty ticket matches nothing.

    Returns:
        Matching passages in stable article order.
    """
    text = ticket.casefold().strip()
    if not text:
        return []
    passages: list[Passage] = []
    for article_id, keywords in _KEYWORDS.items():
        if any(keyword.casefold() in text for keyword in keywords):
            passages.append(Passage(id=article_id, text=_read_article(article_id)))
    return passages
