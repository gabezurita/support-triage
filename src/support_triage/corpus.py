"""Deterministic retrieval over the checked-in help center."""

from dataclasses import dataclass
from pathlib import Path

_HELP_CENTER = Path(__file__).resolve().parents[2] / "data" / "help_center"

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing-001": ("charged", "charge", "billing", "invoice", "payment"),
    "shipping-001": ("shipping", "tracking", "where is"),
    "refunds-001": ("refund", "return"),
}


@dataclass(frozen=True)
class Passage:
    """One retrieved help-center article."""

    id: str
    text: str


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
            article = (_HELP_CENTER / f"{article_id}.md").read_text(encoding="utf-8")
            passages.append(Passage(id=article_id, text=article))
    return passages
