"""Groundedness agreement and the live-judge switch."""

from __future__ import annotations

import os
from collections.abc import Sequence


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

    The offline suite does not call a provider. A live run needs both
    ``ANTHROPIC_API_KEY`` and ``SUPPORT_TRIAGE_LIVE_JUDGE=1``.

    Returns:
        ``skipped`` or ``enabled``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "skipped"
    if os.environ.get("SUPPORT_TRIAGE_LIVE_JUDGE") != "1":
        return "skipped"
    return "enabled"
