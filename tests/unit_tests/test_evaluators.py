"""Dataset scores, trajectory checks, and fixture-judge agreement."""

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from support_triage.evaluate import (
    fixture_agreement,
    load_dataset,
    load_human_labels,
    run_dataset,
)
from support_triage.evaluators import (
    citation_subset_ok,
    late_refund_ok,
    trajectory_ok,
)
from support_triage.graph import graph
from support_triage.judge import groundedness_agreement, live_judge_status

NOTE = Path(__file__).resolve().parents[2] / "docs" / "evaluations" / "EVAL-001-citation-gate.md"


def test_human_labels_cover_two_variants_of_each_scenario() -> None:
    labels = load_human_labels()
    assert len(labels) == 8
    scenarios = {str(label["scenario"]) for label in labels}
    assert scenarios == {
        "billing-double-charge",
        "late-refund",
        "missing-order-9999",
        "multi-article",
    }
    for scenario in scenarios:
        variants = sorted(
            str(label["variant"]) for label in labels if label["scenario"] == scenario
        )
        assert variants == ["grounded", "ungrounded"]


def test_late_refund_fails_when_approved_and_passes_when_withheld() -> None:
    approved = "Your refund is approved and has been issued."
    withheld = "The refund is not approved. The 30 day window has passed."
    assert late_refund_ok("refunds", approved) is False
    assert late_refund_ok("human", withheld) is True


def test_trajectory_missing_route_ticket_fails() -> None:
    messages: list[AnyMessage] = [
        HumanMessage(content="Where is order 9999?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup_order",
                    "args": {"order_id": "9999"},
                    "id": "lookup-1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Order 9999 was not found."),
    ]
    assert trajectory_ok(messages, lookup_required=True, passages_present=True) is False


def test_trajectory_missing_required_lookup_fails() -> None:
    messages: list[AnyMessage] = [
        HumanMessage(content="Where is order 9999?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "route_ticket",
                    "args": {"queue": "human", "citation_ids": ["shipping-001"]},
                    "id": "route-1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Order 9999 was not found."),
    ]
    assert trajectory_ok(messages, lookup_required=True, passages_present=True) is False


def test_citation_outside_retrieved_ids_fails() -> None:
    assert citation_subset_ok(["shipping-001"], ["billing-001"], "Citations: shipping-001") is False


def test_fixture_judge_agreement_is_computed() -> None:
    human = [bool(label["grounded"]) for label in load_human_labels()]
    assert fixture_agreement() == 1.0
    flipped = list(human)
    flipped[0] = not flipped[0]
    assert groundedness_agreement(human, flipped) == 7 / 8


def test_live_judge_skips_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORT_TRIAGE_LIVE_JUDGE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert live_judge_status() == "skipped"


def test_live_judge_stays_offline_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPPORT_TRIAGE_LIVE_JUDGE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    assert live_judge_status() == "skipped"
    assert os.environ["ANTHROPIC_API_KEY"] == "present"


def test_live_judge_enables_with_key_and_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORT_TRIAGE_LIVE_JUDGE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    assert live_judge_status() == "enabled"


def test_dataset_before_scores_and_note() -> None:
    summary = run_dataset()
    by_id = {case.id: case for case in summary.cases}
    assert [case.id for case in summary.cases] == [
        "billing-double-charge",
        "late-refund",
        "missing-order-9999",
        "multi-article",
    ]
    assert by_id["multi-article"].citation_subset is False
    assert by_id["multi-article"].accepted is True
    assert by_id["billing-double-charge"].queue_ok is True
    assert by_id["late-refund"].queue_ok is True
    assert summary.queue_accuracy == 1
    assert summary.trajectory_rate == 1
    assert summary.citation_subset_rate == 0.75
    assert summary.citation_score == 0.75
    assert summary.ungrounded_accepted == 1
    note = NOTE.read_text(encoding="utf-8")
    assert "## After" not in note
    assert "0.75" in note
    for row in load_dataset():
        assert str(row["id"]) in note
    assert "citation_gate" not in graph.nodes
