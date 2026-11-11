"""Offline dataset run for the triage graph."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage

from .context import Context
from .evaluators import (
    citation_subset_ok,
    grounded_outcome,
    model_reply_text,
    queue_correct,
    trajectory_ok,
)
from .graph import graph, use_chat_model
from .judge import groundedness_agreement
from .scripted import ScriptedChatModel
from .state import InputState
from .tools import clear_recorded_routes, recorded_routes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "dataset.json"
HUMAN_LABELS_PATH = REPO_ROOT / "data" / "evaluation" / "human_labels.json"
JUDGE_FIXTURE_PATH = REPO_ROOT / "data" / "evaluation" / "judge_fixture.json"


@dataclass(frozen=True)
class CaseResult:
    """Scores for one dataset row."""

    id: str
    queue_ok: bool
    citation_subset: bool
    trajectory_ok: bool
    accepted: bool
    grounded_outcome: bool
    actual_queue: str


@dataclass(frozen=True)
class Summary:
    """Aggregate scores for one dataset run."""

    cases: list[CaseResult]
    queue_accuracy: float
    citation_subset_rate: float
    trajectory_rate: float
    citation_score: float
    ungrounded_accepted: int


def load_dataset() -> list[dict[str, Any]]:
    """Load the checked-in evaluation rows.

    Returns:
        The dataset rows in file order.
    """
    return _load_object_list(DATASET_PATH)


def load_human_labels() -> list[dict[str, Any]]:
    """Load the handwritten groundedness labels.

    Returns:
        The eight human labels.
    """
    return _load_object_list(HUMAN_LABELS_PATH)


def load_judge_fixture() -> list[dict[str, Any]]:
    """Load the checked-in judge results.

    Returns:
        Fixture judge rows, each with an id and a groundedness label.
    """
    loaded = json.loads(JUDGE_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = "judge fixture must be an object"
        raise TypeError(message)
    results = loaded.get("results")
    if not isinstance(results, list):
        message = "judge fixture results must be a list"
        raise TypeError(message)
    return [cast("dict[str, Any]", item) for item in results]


def fixture_agreement() -> float:
    """Compare human groundedness labels with the fixture judge.

    Returns:
        The agreement fraction.
    """
    human = load_human_labels()
    judged = {str(item["id"]): bool(item["grounded"]) for item in load_judge_fixture()}
    return groundedness_agreement(
        [bool(item["grounded"]) for item in human],
        [judged[str(item["id"])] for item in human],
    )


def run_dataset() -> Summary:
    """Run every dataset row through the graph with its scripted model.

    Returns:
        Per-row scores and the aggregate citation score.
    """
    try:
        cases = [_run_row(row) for row in load_dataset()]
    finally:
        clear_recorded_routes()
    return _summarize(cases)


def _run_row(row: dict[str, Any]) -> CaseResult:
    clear_recorded_routes()
    model = ScriptedChatModel(responses=_script_messages(row["script"]))

    async def _invoke() -> dict[str, Any]:
        with use_chat_model(model):
            result = await graph.ainvoke(
                InputState(messages=[HumanMessage(content=str(row["ticket"]))]),
                context=Context(),
            )
        if not isinstance(result, dict):
            message = "graph result must be a dict"
            raise TypeError(message)
        return cast("dict[str, Any]", result)

    state = asyncio.run(_invoke())
    routes = recorded_routes()
    actual_queue = routes[-1]["queue"] if routes else ""
    citation_ids = list(routes[-1]["citation_ids"]) if routes else []
    passages = state.get("passages", [])
    retrieved_ids = (
        [str(passage["id"]) for passage in passages] if isinstance(passages, list) else []
    )
    messages = state["messages"]
    reply = model_reply_text(messages)
    citations_ok = citation_subset_ok(citation_ids, retrieved_ids, reply)
    accepted = bool(state.get("accepted"))
    return CaseResult(
        id=str(row["id"]),
        queue_ok=queue_correct(actual_queue, str(row["expected_queue"])),
        citation_subset=citations_ok,
        trajectory_ok=trajectory_ok(
            messages,
            lookup_required=bool(row["lookup_order_required"]),
            passages_present=bool(passages),
        ),
        accepted=accepted,
        grounded_outcome=grounded_outcome(accepted, citations_ok),
        actual_queue=actual_queue,
    )


def _summarize(cases: list[CaseResult]) -> Summary:
    count = len(cases)
    if count == 0:
        message = "dataset has no rows"
        raise ValueError(message)
    return Summary(
        cases=cases,
        queue_accuracy=sum(case.queue_ok for case in cases) / count,
        citation_subset_rate=sum(case.citation_subset for case in cases) / count,
        trajectory_rate=sum(case.trajectory_ok for case in cases) / count,
        citation_score=sum(case.grounded_outcome for case in cases) / count,
        ungrounded_accepted=sum(case.accepted and not case.citation_subset for case in cases),
    )


def _script_messages(script: list[dict[str, Any]]) -> list[AIMessage]:
    messages: list[AIMessage] = []
    for index, step in enumerate(script):
        if step["type"] == "tool":
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": str(step["name"]),
                            "args": cast("dict[str, Any]", step["args"]),
                            "id": f"{step['name']}-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            continue
        messages.append(AIMessage(content=str(step["content"])))
    return messages


def _load_object_list(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        message = f"{path.name} must be a list"
        raise TypeError(message)
    return [cast("dict[str, Any]", item) for item in loaded]
