"""Record one live judge run and one redacted billing trace.

Run this module directly. The offline test suite does not call it.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from .context import Context
from .evaluate import load_human_labels
from .graph import graph
from .judge import groundedness_agreement, judge_reply
from .providers import load_chat_model, provider_key, redact, resolve_provider
from .state import InputState
from .tools import clear_recorded_routes, recorded_routes

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = REPO_ROOT / "data" / "evaluation" / "live_judge.json"
TRACE_PATH = REPO_ROOT / "data" / "evaluation" / "traces" / "billing-double-charge.json"
BILLING_TICKET = "Why was I charged twice for order 1001?"
_ARTICLES = ("billing-001", "shipping-001", "refunds-001")


def help_center_text() -> str:
    """Return the three help-center articles, labeled by id.

    Returns:
        The article text the live judge is allowed to use.
    """
    blocks: list[str] = []
    for article_id in _ARTICLES:
        ref = importlib.resources.files("support_triage.help_center").joinpath(f"{article_id}.md")
        blocks.append(f"[{article_id}]\n{ref.read_text(encoding='utf-8')}")
    return "\n\n".join(blocks)


def score_labels() -> dict[str, Any]:
    """Score the eight human labels with the configured model.

    Returns:
        The provider, the model, the agreement, and one result per label.
    """
    spec = resolve_provider()
    model = load_chat_model(spec.model_name)
    articles = help_center_text()
    labels = load_human_labels()
    results: list[dict[str, Any]] = []
    for label in labels:
        citation_ids = label["citation_ids"]
        if not isinstance(citation_ids, list):
            message = f"{label['id']} is missing citation ids"
            raise ValueError(message)
        score = judge_reply(
            model,
            ticket=str(label["ticket"]),
            reply=str(label["reply"]),
            citation_ids=[str(item) for item in citation_ids],
            articles=articles,
        )
        results.append({"id": str(label["id"]), **score})
    human = [bool(label["grounded"]) for label in labels]
    judged = [bool(result["grounded"]) for result in results]
    return {
        "provider": spec.name,
        "model": spec.model_name,
        "agreement": groundedness_agreement(human, judged),
        "results": results,
    }


async def billing_trace() -> dict[str, Any]:
    """Run the billing ticket through the live graph.

    Returns:
        Passages, messages, and routes for the redacted trace file.
    """
    spec = resolve_provider()
    clear_recorded_routes()
    result = await graph.ainvoke(
        InputState(messages=[HumanMessage(content=BILLING_TICKET)]),
        {"recursion_limit": 25},
        context=Context(),
    )
    if not isinstance(result, dict):
        message = "The graph did not return a state dictionary."
        raise TypeError(message)
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        message = "The graph state did not include messages."
        raise TypeError(message)
    return {
        "provider": spec.name,
        "model": spec.model_name,
        "ticket": BILLING_TICKET,
        "passages": result.get("passages", []),
        "accepted": bool(result.get("accepted")),
        "trajectory_ok": bool(result.get("trajectory_ok")),
        "fallback_reason": str(result.get("fallback_reason", "")),
        "citation_rejection": str(result.get("citation_rejection", "")),
        "routes": recorded_routes(),
        "messages": [_public_message(message) for message in messages if _is_message(message)],
    }


def write_redacted(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload after removing configured keys.

    Args:
        path: The destination file.
        payload: The judge result or the trace.

    Raises:
        RuntimeError: A configured key is still present after redaction.
    """
    secrets = [
        value
        for name in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "LANGSMITH_API_KEY",
        )
        if (value := os.environ.get(name))
    ]
    cleaned = redact(payload, secrets)
    text = json.dumps(cleaned, indent=2) + "\n"
    if any(secret in text for secret in secrets):
        message = f"Refusing to write {path.name} because a key survived redaction."
        raise RuntimeError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Score the labels and record the billing trace."""
    if provider_key() is None:
        message = "The selected provider has no key."
        raise SystemExit(message)
    write_redacted(JUDGE_PATH, score_labels())
    write_redacted(TRACE_PATH, asyncio.run(billing_trace()))
    print(f"wrote {JUDGE_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {TRACE_PATH.relative_to(REPO_ROOT)}")


def _is_message(value: object) -> bool:
    return isinstance(value, AIMessage | HumanMessage | ToolMessage)


def _public_text(content: object) -> str:
    """Return visible text, without model signature blocks.

    Args:
        content: A message body, either a string or a list of content blocks.

    Returns:
        The text a person would read.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _public_message(message: AnyMessage) -> dict[str, Any]:
    view: dict[str, Any] = {"type": message.type, "content": _public_text(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        view["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args")} for call in message.tool_calls
        ]
    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        view["name"] = name
    return view


if __name__ == "__main__":
    main()
