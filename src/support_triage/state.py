"""Define the state structures for the agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps


@dataclass
class InputState:
    """Input state for the agent.

    This is the narrow interface for a new conversation turn.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """Messages tracking the primary execution state of the agent.

    A turn is a human message, an optional tool call and tool result, then an
    assistant message. ``add_messages`` merges new messages with existing ones
    by id.
    """


@dataclass
class State(InputState):
    """Complete agent state, including retrieved passages and the route outcome."""

    is_last_step: IsLastStep = field(default=False)
    """True when the step count reaches the recursion limit minus one."""

    remaining_steps: RemainingSteps = field(default=0)
    """Steps left before LangGraph stops the run, including the current step."""

    passages: list[dict[str, str]] = field(default_factory=list)
    """Help-center passages retrieved for this ticket, each with ``id`` and ``text``."""

    accepted: bool = False
    """True when the model called ``route_ticket`` and the queue was accepted."""

    trajectory_ok: bool = False
    """True when the model itself called ``route_ticket``."""

    fallback_reason: str = ""
    """Set when the graph routes to human because the model never called ``route_ticket``."""

    citation_rejection: str = ""
    """Set when the citation gate rejects the reply."""
