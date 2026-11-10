"""Define the state structures for the agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


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
    """Complete agent state, including managed graph fields."""

    is_last_step: IsLastStep = field(default=False)
    """True when the step count reaches the recursion limit minus one."""
