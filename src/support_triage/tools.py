"""Tools bound into the triage graph.

The scaffold list is empty. The template web-search tool is not part of this agent.
"""

from collections.abc import Callable
from typing import Any

TOOLS: list[Callable[..., Any]] = []
