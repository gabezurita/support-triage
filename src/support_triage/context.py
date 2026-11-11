"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from . import prompts
from .providers import resolve_provider


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": (
                "The system prompt to use for the agent's interactions. "
                "This prompt sets the context and behavior for the agent."
            )
        },
    )

    model: str = field(
        default="google_genai/gemini-2.5-flash",
        metadata={
            "description": (
                "The name of the language model to use for the agent's main interactions. "
                "Should be in the form: provider/model-name."
            )
        },
    )

    def __post_init__(self) -> None:
        """Fill unset fields from namespaced environment variables."""
        for context_field in fields(self):
            if not context_field.init:
                continue
            if context_field.name == "model" and os.environ.get("SUPPORT_TRIAGE_MODEL") is None:
                if getattr(self, context_field.name) == context_field.default:
                    setattr(self, context_field.name, resolve_provider().model_name)
                continue
            if getattr(self, context_field.name) == context_field.default:
                env_key = f"SUPPORT_TRIAGE_{context_field.name.upper()}"
                env_value = os.environ.get(env_key, context_field.default)
                setattr(self, context_field.name, env_value)
