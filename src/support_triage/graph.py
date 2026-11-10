"""Reasoning and action loop for support triage.

The scaffold is the react-agent tool cycle with no tools bound.
Retrieval and the routing tools land in later units.
"""

from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .context import Context
from .state import InputState, State
from .tools import TOOLS
from .utils import load_chat_model


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, list[AIMessage]]:
    """Call the model with the current messages and bound tools.

    Args:
        state: The current conversation state.
        runtime: Runtime context, including the model name and system prompt.

    Returns:
        A state update containing the model's response message.
    """
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    response = cast(
        "AIMessage",
        await model.ainvoke([{"role": "system", "content": system_message}, *state.messages]),
    )
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content=(
                        "Sorry, I could not find an answer to your question "
                        "in the specified number of steps."
                    ),
                )
            ]
        }
    return {"messages": [response]}


builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Choose the next node from the model's last message.

    Args:
        state: The current conversation state.

    Returns:
        ``__end__`` when the model made no tool call, otherwise ``tools``.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        message = f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        raise ValueError(message)
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
graph = builder.compile(name="support-triage")
