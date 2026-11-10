# Support Triage

A customer-support triage agent. It retrieves a checked-in help center, then routes the ticket.

This repository started from the [LangGraph react-agent template](https://github.com/langchain-ai/react-agent) (MIT, Copyright 2024 LangChain). The graph is that tool loop, renamed to `support_triage`, with the template web-search tool removed. [ADR-001](docs/adr/ADR-001-langgraph-react-agent.md) records why the loop is the starting point, and how a Deep Agent can replace that graph or wrap it later.

Acceptance criteria are on [issue 1](https://github.com/gabezurita/support-triage/issues/1).

## Quickstart

Python 3.12. From the repository root:

```bash
uv sync
cp .env.example .env
uv run langgraph dev
```

`.env.example` names `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. A live model call needs `ANTHROPIC_API_KEY`. The unit tests, the trajectory checks, and the dataset evaluation do not.

Run the offline suite with:

```bash
uv run pytest
```

## Graph

Retrieval runs before the model. The model may call `lookup_order` and `route_ticket`. The citation gate rejects a reply that cites an article the retrieval step did not return. On a step limit, that same decision runs inside the model node so the run can still stop.

```mermaid
flowchart TD
    startNode([start]) --> retrieve
    retrieve --> call_model
    call_model -->|tool call| tools
    tools --> call_model
    call_model -->|reply| finish
    call_model -->|step limit| endNode([end])
    finish -->|accepted| citation_gate
    finish -->|no accepted route| endNode
    citation_gate --> endNode
```

## Traces

When `LANGSMITH_API_KEY` is set, `uv run langgraph dev` sends traces to the LangSmith project named by `LANGSMITH_PROJECT`. The example project name is `support-triage`. Open that project in LangSmith to see the run.

When the key is not set, the offline record is [EVAL-001](docs/evaluations/EVAL-001-citation-gate.md). The dataset, the human labels, and the fixture judge are in `data/evaluation/`.
