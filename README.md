# Support Triage

A customer-support triage agent. It retrieves a checked-in help center, then routes the ticket.

This repository started from the [LangGraph react-agent template](https://github.com/langchain-ai/react-agent) (MIT, Copyright 2024 LangChain). The graph is that tool loop, renamed to `support_triage`, with the template web-search tool removed. [ADR-001](docs/adr/ADR-001-langgraph-react-agent.md) records why the loop is the starting point, and how a Deep Agent can replace that graph or wrap it later.

The delivered MVP is [issue 1](https://github.com/gabezurita/support-triage/issues/1). What comes next, including later work, is [issue 19](https://github.com/gabezurita/support-triage/issues/19).

## Quickstart

Python 3.12. From the repository root:

```bash
uv sync
cp .env.example .env
uv run langgraph dev
```

Set `SUPPORT_TRIAGE_PROVIDER` to `gemini`, `anthropic`, or `codex`. The default is `gemini`. Gemini reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Anthropic reads `ANTHROPIC_API_KEY`. Codex reads `OPENAI_API_KEY` and calls `gpt-5.3-codex`. `.env.example` also names `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`. The unit tests, the trajectory checks, and the dataset evaluation do not call a provider.

Run the offline suite with:

```bash
uv run pytest
```

The Studio walkthrough is [docs/demo.md](docs/demo.md). It covers the tunnel Chrome needs, the four tickets, and where the citation-gate rejection is recorded.

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

Each name in the diagram is one box or one arrow.

Boxes:

- [`start`](src/support_triage/graph.py#L312) hands the ticket to `retrieve`.
- [`retrieve`](src/support_triage/graph.py#L120) searches the help center and stores the matches on `passages`.
- [`call_model`](src/support_triage/graph.py#L137) reads those passages. It may call `lookup_order` and `route_ticket`.
- [`tools`](src/support_triage/graph.py#L309) runs [`lookup_order`](src/support_triage/tools.py#L54) and [`route_ticket`](src/support_triage/tools.py#L70), then returns to `call_model`.
- [`finish`](src/support_triage/graph.py#L200) records whether `route_ticket` was accepted.
- [`citation_gate`](src/support_triage/graph.py#L251) accepts the reply when the ids passed to `route_ticket` are among the retrieved articles and the reply cites that same set.
- [`end`](src/support_triage/graph.py#L351-L354) stops the run.

Arrows:

- [`tool call`](src/support_triage/graph.py#L334) sends `lookup_order` or `route_ticket` to `tools`, which returns to `call_model`.
- [`reply`](src/support_triage/graph.py#L333) means the model stopped calling tools, so the run goes to `finish`.
- [`accepted`](src/support_triage/graph.py#L346) means `route_ticket` recorded a queue. The run goes through `citation_gate`, then to `end`.
- [`no accepted route`](src/support_triage/graph.py#L217-L225) means the model skipped `route_ticket`, or the queue was rejected. A skipped call records `human` with no citation ids, sets `accepted` false, and fails the trajectory check. A rejected queue records nothing. The run goes to `end` and skips the gate.
- [`step limit`](src/support_triage/graph.py#L166) means too few steps remain for another node. `call_model` applies the `finish` and `citation_gate` decision itself, and the arrow goes to [`end`](src/support_triage/graph.py#L330).

## Traces

When `LANGSMITH_API_KEY` is set, `uv run langgraph dev` sends traces to the LangSmith project named by `LANGSMITH_PROJECT`. The example project name is `support-triage`. Open that project in LangSmith to see the run.

The checked-in redacted trace is [data/evaluation/traces/billing-double-charge.json](data/evaluation/traces/billing-double-charge.json). The live judge result is [data/evaluation/live_judge.json](data/evaluation/live_judge.json). The offline record is [EVAL-001](docs/evaluations/EVAL-001-citation-gate.md). The dataset, the human labels, and the fixture judge are in `data/evaluation/`.
