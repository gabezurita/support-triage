# ADR-001: Start from the ReAct template

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** Gabriel Zurita

## Context

The interview walks a graph the author can point at. The react-agent template's inner cycle is a model call, then tools, then the model again. [Deep Agents](https://github.com/langchain-ai/deepagents/blob/main/libs/ARCHITECTURE.md) adds middleware on that same loop. Retrieval, routing, and the citation-gate change do not need that middleware.

## Decision

The exported `agent` graph starts as that ReAct loop, in the `support_triage` package. A retrieve step is added in front of the loop. The template web-search tool is not part of the graph.

Deep Agents waits until the citation-gate before-and-after exists. After that, either replace the exported graph with `create_deep_agent`, or pass this graph to a Deep Agent as a subagent. Re-run the dataset after either move.

## Alternatives considered

### Prebuilt `create_agent` / `create_react_agent`

The prebuilt helper wraps the loop in a single compiled structure. However, this agent requires an explicit retrieval step before model execution and a citation-verification gate on outputs. Starting from the explicit state graph makes adding these custom boundary nodes direct and inspectable in graph visualizations.

### Start on Deep Agents

The middleware would be present before the retrieval and routing behavior exists. The interview would walk a larger graph than this exercise needs.

## Consequences

- The first graph is the template's model-and-tools cycle, without web search.
- A later Deep Agents move is a graph swap. The corpus, the tools, and the dataset stay.
- The dataset is re-run after that swap.
