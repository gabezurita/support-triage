# Support Triage

A customer-support triage agent. It retrieves a checked-in help center, then routes the ticket.

This repository started from the [LangGraph react-agent template](https://github.com/langchain-ai/react-agent) (MIT, Copyright 2024 LangChain). The graph is that tool loop, renamed to `support_triage`, with the template web-search tool removed. [ADR-001](docs/adr/ADR-001-langgraph-react-agent.md) records why the loop is the starting point, and how a Deep Agent can replace that graph or wrap it later.

Acceptance criteria are on [issue 1](https://github.com/gabezurita/support-triage/issues/1). The quickstart, the diagram, and the trace instructions land with the submission docs.
