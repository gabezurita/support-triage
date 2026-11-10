# Documentation

Decisions live in `adr/`. Evaluation notes will live in `evaluations/`. Acceptance criteria, and the implementation plan, are on [issue 1](https://github.com/gabezurita/support-triage/issues/1).

## Directory taxonomy

| Directory | Purpose |
| --- | --- |
| [`adr/`](adr/) | Architecture decisions. |
| `plans/` | Later implementation plans, when one is written in the repo. |
| `evaluations/` | Evaluation writeups, including the citation-gate before-and-after. |

## Architecture decisions

- [ADR-001: Start from the ReAct template](adr/ADR-001-langgraph-react-agent.md) — The graph starts as the react-agent tool loop. Deep Agents can replace that graph or wrap it as a subagent after the citation-gate before-and-after.

## Evaluations

The citation-gate note will be `evaluations/EVAL-001-citation-gate.md`. That directory is created with the note.
