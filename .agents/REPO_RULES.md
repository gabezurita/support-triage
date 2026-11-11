# Repository Rules: support-triage

GitHub issue 1 is the delivered MVP. Issue 19 is what comes next, including later work. Implementation plans are the comments on its sub-issues.

## Product constraints

- Python 3.12 and uv. Ruff and mypy use the settings in `pyproject.toml`. There is no coverage fail-under.
- Checks: `uv run ruff check .`, `uv run mypy src`, and `uv run pytest`.
- Commits are `type: summary`, with no scope and no ticket id.
- The only side effect is `route_ticket`. Do not send mail, issue a refund, or call agent-orchestrator.
- Do not start the graph as a Deep Agent. ADR-001 records the two later moves.
- Docs keep `adr/`, `plans/`, and `evaluations/` only. `.agents/standard-layout-off` stops the other taxonomy directories from being recreated.
- The template web-search tool and `TAVILY_API_KEY` stay out.

## Tooling

Ruff formats and lints. Mypy typechecks `src`. This repo has no JavaScript toolchain and no pre-commit config. Editor settings do not replace the ruff and mypy settings in `pyproject.toml`.
