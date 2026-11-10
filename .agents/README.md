# Repository Agent Context

This repository maintains self-contained context and learning records for the `support-triage` project. Read this guide and [LEARNING_LOG.md](LEARNING_LOG.md) at session start.

## Context Layers

| Layer | Repository application |
| --- | --- |
| Repository entry point | `AGENTS.md` owns repository instructions and references this directory. |
| Repository specifics | Keep established rules in `AGENTS.md` or `.agents/REPO_RULES.md`. Read the latter when present. |
| Learning record | `.agents/LEARNING_LOG.md` is tracked and enabled by its presence. Keep verified findings short. |

Apply the host's instruction priority and explicit user requests, then the repository guidance in `.agents/REPO_RULES.md`.

## Recording and Promoting Discoveries

Before appending, look for an existing rule, solution, or entry. Record only a verified finding or correction: date, observation, and evidence or a canonical link. Correct stale claims; keep secrets, private account details, and session transcripts out of this tracked log.

Follow the repository's [documentation index](../docs/README.md) for durable destinations. Prefer code, a test, or a check that prevents recurrence.
