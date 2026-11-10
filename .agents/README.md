# Personal Repository Agent Context

This repository uses the [personal context architecture](https://github.com/gabezurita/gab0-system/blob/main/.agents/README.md). Read this guide and [LEARNING_LOG.md](LEARNING_LOG.md) at session start; preserve the repository-specific instructions already in the entry point.

When `~/AGENTS.md` exists and the host has not loaded it, read it once for OS defaults. Consult `~/.agents/README.md` for the full local architecture; this repository guide also works without the personal machine setup.

## Context Layers

| Layer | Local application |
| --- | --- |
| OS defaults | `~/AGENTS.md`, when the host loads it. The local `~/.agents/README.md` explains the full architecture and host setup. |
| Repository entry point | `AGENTS.md` owns repository instructions and references this directory. |
| Repository specifics | Keep established rules in `AGENTS.md` or `.agents/REPO_RULES.md`, when present. Read the latter if it exists; avoid duplicate rules. |
| Learning record | `.agents/LEARNING_LOG.md` is tracked and enabled by its presence. Keep verified findings short and link their canonical home. |
| Collaborative overlays | Optional, gitignored `USER_RULES.md` and `TEAM_RULES.md` are reserved for repos actually worked on by a team. Create neither here; existing example templates remain dormant. |

Apply the host's instruction priority and explicit user requests, then the most specific compatible repository guidance. Learning records hold evidence and do not override policy. Host support for automatic discovery and `@` imports varies: explicitly read referenced files, and consult the canonical architecture for supported host paths. Protocol aliases, when needed, should be small pointers to `AGENTS.md` to avoid repeated full context.

## Recording and Promoting Discoveries

Before appending, look for an existing rule, solution, or entry. Record only a verified finding or correction that a future session would otherwise rediscover: date, observation, and evidence or a canonical link. Correct stale claims; keep secrets, private account details, and session transcripts out of this tracked log.

Follow the repository's [documentation index](../docs/README.md) for durable destinations. Prefer code, a test, or a check that prevents recurrence. Promote lasting conventions into repository rules, recurring fixes into the existing solutions system, and architecture into decision records. Link those records from the log rather than copying their contents. Repeated failures warrant a preventive mechanism or a tracked issue. Periodically consolidate entries so session-start context stays small.

## Ownership and Distribution

Edit repository rules and discoveries here; edit machine-wide policy in `gab0-system`. Personal preferences are covered by OS defaults and tracked repository context. This guide is a local summary with a link to the full architecture; the OS skill sync command does not synchronize it. Organization `agent-sync` configurations, when present, are a separate mechanism that may overwrite their managed files.
