# 0024. APICookbook fork: decode as the website's "Build & run" agent

**Status:** Accepted (F0 of docs/decode-fork-plan.md in the APICookbook repo)
**Date:** 2026-11-09

## Context

APICookbook (a developer portal for the TCS BaNCS banking APIs) adopts decode as the agent behind a
"Build & run" chat mode: a public visitor describes a task, and the agent writes code, runs it in a
per-visitor Modal sandbox, calls the BaNCS demo host and fixes its own errors. The owner's decisions
(interview of 2026-11-09) are recorded in APICookbook's `docs/decode-fork-plan.md`. They make large
parts of decode unnecessary: the website replaces the terminal UI, Modal replaces Docker, Kitaru's
recording/replay is dropped, and web fetch and skills are not offered to anonymous visitors.

## Decision

1. **Vendor, don't depend.** decode is copied into `agent_runtime/` at upstream commit `2ba55b5`
   and trimmed. `NOTICE` lists every change; `LICENSE` is kept.
2. **Removed:** `tui/` and the REPL path, Kitaru (recording seam, worker task inputs, eval cohorts),
   the Docker backend, `web_fetch`, skills, memory extraction on exit, `remote/cli.py`.
3. **Kept but trimmed:** `cli.py` keeps `decode run` and `decode sessions`. The Modal headless app
   (`remote/app.py`, `remote/headless.py`) runs `decode run` as a subprocess, so the command is
   the fork's entry point. The bare `decode` command now prints help.
4. **Claude is the default provider.** A fourth `llm_provider`, `anthropic`, builds an
   `AnthropicModel`. It uses effort `medium`, cached instructions and tool definitions, and server-side
   refusal fallback (the settings APICookbook's own agent measured). The upstream providers stay
   selectable so upstream's tests keep running. Removing them is deferred; it costs test rewrites,
   not risk.
5. **pydantic-ai 2.66.0**, matching APICookbook. Kitaru's adapter capped it below 2.47; removing
   Kitaru lifts the cap.
6. **Moved, not deleted:** the event renderer (`decode/render.py`), which the integration capstones
   assert through, and `InputIntent` (`harness/queue.py`), which the steering queue still needs.

## Consequences

- No replay or crash-resume of runs. Session logs and APICookbook's usage ledger remain.
- `SANDBOX_MODE` is `none` or `modal`. Local development without Modal runs tools on the host,
  which is acceptable for a developer, never for the website.
- Pulling upstream changes means re-applying this ADR's deletions. Record the upstream commit on
  every sync.
- Still to do in later milestones: the Modal deployment (F1), the approval proxy for BaNCS writes (F2),
  the web UI (F3), guardrails (F4) and BaNCS evals that report without Opik (F5).

## Amendment, 2026-11-09 (F1): the sandbox moves from Modal to a dedicated OCI instance

The owner chose a free, dedicated OCI instance (`instance-20261025-1657`, A1.Flex, 1 OCPU, 6 GB, in
the main VM's subnet) over a Modal account. So:

1. **The Docker backend is restored** from upstream and can run the worker hardened: gVisor
   (`runsc`), no capabilities, read-only root, uid 1000, CPU/memory/PID limits, and a docker network
   whose only destination is the BaNCS proxy on the main VM. All three settings default off, so
   upstream's Docker tests still describe the default worker.
2. **`remote/` (the Modal headless app) is removed.** Its role goes to `decode.runner`, a stdlib HTTP
   service on the host's private address that starts one `decode run` subprocess per task (decode
   keeps per-process state, so runs can't share a process). It caps concurrent runs (429 past the
   cap), enforces a wall-clock limit, and removes a killed run's worker by its label.
3. The Modal sandbox backend stays, inert, like the unused providers.
4. Per-step live events are not in F1: `decode run` returns the answer and a summary. They arrive
   in F2 with approvals, which need the same loop.

