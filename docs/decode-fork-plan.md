# Forking decode into APICookbook: decisions and plan

Status: F0 and F1 done (2026-11-09), F2 next · Sandbox changed from Modal to a dedicated OCI instance on 2026-11-09 · Decided with the owner in an interview on 2026-11-09 · Upstream: [decodingai-magazine/building-a-coding-agent-from-scratch-course](https://github.com/decodingai-magazine/building-a-coding-agent-from-scratch-course) ("decode", Apache-2.0)

## Intent

Add a **"Build & run"** mode to the APICookbook website, powered by a trimmed fork of decode. A visitor describes a BaNCS task. The agent writes code, runs it in an isolated sandbox, calls the BaNCS demo host, reads the results and fixes its own errors, until the code works.

This **replaces** today's code agent (`backend/code_agent.py`), which writes code but never runs it. Decided 2026-11-09. Until Build & run passes F4 (guardrails) and F5 (evals), the current agent stays live, so the site never goes without an agent. Then the chat switches over and `code_agent.py`, its routes and its tests are removed. The BERT search chat stays as the fallback when the agent is capped or down.

## Decisions

| Topic | Decision |
|---|---|
| Where it runs | Inside the APICookbook website |
| What it may do | Write files and run code; send **any** BaNCS call, including POST/PUT/DELETE |
| Who may use it | **Anyone, with caps**: no sign-in |
| Sandbox | **A dedicated OCI instance** (`instance-20261025-1657`, A1.Flex, 1 OCPU, 6 GB, same subnet as the main VM): one gVisor container per visitor session. Replaced Modal on 2026-11-09: no new vendor or bill. |
| Agent loop | **Whole agent on the sandbox host**: a small runner there starts `decode run` per session; cookbook-api relays events |
| Approvals | Files and commands run without asking inside the sandbox; **every bank write waits for the visitor's Approve** |
| Kitaru | **Removed**, so the fork uses pydantic-ai 2.66 like the rest of the repo |
| Tracing | **OpenTelemetry, no vendor**: logfire spans kept, pointed at a collector chosen later; off until configured |
| Optional tools kept | **Subagents** (Explore helpers) and **LSP** diagnostics. Web fetch and skills are removed. |
| Memory | **Fixed instructions only**: one `AGENTS.md` with the BaNCS conventions. No learning between sessions. |
| Harness kept | Compaction, steering (mid-run corrections), the agents catalog (Build, Plan, Explore, Code-Reviewer) |
| Repo extras | Keep the eval harness, regression framework, tests and ADRs. Replace the generic benchmark with BaNCS tasks. Delete the course media and site. |
| Fork home | **Inside the APICookbook repo**, under `agent_runtime/`, keeping decode's Apache-2.0 licence and notices |
| Today's agent | **Replaced** by Build & run once F4 and F5 pass; it stays live until then |
| Model | Claude by default. F0 added an `anthropic` provider and made it the default; Gemini, OpenRouter and Modal stay selectable but unused, so upstream's tests keep running. Removing them is deferred. |

## What stays and what goes

Sizes are decode's source as of 2026-11-09 (about 630 KB in `src/decode`).

| decode part | Size | Fate |
|---|---|---|
| `agent/` (loop, factory, deps, context window) | 52 KB | **Keep.** Add Anthropic to the factory and drop the other providers. Keep decode's retry budgets. |
| `tools/` files, bash, exec, todo, ask-user, truncate, registry, approval | ~85 KB | **Keep** |
| `tools/agent.py` + `orchestration.py` (subagents) | 32 KB | **Keep** |
| `tools/lsp.py` + `services/lsp/` | 29 KB | **Keep.** Language servers go into the session image. |
| `tools/web.py` | 10 KB | **Remove**, with the `markdownify` dependency |
| `skills/` + `tools/skills.py` | 18 KB | **Remove** |
| `permissions/` | 13 KB | **Keep, slimmed.** One policy: allow inside the sandbox, ask for bank writes. The four modes go. |
| `agents/` (catalog) | 16 KB | **Keep** |
| `memory/` | 15 KB | **Keep loading only.** Remove `extract.py` (learning on exit). |
| `context/` (compaction, session log) | 27 KB | **Keep** |
| `harness/` (runner, queue, decisions) | 11 KB | **Keep.** The steering queue gets a web input. |
| `sandbox/` executor, workspace, handback | 58 KB | **Keep.** The Modal backend in it goes in F1. |
| `sandbox/docker_backend.py` | 19 KB | Removed in F0; **restored in F1** with gVisor, when the sandbox moved from Modal to the OCI instance |
| `remote/` app, headless, image | 54 KB | Kept in F0; **removed in F1** (Modal-only). The runner on the sandbox host takes its place. |
| `remote/cli.py` | 12 KB | **Remove**; the website starts runs, not a CLI |
| `runtime/` headless, summary, task inputs | 32 KB | **Keep** |
| `runtime/recording.py` (Kitaru) | 14 KB | **Remove**, with `kitaru` and `kitaru-pydantic-ai` |
| `observability/` | 19 KB | **Adapt.** Keep the OpenTelemetry spans and drop the Opik-specific parts. Cost goes into APICookbook's usage ledger. |
| `tui/` | 53 KB | **Remove**, with `prompt-toolkit`. The event renderer `tui/render.py` is kept as `decode/render.py`, since the capstone tests and logs use it, so `rich` stays. |
| `cli.py` | 26 KB | **Trim**, not remove (found in F0): the runner calls `decode run` as a subprocess. It keeps `run` and `sessions`; the REPL path goes. |
| `config/` | 22 KB | **Keep.** Anthropic settings replace the other providers'. |
| `evals/` harness and regression framework | — | **Keep.** Replace the generic tasks with BaNCS gold call sequences; report to a file and GitHub Actions instead of Opik. |
| `tests/` | 2.6 MB | **Keep** those for kept modules; delete the rest |
| `docs/adr/` | — | **Keep** as history; add an ADR for this fork |
| `assets/`, `tasks/`, `index.html`, `running_the_code/`, `.decode/skills/` demos | ~67 MB | **Remove** |

Result: about 470 KB of the 630 KB of source stays. Dependencies removed: `kitaru`, `kitaru-pydantic-ai`, `prompt-toolkit`, `markdownify`, and the pydantic-ai `google`/`openai` extras. `rich` and `click` go if nothing kept needs them.

## Architecture

```
Browser ── POST /api/build/stream ──▶ cookbook-api (main VM, 10.0.0.49) ── private HTTP ──▶ runner (sandbox host, 10.0.0.223)
   ▲  live events (steps, files, output, approval requests)                 │  decode run per session (loop + Claude)
   └─────────────────────────────────────────────────────────────────────────┘  tools in that session's gVisor container
Browser ── POST /api/build/{run}/approve | /steer ──▶ cookbook-api ──▶ runner ──▶ the run

Sandbox containers' network: ONLY 10.0.0.49:<proxy port> (no internet, no other address)
cookbook-api proxy: GET passes through; POST/PUT/DELETE held until the visitor approves in the browser
```
Two design points the answers imply:
1. **Bank writes are approved at the proxy, not by trusting the agent.** The bash tool could run `curl -X POST` directly, so the sandbox's network is locked to APICookbook's own BaNCS proxy (the Playground's allow-listed proxy, extended). The proxy holds every write as a pending request until the visitor approves it. decode's approval flow becomes the message the visitor sees, but the proxy is what enforces it.
2. **The agent runs away from the website, so it needs a two-way channel** for approvals and steering. The runner on the sandbox host exposes a small private HTTP API (start, events, approve, steer, stop) that only cookbook-api can reach. This is the main piece decode doesn't already have.

## Milestones

**F0. Vendor and slim.**
- Import decode into `agent_runtime/` at a pinned commit, keeping `LICENSE` and a `NOTICE` listing our changes.
- Delete the parts marked Remove.
- Add Anthropic to the factory; move to pydantic-ai 2.66 and fix whatever its API changes break.
- The kept tests pass locally against a local sandbox.

*F0 result (2026-11-09):* every kept `decode` unit test matches upstream's result. On Windows the remaining failures also fail on upstream: bash-driven eval tests, the Modal backend's `signal.SIGKILL`, and path separators. The Linux run is in F1. pydantic-ai 2.66 changed one thing we depended on: a tool retry now arrives as `ToolReturnPart(outcome="retried")`, not `RetryPromptPart`. Compaction and the test models were updated to match. What stays inert until F5 is listed in `agent_runtime/NOTICE`.

**F1. Sandbox host.** On `instance-20261025-1657` (upgraded in place to Ubuntu 22.04 by *Sandbox host upgrade*; inspected by *Sandbox host check*).
- **Restore decode's Docker backend** from upstream, with containers run under **gVisor** (`runsc`): no privileges, read-only root except the workspace, CPU, memory and PID limits, one throwaway container per session. Remove `remote/`; the Modal backend stays inert.
- **Session image** (`agent_runtime/sandbox/Dockerfile`): Python 3.12 with requests and httpx, Node, JDK 17, curl, git and jq, for aarch64. LSP diagnostics are off on the host for now.
- **Container network:** a Docker network whose only allowed destination is `10.0.0.49:<proxy port>`, enforced in the host's `DOCKER-USER` chain. No internet, no DNS beyond what the proxy needs, no other `10.0.0.x` address.
- **Runner:** a small service on `10.0.0.223` that (`decode.runner`) starts one `decode run` subprocess per task and returns its answer and summary. Its firewall accepts only cookbook-api's address, and every call needs a bearer token (`RUNNER_TOKEN`). It holds the Anthropic key; the containers never see it. 2 runs at once (1 OCPU); a start past that gets 429 and cookbook-api retries. Live per-step events move to F2, which needs the same loop for approvals.
- **Main VM side** (a separate step on the shared VM; you approve it): firewall rules dropping all traffic from `10.0.0.223` except the proxy port, in both `INPUT` and `DOCKER-USER`. Today the sandbox host can reach the main VM's SSH and HTTPS ports; after this it can't.
- **Setup by GitHub Actions:** *Sandbox host setup* (manual, idempotent) runs `agent_runtime/sandbox/setup-host.sh`, writes the runner's secrets, runs `verify-host.sh` inside a real worker, and does one smoke run. Secrets: `SANDBOX_HOST`, `SANDBOX_USER`, `SANDBOX_SSH_KEY`, `ANTHROPIC_API_KEY`, `RUNNER_TOKEN`.
- **Linux CI** for `agent_runtime/` (*agent_runtime tests*). `agent_runtime/` changes no longer redeploy the cookbook.
- **Done when:** one run started from a script completes end to end, the network rules are proved from inside a container (the proxy reachable; the internet, `10.0.0.49:22` and `:443` refused), and the fork's test suite passes on Linux.

*F1 result (2026-11-09):* the sandbox host is set up by *Sandbox host setup*.
- Inside a gVisor worker, these are blocked: the main VM's SSH and HTTPS, the host's own SSH and runner port, the internet, and the BaNCS host directly.
- A worker can write its workspace. It runs as the workspace owner's uid, which is 1001 on this image, not 1000.
- Smoke runs through the runner completed: answer 5050, about $0.006–0.03 per run, no leftover containers.
- All 2193 unit tests pass in Linux CI.

Still open, both outside the code:
1. OCI's subnet security list silently drops the proxy port. It needs ingress rules `10.0.0.223/32 → TCP 7601` and `10.0.0.49/32 → TCP 7700`.
2. The *Main VM sandbox firewall* workflow hasn't run yet; it needs the owner's go-ahead because it changes the shared VM.

**F2. Proxy with approvals.**
- The BaNCS proxy holds writes pending approval.
- Routes: `POST /api/build/stream`, `/approve`, `/steer`.
- The runner channel for approvals and steering.

**F3. Web UI.**
- A "Build & run" mode in the chat: live steps, file tree and contents, command output, Approve and Reject for each bank write, a steering box and Stop.

**F4. Guardrails** (needed before public launch):
- per-IP and USD caps for this mode;
- limits per run (wall time, steps, subagents, sandbox CPU and memory);
- a notice that bank writes change shared demo data;
- a way to reset or monitor the demo data.

**F5. Evals.**
- BaNCS tasks with gold call sequences, plus "the code ran and the GET calls returned 2xx".
- Regression cases mined from session logs, including the two empty-tool-call prompts.
- It must also beat the current agent on the same tasks before replacing it.

**F6. Switch over.**
- The chat's default goes to Build & run.
- Remove `backend/code_agent.py`, `/api/agent/chat|code|stream` and their tests.
- Keep the guard and ledger code, which Build & run reuses.
- Milestones 3 and 4 of `docs/coding-agent-plan.md` (sessions, evals) are superseded by decode's session log and F5.

## Risks

- **Anyone can run code and change shared demo data.** The sandbox protects our servers, but not the demo bank's data from other visitors. Mitigations: approval per write, caps, per-run limits, monitoring. The same machinery would make sign-in easy to add later.
- **Cost.** Subagents and test-and-fix loops make a Build & run answer cost many times today's $0.02. Because it replaces the cheap agent, every chat question now pays that price: simple lookups ("which endpoint updates an address?") too. The $5/day cap buys far fewer answers. Sandbox compute is a fixed, free instance, so the cost is Claude's tokens. Measure in F1, then decide the caps, and whether Build & run should skip the sandbox when no code needs running.
- **Prompt injection.** API responses and file contents the agent reads could try to steer it. The proxy-enforced approvals and the network lock keep the damage inside the sandbox and demo data.
- **Diverging from upstream.** A vendored, slimmed fork won't pull course updates easily. Record the upstream commit and our deletions in `NOTICE` and an ADR.
- **The pydantic-ai jump (2.46 to 2.66)** may break decode code in the loop and deferred-tool handling. F0 surfaces it.
- **One small sandbox host.** 1 OCPU runs about 2 sessions at once; the rest queue. The Always Free A1 allowance (4 OCPUs, 24 GB across instances) may leave room to grow it for free.
- **The sandbox host shares a subnet with the main VM.** A host compromise (a gVisor and Docker escape) would put it next to VAM and the other apps; the main VM's firewall rules above limit what it could reach.
