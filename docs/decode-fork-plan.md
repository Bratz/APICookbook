# Forking decode into APICookbook: decisions and plan

Status: F0–F6 done (2026-11-09): Build & run is the site's agent · Sandbox changed from Modal to a dedicated OCI instance on 2026-11-09 · Decided with the owner in an interview on 2026-11-09 · Upstream: [decodingai-magazine/building-a-coding-agent-from-scratch-course](https://github.com/decodingai-magazine/building-a-coding-agent-from-scratch-course) ("decode", Apache-2.0)

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

**F2a. Proxy with approvals, and API knowledge** (decided 2026-11-09):
- `backend/build_proxy.py`: a second server inside cookbook-api's process on `10.0.0.49:7601`, published on the private address only.
  - Every path carries a per-run token: `/r/<token>/...`.
  - GETs pass through to the BaNCS host, which is the only host allowed, with the bank headers added.
  - POST, PUT, PATCH and DELETE are **held** until approved. A rejection, a 2-minute timeout or the run ending all answer 403, and nothing reaches the bank.
  - `/_kb/*` serves the code agent's search, endpoint and body-schema lookups. `url_template` is rewritten to the run's proxy address.
- decode gets three read-only tools (`search_endpoints`, `get_endpoint`, `get_body_schema`) and a BaNCS part of its prompt, both active only when the run has `BANCS_PROXY_URL`. The runner passes that on, accepting only the cookbook proxy's prefix.
- Operator-only routes `/api/build/runs` (start, status with held writes and bank calls, approve or reject, stop) behind `RUNNER_TOKEN` until F3 and F4.
- *Build smoke* (manual) drives one task through all of it; by default it rejects every write, so no demo data changes.

*F2a result (2026-11-09):* deployed. Inside a worker the proxy is the only open destination. *Build smoke*'s default task ("look up a customer, then try updating a phone number", writes rejected):
- 15 GETs reached the demo bank through the proxy: 4 returned 200, and 11 returned 400 for "No Data found" on test customers.
- The agent's `PUT /customerManagement/customer/contact` for customer 100698 was **held, shown and rejected**. It never reached the bank.
- The agent wrote and ran `amend_phone.py`, explained the rejection without retrying, and finished in 12 requests for **$0.18**, about 9× today's agent. Measuring cost is part of F4 and F5.

**F2b. Live events** (built 2026-11-09):
- `decode run --events-jsonl` appends each tool call and result as a JSON line, through pydantic-ai's `event_stream_handler` (the loop is unchanged).
- The runner serves the log with a cursor: `GET /runs/<id>/events?after=N`.
- cookbook-api streams a run live as server-sent events: `GET /api/build/runs/{id}/events` sends `step`, `write` (held), `write_decided`, `bank_call` and finally `done`, with the answer and summary. It pings every 15 s so the proxies along the way keep the stream open.
- Approve, reject and stop stay at their F2a routes. The routes are still operator-only; visitors get them with F3's UI behind F4's caps.
- *Result (2026-11-09):* deployed; 2201 decode tests pass on Linux. *Build smoke* read the live stream:
  - 27 steps, interleaved with 15 bank calls as they happened (GETs, mostly "No Data found" on test customers, until customer 135215 returned 200);
  - then `PUT /customer/contact` was held, shown and rejected the moment it appeared;
  - the agent's answer said the write never reached the bank; 14 requests, $0.22.
- **Deferred: steering** (mid-run corrections). `decode run` has no input channel. It would need the REPL loop's steering queue fed from a file or socket. Stop covers "that's wrong" for now.

**F3. Web UI** (built 2026-11-09, operator preview):
- A **Build & run** toggle in the chat composer. It is hidden unless the operator key is saved in that browser: open the site with `#build` to enter it. The key lives in `localStorage` and is cleared on a 401.
- Live view, from the F2b stream:
  - each tool call as a step, ticked or crossed on its result, with collapsible output;
  - bank calls with method badges;
  - each held write as an amber card with the formatted body, **Approve / Reject** and a 2-minute countdown;
  - a **Stop** button;
  - the answer as sanitised markdown, then time and cost.
- The run's proxy URL and token are shown as `$BANK` everywhere.
- On phones the composer's buttons wrap below the text box.
- Tried against a local mock of the stream (approve and reject paths, desktop and 375 px).
- Opening it to visitors waits for F4: caps on this mode, a per-visitor run key in place of the operator key, and the shared-data notice.

**F4. Guardrails** (decided 2026-11-09: open to every visitor, with caps):
- **No key to type.** Starting a run returns a random per-run key that the page keeps in memory and sends as `X-Run-Key`. Only that tab can watch the run, approve its writes or stop it; everyone else gets 404. The operator's `RUNNER_TOKEN` still works on every run and skips the per-IP limit.
- **Caps for this mode,** with their own ledger (`/app/data/build`): 3 runs per hour per IP, $5 a day, $50 a month. Past a cap, a start answers 429 with a plain message.
- **A watcher per run** records its cost and ends its proxy token when it finishes, even if nobody is watching.
- **Per-run limits,** from F1 and F2a: 15 minutes of wall time, 40 model requests, 300 bank calls, 1.5 GB, one core, and 2 runs at once on the host.
- **The composer note** says approved changes are visible to everyone using the demo.
- **Still open:** a way to monitor or reset the demo data (*Build smoke* with `reject` is the safe check for now), and alerts when the ledger nears its caps.

**F5. Evals** (built 2026-11-09):
- `backend/agent_eval.py` runs **20 BaNCS tasks**, 11 read-only and 9 that need a write, through **both** agents.
  - The tasks come from the code agent's exit check and the retrieval benchmark's questions.
  - Each has a **gold sequence** of Swagger operations, in order, with acceptable alternatives per step. A test checks that all of them exist in the spec.
- Scored the same way for both: did the agent hit the gold sequence?
  - **Code agent:** its plan's endpoints, and whether its Python parses.
  - **Build & run:** the operations its code really called, mapped from each bank call through the path templates; whether the run completed; whether a gold GET returned 2xx; whether each held write's body is valid against the schema.
  - Cost, time and requests for both.
- **Every Build & run write is auto-rejected,** so evals never change demo data. Operator runs stay out of the visitors' caps.
- **Agent eval** (manual workflow) runs it in a one-off container on the VM, never inside the live one, and posts the side-by-side table on the run page.
- **Replace rule:** Build & run replaces the code agent (F6) if it hits at least as many gold sequences overall, **and** a gold GET answered 2xx on at least 8 of the 11 read tasks.
- *Results (2026-11-09):*

  | | Code agent | Build & run, first run | Build & run, after the prompt fix |
  |---|---|---|---|
  | Gold sequence hit | 20/20 | 6/20 | **19/20** |
  | … read · write | 11/11 · 9/9 | 6/11 · 0/9 | 11/11 · 8/9 |
  | A gold GET answered 2xx (read) | — | 4/11 | **11/11** |
  | Held write bodies valid against the schema | — | none held | **8/8** |
  | Cost per task, median time | $0.019, 8 s | $0.052, 18 s | $0.081, 22 s |

  - **First run:** Build & run picked the right operations but stopped to ask for IDs that nobody could give in a website run. The fix (`9f276ea`): never ask; find a real record with GETs; make the write, since the visitor approves it; don't finish before the code has called the bank.
  - **The one remaining miss (task 20):** the code read the profile from `GET /customer/detail`, a path that is **not in the spec**, although the prompt says to use only operations the tools returned. The gold is right; the miss is real.
  - **The code agent's 20/20 scores plans with placeholders that were never run;** Build & run's scores come from calls the bank answered.
  - **Replace rule:** the 2xx part is met (11/11, needing 8). The gold-hit part is short by one task (19 vs 20).
  - **After tightening** the prompt (`c326bdd` with its fix `c2b06f5`: call only the `url_template`s that `get_endpoint` returned), two Build & run runs scored **20/20 and 20/20** (writes 9/9). A gold GET answered 2xx on 11/11 and 10/11, held write bodies were valid 8/8, at $0.10 per task, median 22 s.
  - **The replace rule is met** (20 ≥ 20; 2xx ≥ 8 of 11). F6 can switch the chat over.
- *Not yet:* an LLM judge, and regression cases mined from session logs (the two empty-tool-call prompts). Add them if scores and reading the answers disagree.

**Developer UX fixes** (2026-11-09, after a review of the live page):
- **Sidebar endpoints and Cookbook recipes** open an endpoint card built from the spec by `GET /api/endpoint/{name}`, with no model involved: method, path, parameters, an example body, and a complete curl with the required query parameters as `{placeholders}`. The card offers Try in Playground, Copy curl, Write code for it and Build & run it. Before, the keyword-search chat named the wrong endpoint (`GetAccountBalance` → `FAET2021…`).
- **Quick actions:**
  - Browse API areas lists 31 areas with example endpoints.
  - Search endpoints is a sidebar search over the semantic index.
  - Try a balance call is a real Playground request. The MCP execute path is a stub (`{"message": "MCP tool execution"}`), so no quick action uses it.
  - Authentication is answered directly: no token, three headers.
  - Python example goes to the code agent.
- **The welcome** gives one starter card per job (find an endpoint, Plan & code, Build & run) and the no-sign-in basics, replacing the MCP-tool tips.
- **Composer:** a Plan & code | Build & run switch, with the note and placeholder following the mode.
- **Fixed:** the active tab didn't follow programmatic switches.
- **Still open:** a full endpoint browser, real multi-step recipes, and a Connect via MCP panel (needs a real MCP endpoint).

**F6. Switch over** (done 2026-11-09; the owner chose to retire the code agent rather than keep it as a second mode):
- **Build & run is the chat's one agent.** The composer has no mode switch or language picker. When Build & run is off on a server (no runner token), typed questions fall back to the search chat, and the note says so.
- **Removed:**
  - `backend/code_agent.py`, the `/api/agent/chat|stream|code|status` routes, and their guard and `AGENT_*` settings;
  - `tests/test_agent.py`;
  - the code agent's UI (plan rendering, per-language code, clarifications);
  - the Plan & code starter card and "Write code for it";
  - `pydantic-ai` and `anthropic` from the cookbook image.
- **The Anthropic key is no longer sent to the main VM;** only the sandbox host calls Claude.
- **Kept:**
  - the Swagger lookups, now `backend/api_lookup.py`, used by the proxy's `/_kb/*`, the endpoint cards and the eval (their tests moved to `tests/test_api_lookup.py`);
  - `agent_guard.py` for Build & run's caps;
  - the search chat as the fallback.
- **The eval** (`backend/agent_eval.py`, *Agent eval*) now scores Build & run alone and reads the spec directly instead of starting the app.
- `docs/coding-agent-plan.md` milestones 3 and 4 (sessions, evals) are superseded by decode's session log and F5.

**MCP endpoint** (decided 2026-11-09):
- **Tools:** APIWeaver's four (`list_apis`, `call_api`, `get_api_schema`, `search_endpoints`), served over streamable HTTP at `https://apicookbook.vercel.app/api/mcp` for any MCP client, without a key.
- **Limit:** 300 requests per hour per IP.
- **Writes are held:** `call_api` with POST, PUT, PATCH or DELETE returns `pending_approval` and a link. A person opens `/#approve-<id>` on the site and approves within 10 minutes, which sends the stored request. The client then calls `call_api` with `{"approval_id": …}` for the result.
- **Also changed:** the Playground's MCP mode runs the real tools (it was a stub); the broken stdio `backend/mcp_server.py` is deleted.

## Risks

- **Anyone can run code and change shared demo data.** The sandbox protects our servers, but not the demo bank's data from other visitors. Mitigations: approval per write, caps, per-run limits, monitoring. The same machinery would make sign-in easy to add later.
- **Cost.** Subagents and test-and-fix loops make a Build & run answer cost many times today's $0.02. Because it replaces the cheap agent, every chat question now pays that price: simple lookups ("which endpoint updates an address?") too. The $5/day cap buys far fewer answers. Sandbox compute is a fixed, free instance, so the cost is Claude's tokens. Measure in F1, then decide the caps, and whether Build & run should skip the sandbox when no code needs running.
- **Prompt injection.** API responses and file contents the agent reads could try to steer it. The proxy-enforced approvals and the network lock keep the damage inside the sandbox and demo data.
- **Diverging from upstream.** A vendored, slimmed fork won't pull course updates easily. Record the upstream commit and our deletions in `NOTICE` and an ADR.
- **The pydantic-ai jump (2.46 to 2.66)** may break decode code in the loop and deferred-tool handling. F0 surfaces it.
- **One small sandbox host.** 1 OCPU runs about 2 sessions at once; the rest queue. The Always Free A1 allowance (4 OCPUs, 24 GB across instances) may leave room to grow it for free.
- **The sandbox host shares a subnet with the main VM.** A host compromise (a gVisor and Docker escape) would put it next to VAM and the other apps; the main VM's firewall rules above limit what it could reach.
