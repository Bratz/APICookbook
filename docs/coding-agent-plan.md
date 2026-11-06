# Code-writing agent for APICookbook: intent and plan

Status: proposed · Decided with the owner in two interviews on 2026-11-06 · Reference: [decodingai-magazine/building-a-coding-agent-from-scratch-course](https://github.com/decodingai-magazine/building-a-coding-agent-from-scratch-course) ("decode", Apache-2.0)

## Intent

A developer describes a banking task in plain words ("the customer moved house, update their address and confirm it"). The agent finds the right TCS BaNCS endpoints, reads their schemas, decides the order of calls and how data flows between them, and writes working code for that sequence in the developer's chosen language: **curl, Python, JavaScript/TypeScript or Java**. Each request also opens in the Playground with method, URL, headers and body filled in. When the request is ambiguous, the agent asks one clarifying question before writing code.

This is the **generation** half of RAG. The retrieval half (the BERT index over the endpoints) already exists. The agent decides what to search for, reads the endpoints and schemas it found, and writes from them.

The agent **writes code; it never runs it and never calls the bank**. The developer stays in control: they read the plan, then run the code or press Execute in the Playground themselves. Only the eval suite makes live calls, and only read-only GETs.

It is built the way the decode course builds a coding agent: a small tool-calling loop on **pydantic-ai**, and the real work in the harness around it (tools, guardrails, sessions, evals).

**Done means:**
- the eval suite below passes its thresholds;
- the public site stays inside its spend caps;
- the old BERT chat still answers when the agent is unavailable or capped.

**Not in scope:** executing requests or code, editing files or repos, sign-in, local models, the Search KB bar (removed).

## Decisions

| Topic | Decision | Notes |
|---|---|---|
| Job | **Code writer only** | Read-only tools. No sandbox, no approval prompts, no writes to the demo bank. |
| Model | **`claude-sonnet-5-5`** | $2 / $10 per million tokens. Effort starts at `medium` and is measured against `high` in milestone 1. The key lives only on the server. |
| Refusals | **Server-side fallback on** | `fallbacks: "default"` with beta `server-side-fallback-2026-07-01`. A declined request reruns on another Claude model in the same call, billed at that model's rates and counted toward the caps. If every model declines, the chat says so plainly. |
| Loop | **pydantic-ai, as in the course** | `pydantic-ai-slim[anthropic]==2.66.0` on Python 3.12 (milestone 0, done). |
| Answers | **Native structured output** | Forcing a tool call (`tool_choice` any or tool) returns a 400 on this model, so pydantic-ai must use the API's native structured output, not its tool-output mode. |
| Code targets | **The plan plus the chosen language** | The developer picks curl, Python, JS/TS or Java; the choice is remembered in the browser and defaults to curl. Other languages generate on click from the same plan. Python uses `httpx`, JS/TS uses `fetch`, and Java uses the JDK's built-in `java.net.http.HttpClient` (no Spring). |
| Ambiguity | **Ask first when unclear** | Instead of code, the answer can be one clarifying question with choices. |
| Endpoints | **Swagger only** | The agent sees the 927 Swagger operations, which have path templates and body schemas. The 23 Postman and default entries are hidden; 22 have a Swagger twin, so only `CreateTransaction` is lost. |
| Access | **Open, with caps built in milestone 1** | 10 questions/hour per client IP, hard caps of **$5/day and $50/month**. At a cap the agent says so and the BERT chat answers instead. |
| UI | **Chat becomes the agent** | Streamed steps, the plan, code with Copy and "Open in Playground". The Search KB bar and the sidebar "Search Knowledge" item go. |
| Sessions | **Server session log (JSONL), 30 days** | Deleted automatically after 30 days. A notice under the chat says conversations are stored, and asks people not to paste real customer data. |
| Evals | **Gold call sequences + static checks + live GET checks + LLM judge** | Run by hand from GitHub Actions. The judge is `claude-sonnet-5-5`, cheaper, but it may grade Sonnet's own answers leniently, so its rubric asks for evidence. |
| Local key | **`ANTHROPIC_API_KEY` in the local `.env`** | `.env` is git-ignored. In production the key is a GitHub secret, written to `deploy/oci/.env` on the VM (mode 600). |

## What we take from decode, and what we skip

| decode piece (ADR) | Here | Why |
|---|---|---|
| `agent.iter()` loop (ADR-0002) | **Keep** | The core of the course; it streams model and tool events. |
| Flat tool registry with tool kinds (ADR-0002 §7, ADR-0003) | **Keep, read-only tools only** | Every tool reads the knowledge base, so every tool is allowed. |
| Permission gate, modes, approvals (ADR-0003) | **Skip** | Nothing mutates. The safety lives in the tool design. |
| Sandbox, credential proxy, workspace (ADR-0011/0012/0016) | **Skip** | No code runs. |
| Steering queue (ADR-0002) | **Skip in v1** | A Stop button is enough for one-shot answers. |
| Memory: AGENTS.md (ADR-0002) | **Adapt** | A fixed system prompt with the BaNCS conventions: base URL, the three headers, "only use endpoints a tool returned". It is cached (prompt caching), since it never changes. |
| Session log JSONL + resume (ADR-0002 §9) | **Keep** | Follow-ups ("now in Java") resume from the server log by session id. History is capped at the last 10 turns. |
| Compaction (ADR-0006/0018) | **Skip in v1** | The 10-turn cap keeps context small. |
| Provider seam (ADR-0005) | **Free with pydantic-ai** | Anthropic only in v1. |
| Observability, cost (ADR-0014) | **Adapt** | A usage ledger (tokens, USD, tools, timing) instead of Opik. It also drives the caps. |
| Evals: benchmark + regression (ADR-0017/0022) | **Keep** | Gold-sequence tasks plus regression cases mined from session logs. |
| Skills, subagents, LSP, remote runs | **Skip** | Not needed for one read-only agent. |

## Architecture

```
Browser chat ──fetch POST /api/agent/chat, read as a stream──▶ Vercel rewrite ──▶ Caddy ──▶ cookbook-api
             (EventSource can't POST)                                     (flush_interval -1, as uc05)
cookbook-api
  guard:   per-IP rate limit (10/h) · daily and monthly USD caps from the usage ledger
  agent:   pydantic-ai Agent(model=anthropic:claude-sonnet-5-5, effort=medium, fallbacks=default,
                             deps=KB+RAG, output=native structured AgentAnswer)
  tools:   search_endpoints(query, k)  → top k Swagger operations with scores, no cut-off
                                         (the 0.62 cut-off stays for the BERT chat; the agent judges relevance)
           get_endpoint(name)          → method, path template, own parameters, area
           get_body_schema(name)       → the resolved schema behind /api/body-schema
           list_endpoints(category)    → names in one business area, for browsing
  output:  AgentAnswer = Clarification { question, choices }
                       | Plan { steps: [{endpoint, method, path, purpose, request_body, uses_from_previous}],
                                language, code, notes, assumptions }
  checks:  output validator rejects any step whose endpoint is not a Swagger operation, or whose
           body fails its schema, and sends it back to the model (ModelRetry): no invented APIs
  extra:   POST /api/agent/code {plan, language} writes another language from an existing plan
  logs:    sessions/<yyyy-mm-dd>/<session_id>.jsonl (30-day purge) · usage ledger
```

- **Structured output, not free text.** The Playground buttons and the evals both need the steps as data. Prose goes in `notes`.
- **Streaming** sends an event per tool call ("Searched: change address", "Read schema: RSTBTAmendPostalAddressUsingPUT"), then the answer.
- **Client IP** comes from `X-Forwarded-For`. Calls straight to the VM's address can forge it, so the per-IP limit is advisory there, and the USD caps are the real guard. Caddy can restrict `/api/agent/*` to traffic that came through Vercel if that becomes a problem.
- **Prices** for the ledger are settings: $2 / $10 per million for `claude-sonnet-5-5`, from Anthropic's current price table, plus the fallback model's rates when a fallback runs.

## Milestones

Each one ships on its own, as the course does.

**0. Python 3.12. Done.** The image and CI use 3.12.15, and pydantic-ai 2.66.0 and anthropic 2.6.0 are pinned. 38 of 38 tests pass, and it's deployed.

**1. Agent and caps (backend). Done locally, exit check passed (2026-11-06).** Measured on the 5 tasks:
- Correct endpoints in order: 5/5 at `medium` and 5/5 at `high` effort.
- Code: 25/25 compiled or linted (curl, Python, JavaScript, Java).
- Cost: $0.016 per answer at medium vs $0.020 at high, so effort stays `medium`, about 300 answers per $5 day. Another language costs about $0.007.
- Latency: about 9 s per answer.

Fixes from the run and from reviewing decode's code:
- Strict tool schemas. Sonnet 5.5 sometimes called a tool with `{}` when native structured output was on; this failed 4 of 10 runs before the fix and 0 of 4 after.
- decode's retry budgets (`tools=5, output=3`).
- Failed runs priced into the ledger.
- Tool output capped at 30k chars: 3 of 495 schemas exceed it, the largest at 107 KB.

Deployed without a key, the agent answers 503.
1. One real call confirms native structured output and the refusal fallback work through pydantic-ai on `claude-sonnet-5-5`.
2. The four tools (Swagger only), the cached system prompt, `AgentAnswer` with its validator, and `POST /api/agent/chat` (not streaming yet) plus `POST /api/agent/code`.
3. The guard: per-IP rate limit, daily and monthly caps, and the usage ledger.
4. Unit tests on pydantic-ai's `TestModel`/`FunctionModel`, so they cost nothing.

Exit: 5 hand-picked tasks give correct steps and compiling code in each language. Measured cost per answer and medium vs high effort are reported. Production gets the key only when this passes.

**2. Streaming chat UI.** The chat renders steps, the plan or the clarifying question, a language picker, and Copy and "Open in Playground". The Search KB bar and sidebar item are removed. The BERT chat answers when the agent is down or capped, and the notice about stored conversations goes under the chat. SSE is checked through both Vercel and Caddy.

**3. Sessions.** The JSONL session log, resume by session id, history capped at 10 turns, and a daily purge of files older than 30 days.

**4. Evals.** About 20 multi-step tasks with gold Swagger endpoint sequences, built from the 30 retrieval questions already in the tests. Run as a `workflow_dispatch` GitHub Action with the key secret. Scorers and thresholds that fail the run:

| Scorer | Threshold |
|---|---|
| right endpoints in the right order | ≥ 70% of tasks |
| request bodies valid against their schemas (`jsonschema`) | 100% of bodies produced |
| code compiles: Python `ast.parse`, `node --check`, `javac`, curl parse (BaNCS host, headers present) | 100% |
| live read-only checks: every GET step called against the demo host, evals only | ≥ 90% return 2xx |
| LLM judge (`claude-sonnet-5-5`): plan fits the task, data passes correctly between steps, notes accurate; it must quote the step it criticises | mean ≥ 4 / 5 |

Each failed task records whether the correct endpoint was ever returned by a search, which separates retrieval failures from planning failures. Regression cases are mined from session logs.

**Later, if wanted:** expose the agent as an MCP tool for Claude Desktop and Cursor; compaction; other model providers; query expansion with BaNCS abbreviations if the evals show retrieval is the bottleneck.

## Risks and open points

- **Cost per answer is unmeasured** until milestone 1. Generating one language per answer keeps it lower; the caps hold either way.
- **Invented endpoints or fields.** Mitigated by Swagger-only tools, the output validator, schema checks and the evals.
- **A lenient judge.** The judge is the same model as the agent, so its rubric demands quoted evidence, and its scores are checked against the objective scorers.
- **Stored conversations on a shared VM.** Mitigated by 30-day deletion, the notice, and the log directory on its own volume, readable only by the app.
- **SSE through Vercel.** uc05 already streams SSE through the same Caddy (`flush_interval -1`). Confirm Vercel's external rewrite doesn't buffer it; if it does, stream straight from the sslip host.
- **Disk on the VM** had 8 GB free at the last deploy. Session logs are small.
