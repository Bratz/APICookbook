# Code-writing agent for APICookbook: intent and plan

Status: proposed · Decided with the owner in an interview on 2026-11-06 · Reference: [decodingai-magazine/building-a-coding-agent-from-scratch-course](https://github.com/decodingai-magazine/building-a-coding-agent-from-scratch-course) ("decode", Apache-2.0)

## Intent

A developer describes a banking task in plain words ("the customer moved house, update their address and confirm it"). The agent finds the right TCS BaNCS endpoints among the 950, reads their schemas, decides the order of calls and how data flows between them, and writes working code for that sequence in **curl, Python, JavaScript/TypeScript and Java**. Each request also opens in the Playground with method, URL, headers and body filled in.

The agent **writes code; it never runs it and never calls the bank**. The developer stays in control: they read the plan, run the code or press Execute in the Playground themselves.

It is built the way the decode course builds a coding agent: a small tool-calling loop on **pydantic-ai**, and the real work in the harness around it (tools, guardrails, sessions, evals).

**Done means** the eval suite below passes its thresholds on the deployed agent, the public site stays inside its spend caps, and the old BERT chat still answers when the agent is unavailable.

**Not in scope:** executing requests or code, editing files or repos, sign-in, local models, the Search KB bar (removed).

## Decisions

| Topic | Decision | Notes |
|---|---|---|
| Job | **Code writer only** | Read-only tools. No sandbox, no approval prompts, no writes to the demo bank. |
| Model | **Claude via the Anthropic API** | `ANTHROPIC_MODEL` default `claude-sonnet-5`, configurable. Key lives only on the server. |
| Loop | **pydantic-ai, as in the course** | Requires Python ≥ 3.11 (current `pydantic-ai-slim` 2.66.0 says `>=3.11`; decode uses 3.12). This app runs 3.10.11, so milestone 0 upgrades it. |
| Access | **Open, with caps** | 10 questions/hour per client IP, hard caps of **$5/day and $50/month**. At a cap the agent says so and the free BERT chat answers instead. |
| UI | **Chat becomes the agent** | Streamed steps, the plan, code in 4 tabs with Copy and "Open in Playground". The Search KB bar and the sidebar "Search Knowledge" item go. |
| Code targets | **curl, Python, JavaScript/TypeScript, Java** | Python on `httpx`, JS/TS on `fetch`, Java on the JDK's built-in `java.net.http.HttpClient` (confirmed: no Spring, no dependencies, so `javac` alone can check it). |
| Sessions | **Server session log (JSONL)** | Full conversations kept **30 days**, then deleted automatically. A notice under the chat says so and asks people not to paste real customer data. |
| Evals | **Gold call sequences + static checks + live GET checks + LLM judge** | Run by hand from GitHub Actions, not on every push. The judge (confirmed) is a second Claude call that grades each answer, so each eval run costs roughly twice as much. |

## What we take from decode, and what we skip

| decode piece (ADR) | Here | Why |
|---|---|---|
| `agent.iter()` loop (ADR-0002) | **Keep** | The core of the course; streams model and tool events. |
| Flat tool registry with tool kinds (ADR-0002 §7, ADR-0003) | **Keep, read-only tools only** | Every tool here reads the knowledge base, so every tool is "allow". |
| Permission gate, modes, approvals (ADR-0003) | **Skip** | Nothing mutates. The safety lives in the tool design. |
| Sandbox, credential proxy, workspace (ADR-0011/0012/0016) | **Skip** | No code runs. |
| Steering queue (ADR-0002) | **Skip in v1** | A Stop button is enough for one-shot answers. |
| Memory: AGENTS.md (ADR-0002) | **Adapt** | A fixed system prompt with BaNCS conventions: base URL, the three headers, Postman vs Swagger names, "only use endpoints a tool returned". |
| Session log JSONL + resume (ADR-0002 §9) | **Keep** | Follow-ups ("now in Java") resume from the server log by session id. |
| Compaction (ADR-0006/0018) | **Skip in v1** | Cap history at the last N turns; revisit if sessions grow. |
| Provider seam (ADR-0005) | **Free with pydantic-ai** | Anthropic only in v1; OpenRouter or OpenAI-compatible later by config. |
| Observability, cost (ADR-0014) | **Adapt** | A usage ledger (tokens, USD, tools, timing) instead of Opik; it also drives the caps. |
| Evals: benchmark + regression (ADR-0017/0022) | **Keep** | Gold-sequence tasks plus regression cases mined from session logs. |
| Skills, subagents, LSP, remote runs | **Skip** | Not needed for one read-only agent. |

## Architecture

```
Browser chat ──POST /api/agent/chat (SSE)──▶ Vercel rewrite ──▶ Caddy ──▶ cookbook-api
                                                               (flush_interval -1, as uc05)
cookbook-api
  guard:   per-IP rate limit (10/h) · daily/monthly USD caps from the usage ledger
  agent:   pydantic-ai Agent(model=anthropic:${ANTHROPIC_MODEL}, deps=KB+RAG, output_type=AgentAnswer)
  tools:   search_endpoints(query, k)   → the BERT index (rag_engine), method/path/description
           get_endpoint(name)           → method, path, own parameters (transport headers dropped), area
           get_body_schema(name)        → the resolved schema behind /api/body-schema
           list_endpoints(category)     → names in one business area, for browsing
  output:  AgentAnswer {
             steps: [{endpoint, method, path, purpose, request_body, uses_from_previous}],
             code:  {curl, python, javascript, java},
             notes, clarifying_question?
           }
  checks:  output validator rejects any step whose endpoint is not in the KB or whose body
           fails its schema, and sends it back to the model (ModelRetry): no invented APIs
  logs:    sessions/<yyyy-mm-dd>/<session_id>.jsonl (30-day purge) · usage ledger
```

- **Structured output, not free text.** The UI's Playground buttons and the evals both need the steps as data. The model's prose goes in `notes`.
- **Streaming** sends events for each tool call ("Searched: change address", "Read schema: RSTBTAmendPostalAddressUsingPUT"), then the final answer.
- **Secrets:** `ANTHROPIC_API_KEY` becomes a GitHub repository secret. The deploy workflow writes it to `deploy/oci/.env` on the VM (mode 600), and compose reads it. It is never sent to the browser.
- **Client IP** comes from `X-Forwarded-For`, set by Vercel and Caddy. Only the first hop added by our own proxies is trusted.
- **Prices** for the cost ledger are settings, filled from Anthropic's current price list when milestone 3 is built, not from memory.

## Milestones

Each one ships on its own, as the course does.

**0. Python 3.12.** Upgrade the Dockerfile, CI and local venv. Re-pin `requirements.txt` on 3.12 and keep `torch==2.12.0+cpu`, which has cp312 wheels for both aarch64 and x86_64 (checked). Add `pydantic-ai-slim[anthropic]` pinned to one minor version. The 38 existing tests must pass.

**1. Vanilla agent (backend).** Build the agent, the four read-only tools, the system prompt and `AgentAnswer` with its output validator. Expose `POST /api/agent/chat` without streaming. Unit tests use pydantic-ai's `TestModel`/`FunctionModel` (no API cost), plus a small script to try real questions. Exit: 5 hand-picked tasks produce correct steps and compiling code.

**2. Streaming chat UI.** SSE events; the chat renders steps, the plan and 4 code tabs with Copy and "Open in Playground". The Search KB bar and sidebar item are removed. When the agent is down or capped, the existing BERT chat answers instead. Check that SSE streams through both Vercel and Caddy.

**3. Guardrails.** Per-IP rate limit, daily and monthly caps from the usage ledger, a usage readout (today's spend, answers, average cost), and the notice under the chat.

**4. Sessions.** JSONL session log, resume by session id for follow-ups, history limited to the last N turns, and a daily purge of files older than 30 days.

**5. Evals.** About 20 multi-step tasks with gold endpoint sequences, built from the 30 retrieval questions already in the tests. Scorers:
  - the right endpoints in the right order;
  - request bodies valid against their schemas (`jsonschema`);
  - code that compiles: Python `ast.parse`, `node --check`, `javac`, and a parse of the curl line (URL on the BaNCS host, headers present);
  - live read-only checks: every GET step is called against the demo host during evals only, and must return 2xx;
  - an LLM judge: a second Claude call scores each answer against a short rubric (the plan fits the task, the steps pass data correctly between calls, the notes are accurate), with its reasons kept in the eval report.

  It runs as a `workflow_dispatch` GitHub Action with the API key secret, with thresholds that fail the run. Regression cases are mined from session logs.

**Later, if wanted:** expose the agent as an MCP tool for Claude Desktop and Cursor; compaction; other model providers.

## Risks and open points

- **Python upgrade churn.** fastmcp, numpy and others may need new pins on 3.12. That's why milestone 0 stands alone.
- **Cost per answer is unmeasured.** Four languages per answer make outputs long. Measure in milestone 1, and lower caps or generate only the selected language if needed.
- **Invented endpoints or fields.** Mitigated by the output validator, schema checks and evals.
- **Stored conversations on a shared VM.** Mitigated by 30-day deletion, the notice, and the log directory on its own volume, readable only by the app.
- **SSE through Vercel.** uc05 already streams SSE through the same Caddy (`flush_interval -1`). Confirm Vercel's external rewrite doesn't buffer it; if it does, stream straight from the sslip host.
- **Disk on the VM** is about 7 GB free. Logs are small, but the Python 3.12 image rebuild is a full rebuild.
