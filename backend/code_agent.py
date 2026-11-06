"""The code-writing agent (docs/coding-agent-plan.md, milestone 1).

A developer describes a BaNCS task; the agent searches the Swagger operations, reads their schemas,
plans the call sequence and writes code for it in one language. It never runs code or calls the bank.

Built like decode's harness (pydantic-ai ``agent.iter``/``run``, a flat registry of read-only tools,
an output validator), minus everything a read-only agent does not need (permission gate, sandbox).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import jsonschema
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.usage import UsageLimits

from backend.config import settings

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

Language = Literal["curl", "python", "javascript", "java"]
LANGUAGES: tuple[str, ...] = ("curl", "python", "javascript", "java")

MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
EFFORT = os.getenv("AGENT_EFFORT", "medium")
BASE_URL = settings.api_base_url.rstrip("/") + "/Core"  # the Swagger spec's basePath
LIMITS = UsageLimits(request_limit=12, tool_calls_limit=16)  # ponytail: a stuck loop costs at most ~12 requests

# USD per million tokens: (uncached input, output, cache read). Cache writes (5-minute TTL) cost 1.25x input.
# From Anthropic's price table (claude-api reference, 2026-10-06). A fallback can serve the request on
# another model; an unknown model is priced at the dearest rate so the caps never under-count.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-fable-5-1": (10.0, 50.0, 1.00),
}
DEAREST = max(PRICES.values())


# ---------------------------------------------------------------- answer shapes

class Clarification(BaseModel):
    """Ask this instead of writing code when the request maps to clearly different call sequences."""
    kind: Literal["clarification"] = "clarification"
    question: str
    choices: list[str] = Field(description="2 to 5 short options the developer can pick from")


class Step(BaseModel):
    endpoint: str = Field(description="operationId exactly as a tool returned it")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(description="the Swagger path template exactly as get_endpoint returned it")
    purpose: str
    request_body_json: Optional[str] = Field(
        None, description="example JSON body as a string, valid against get_body_schema; null when there is no body")
    uses_from_previous: Optional[str] = Field(
        None, description="which value from an earlier step this step uses, e.g. 'customerId from step 1'")


class Plan(BaseModel):
    kind: Literal["plan"] = "plan"
    steps: list[Step]
    language: Language
    code: str = Field(description="complete code for every step in the requested language")
    notes: str = ""
    assumptions: list[str] = Field(default_factory=list)


class Code(BaseModel):
    code: str


INSTRUCTIONS = f"""You write integration code for the TCS BaNCS banking APIs. You never run code or call any API yourself.

How to work:
- Search with search_endpoints, using short task phrases ("cancel cash block", "create customer"). Search once per sub-task; do not repeat a query.
- Use only endpoints a tool returned. Call get_endpoint for each one you use, and get_body_schema before writing any request body.
- If the request maps to clearly different call sequences (e.g. "open an account": savings, term deposit or loan), return a Clarification instead of a plan.
- Otherwise return a Plan: the steps in order, how values flow between them, and code for every step in the requested language.

BaNCS conventions:
- Base URL {BASE_URL}; append each step's path template and fill {{placeholders}} from variables.
- Every request sends these headers: entity: {settings.bancs_entity}, userId: {settings.bancs_user_id}, languageCode: {settings.bancs_language_code}, plus Content-Type: application/json when there is a body. No Authorization header.
- Request bodies must satisfy the endpoint's schema; use realistic example values.

Code style: curl uses one command per step with the variables at the top; Python uses httpx with one function per step; JavaScript uses fetch with async functions; Java uses the JDK's java.net.http.HttpClient only (no third-party libraries), in one compilable class.
Put assumptions you made in `assumptions`, and anything the developer must check in `notes`."""


@dataclass
class AgentDeps:
    kb: Any
    rag: Any = None
    language: str = "curl"
    trace: list[dict] = field(default_factory=list)  # tool calls, for the chat to show
    _common: Optional[frozenset] = None

    @property
    def common_params(self) -> frozenset:
        """Transport headers present on most calls (Accesstoken, ChannelType...): noise for the model."""
        if self._common is None:
            counts = Counter(k for ep in self.kb.endpoints.values()
                             if isinstance(ep.get("parameters"), dict) for k in ep["parameters"])
            self._common = frozenset(k for k, c in counts.items() if c > 0.3 * max(len(self.kb.endpoints), 1))
        return self._common


def _swagger(kb, name: str) -> Optional[dict]:
    ep = kb.endpoints.get(name)
    return ep if ep and ep.get("source") == "swagger" else None


agent: Agent[AgentDeps, Union[Plan, Clarification]] = Agent(
    deps_type=AgentDeps,
    output_type=NativeOutput([Plan, Clarification]),
    instructions=INSTRUCTIONS,
    # decode's budgets (agent/factory.py): with the default of 1, a second consecutive tool retry raises
    # UnexpectedModelBehavior and kills the whole answer; the counter resets only when the tool succeeds.
    retries={"output": 3, "tools": 5},
)
code_writer: Agent[None, Code] = Agent(
    output_type=NativeOutput(Code),
    instructions="You write integration code for the TCS BaNCS banking APIs. You never run code or call any API yourself.\n"
    "Rewrite the given plan's code in the requested language. Keep the same steps, paths, headers and bodies.\n\n"
    "BaNCS conventions:" + INSTRUCTIONS.split("BaNCS conventions:")[1],
)


# ---------------------------------------------------------------- tools (all read-only)

@agent.tool(strict=True)  # strict: the API guarantees the input matches the schema
def search_endpoints(ctx: RunContext[AgentDeps], query: str) -> list[dict]:
    """Find BaNCS Swagger operations by meaning. Returns the top 8 with a relevance score; judge relevance yourself."""
    kb, k = ctx.deps.kb, 8
    hits: list[tuple[str, float]] = []
    if ctx.deps.rag:
        for doc, score in ctx.deps.rag.search(query, top_k=k * 3, min_score=0):
            name = doc.metadata.get("endpoint_name")
            if _swagger(kb, name):
                hits.append((name, round(score, 3)))
    else:  # RAG disabled: fuzzy name match
        hits = [(n, round(s / 100, 3)) for n, s in kb.search_endpoints(query) if _swagger(kb, n)]
    ctx.deps.trace.append({"tool": "search_endpoints", "query": query})
    return [{"endpoint": n, "method": kb.endpoints[n]["method"], "path": kb.endpoints[n]["url"],
             "description": (kb.endpoints[n].get("description") or "")[:160],
             "area": kb.endpoints[n].get("category"), "score": s} for n, s in hits[:k]]


@agent.tool(strict=True)  # strict: the API guarantees the input matches the schema
def get_endpoint(ctx: RunContext[AgentDeps], name: str) -> dict:
    """Method, path template, parameters and whether a body is needed, for one Swagger operation."""
    ep = _swagger(ctx.deps.kb, name)
    ctx.deps.trace.append({"tool": "get_endpoint", "endpoint": name})
    if not ep:
        return {"error": f"{name} is not a Swagger operation; use a name search_endpoints returned"}
    params = {k: {f: v.get(f) for f in ("in", "type", "required", "description") if v.get(f) is not None}
              for k, v in (ep.get("parameters") or {}).items()
              if isinstance(v, dict) and k not in ctx.deps.common_params and v.get("in") != "body"}
    return {"endpoint": name, "method": ep["method"], "path": ep["url"], "url_template": BASE_URL + ep["url"],
            "description": ep.get("description"), "area": ep.get("category"), "parameters": params,
            "has_body": bool(ep.get("body_schema"))}


@agent.tool(strict=True)  # strict: the API guarantees the input matches the schema
def get_body_schema(ctx: RunContext[AgentDeps], name: str) -> Any:
    """The JSON Schema of the operation's request body, with every definition inlined."""
    ctx.deps.trace.append({"tool": "get_body_schema", "endpoint": name})
    if not _swagger(ctx.deps.kb, name):
        return {"error": f"{name} is not a Swagger operation"}
    schema = ctx.deps.kb.resolved_body_schema(name)
    return compact(schema) if schema else "This operation takes no request body."


@agent.tool(strict=True)  # strict: the API guarantees the input matches the schema
def list_endpoints(ctx: RunContext[AgentDeps], area: str) -> Any:
    """Every Swagger operation in one business area (e.g. 'AccountManagement'), to browse when search misses."""
    kb = ctx.deps.kb
    ctx.deps.trace.append({"tool": "list_endpoints", "area": area})
    rows = [{"endpoint": n, "method": e["method"], "path": e["url"], "description": (e.get("description") or "")[:80]}
            for n, e in kb.endpoints.items() if e.get("source") == "swagger" and e.get("category") == area]
    if not rows:
        return {"error": f"unknown area {area!r}",
                "areas": sorted({e.get("category") for e in kb.endpoints.values() if e.get("source") == "swagger"})}
    if len(rows) > 80:
        return {"shown": 80, "total": len(rows), "note": "use search_endpoints to narrow this area", "rows": rows[:80]}
    return rows


MAX_TOOL_CHARS = 30_000  # ~8k tokens; decode caps tool output too (tools/truncate.py). 3 of 495 schemas exceed it.


def compact(schema: dict) -> Any:
    """A schema small enough for the model: as is when it fits, else without prose, else cut with a note."""
    if len(json.dumps(schema)) <= MAX_TOOL_CHARS:
        return schema

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("description", "title", "example")}
        return [strip(v) for v in node] if isinstance(node, list) else node

    lean = strip(schema)
    text = json.dumps(lean)
    if len(text) <= MAX_TOOL_CHARS:
        return lean
    cut = MAX_TOOL_CHARS
    while True:  # the cut text is a JSON string, so escaping grows it: shrink until the whole result fits
        result = {"truncated": True, "note": "schema too large; field names and types only, cut at the limit",
                  "schema_start": text[:cut]}
        if len(json.dumps(result)) <= MAX_TOOL_CHARS:
            return result
        cut = int(cut * 0.9)


# ---------------------------------------------------------------- the guard against invented APIs

def plan_problems(kb, plan: Plan, language: str) -> list[str]:
    """Everything wrong with a plan: unknown endpoints, wrong method or path, bodies that fail their schema."""
    problems = []
    if plan.language != language:
        problems.append(f"the code must be in {language}, not {plan.language}")
    for i, step in enumerate(plan.steps, 1):
        ep = _swagger(kb, step.endpoint)
        if not ep:
            problems.append(f"step {i}: {step.endpoint} is not a Swagger operation a tool returned")
            continue
        if step.method != ep["method"]:
            problems.append(f"step {i}: {step.endpoint} is {ep['method']}, not {step.method}")
        if step.path != ep["url"]:
            problems.append(f"step {i}: path must be the template {ep['url']}")
        if step.request_body_json:
            try:
                body = json.loads(step.request_body_json)
            except ValueError:
                problems.append(f"step {i}: request_body_json is not valid JSON")
                continue
            schema = kb.resolved_body_schema(step.endpoint)
            if schema is None:
                problems.append(f"step {i}: {step.endpoint} takes no body")
                continue
            for err in list(jsonschema.Draft4Validator(schema).iter_errors(body))[:3]:
                where = "/".join(map(str, err.absolute_path)) or "body"
                problems.append(f"step {i}: {where}: {err.message}")
    return problems


@agent.output_validator
def _validate(ctx: RunContext[AgentDeps], output: Union[Plan, Clarification]) -> Union[Plan, Clarification]:
    if isinstance(output, Plan):
        problems = plan_problems(ctx.deps.kb, output, ctx.deps.language)
        if problems:
            raise ModelRetry("Fix these problems and return the plan again: " + "; ".join(problems))
    return output


# ---------------------------------------------------------------- running and pricing

def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def build_model():
    """The model a run uses. Tests replace this with a FunctionModel."""
    return f"anthropic:{MODEL}"


MODEL_SETTINGS = AnthropicModelSettings(
    anthropic_effort=EFFORT,
    anthropic_betas=["server-side-fallback-2026-07-01"],
    extra_body={"fallbacks": "default"},  # a declined request reruns on another model inside the same call
    anthropic_cache_instructions=True,
    anthropic_cache_tool_definitions=True,
    max_tokens=16000,
)


def usd(messages) -> float:
    """What a run cost, priced per response by the model that actually served it (fallbacks included)."""
    total = 0.0
    for m in messages:
        if not isinstance(m, ModelResponse):
            continue
        u = m.usage
        rate_in, rate_out, rate_read = next(
            (p for name, p in PRICES.items() if (m.model_name or "").startswith(name)), DEAREST)
        write = u.cache_write_tokens or 0
        read = u.cache_read_tokens or 0
        uncached = max((u.input_tokens or 0) - write - read, 0)
        total += (uncached * rate_in + write * rate_in * 1.25 + read * rate_read + (u.output_tokens or 0) * rate_out) / 1e6
    return total


async def answer(question: str, language: str, kb, rag=None):
    """Run the agent on one question; returns (result, deps) so callers can read the trace and usage."""
    deps = AgentDeps(kb=kb, rag=rag, language=language)
    result = await agent.run(f"Language: {language}\n\nTask: {question}", deps=deps, model=build_model(),
                             model_settings=MODEL_SETTINGS, usage_limits=LIMITS)
    return result, deps


async def rewrite(plan: Plan, language: str):
    """Write the same plan's code in another language, from the plan alone (no tools, so it is cheap)."""
    steps = plan.model_dump(include={"steps", "notes", "assumptions"})
    return await code_writer.run(f"Language: {language}\n\nPlan:\n{json.dumps(steps, indent=1)}",
                                 model=build_model(), model_settings=MODEL_SETTINGS,
                                 usage_limits=UsageLimits(request_limit=3))
