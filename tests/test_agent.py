"""The code-writing agent (docs/coding-agent-plan.md, milestone 1). No API calls: the model is a FunctionModel."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from backend import code_agent
from backend.agent_guard import AgentGuard
from backend.main import app

CASH_BLOCK = "AMBTCrtCashBlockWrapperUsingPOST"


def deps(language="curl"):
    return code_agent.AgentDeps(kb=app.state.kb, rag=app.state.rag_engine, language=language)


def ctx(language="curl"):
    return SimpleNamespace(deps=deps(language))


def plan(**over):
    step = {"endpoint": CASH_BLOCK, "method": "POST", "path": "/accountManagement/account/cashBlock",
            "purpose": "block funds", "request_body_json": json.dumps({"cashBlock": {"blockAmount": 100}})}
    step.update(over.pop("step", {}))
    data = {"kind": "plan", "steps": [step], "language": "curl", "code": "curl -X POST ...", "notes": "", "assumptions": []}
    data.update(over)
    return data


def as_output(kind, data):
    return ModelResponse(parts=[TextPart(json.dumps({"result": {"kind": kind, "data": data}}))])


def scripted(*responses):
    """A FunctionModel that replays the given responses in order, one per model request."""
    queue = list(responses)
    return FunctionModel(lambda messages, info: queue.pop(0))


@pytest.fixture
def agent_on(client, monkeypatch, tmp_path):
    """Agent 'configured', with a fresh guard (own ledger) for each test."""
    monkeypatch.setattr(code_agent, "available", lambda: True)
    monkeypatch.setattr(app.state, "agent_guard", AgentGuard(tmp_path, per_hour=10, daily_usd=5, monthly_usd=50))

    def use(*responses):
        monkeypatch.setattr(code_agent, "build_model", lambda: scripted(*responses))
    return use


# ---------------------------------------------------------------- tools

def test_search_returns_swagger_ops_with_scores_and_no_cutoff(client):
    if not app.state.rag_engine:
        pytest.skip("RAG_ENABLED is off")
    hits = code_agent.search_endpoints(ctx(), "cancel a cash block")
    assert "AMBTCancelCashBlockWrapperUsingDELETE" in [h["endpoint"] for h in hits]
    assert all(app.state.kb.endpoints[h["endpoint"]]["source"] == "swagger" for h in hits)
    # the BERT chat's 0.62 cut-off would return nothing here; the agent still gets candidates to judge
    assert len(code_agent.search_endpoints(ctx(), "zzqx nonsense")) == 8


def test_get_endpoint_hides_transport_headers_and_postman_entries(client):
    c = ctx()
    info = code_agent.get_endpoint(c, CASH_BLOCK)
    assert info["path"] == "/accountManagement/account/cashBlock" and info["has_body"] is True
    assert info["url_template"].endswith("/Core/accountManagement/account/cashBlock")
    assert not set(info["parameters"]) & c.deps.common_params
    assert "error" in code_agent.get_endpoint(c, "GetAccountBalance")  # a Postman entry


def test_get_body_schema(client):
    schema = code_agent.get_body_schema(ctx(), CASH_BLOCK)
    assert "createCashBlockResource" in schema["definitions"]
    assert code_agent.get_body_schema(ctx(), "CBPETGetAccountBalanceUsingGET") == "This operation takes no request body."


def test_list_endpoints_by_area(client):
    rows = code_agent.list_endpoints(ctx(), "AccountManagement")
    assert rows and all(app.state.kb.endpoints[r["endpoint"]]["category"] == "AccountManagement" for r in rows)
    assert "areas" in code_agent.list_endpoints(ctx(), "NoSuchArea")


# ---------------------------------------------------------------- the validator

def test_plan_problems(client):
    kb = app.state.kb
    P = code_agent.Plan.model_validate
    assert code_agent.plan_problems(kb, P(plan()), "curl") == []
    assert "not a Swagger operation" in code_agent.plan_problems(kb, P(plan(step={"endpoint": "MadeUpOp"})), "curl")[0]
    assert "template" in code_agent.plan_problems(kb, P(plan(step={"path": "/cashBlock"})), "curl")[0]
    bad_body = P(plan(step={"request_body_json": json.dumps({"cashBlock": {"blockAmount": "lots"}})}))
    assert "is not of type 'number'" in code_agent.plan_problems(kb, bad_body, "curl")[0]
    assert "must be in java" in code_agent.plan_problems(kb, P(plan()), "java")[0]


def test_usd_prices_each_response_by_its_model():
    def resp(model, inp, out, read=0, write=0):
        return ModelResponse(parts=[], model_name=model, usage=RequestUsage(
            input_tokens=inp, output_tokens=out, cache_read_tokens=read, cache_write_tokens=write))
    # sonnet: 1M uncached in ($2) + 1M out ($10); fallback to opus: 1M out ($20); unknown model: dearest rates
    assert code_agent.usd([resp("claude-sonnet-5-5", 1_000_000, 1_000_000)]) == pytest.approx(12.0)
    assert code_agent.usd([resp("claude-opus-5-5", 0, 1_000_000)]) == pytest.approx(20.0)
    assert code_agent.usd([resp("claude-sonnet-5-5", 1_000_000, 0, read=1_000_000)]) == pytest.approx(0.20)
    assert code_agent.usd([resp("mystery-model", 0, 1_000_000)]) == pytest.approx(50.0)


# ---------------------------------------------------------------- the routes, end to end

def test_chat_plans_with_tools_and_records_usage(client, agent_on):
    agent_on(
        ModelResponse(parts=[ToolCallPart("search_endpoints", {"query": "block funds"})], model_name="claude-sonnet-5-5"),
        as_output("Plan", plan()),
    )
    r = client.post("/api/agent/chat", json={"message": "put a hold on funds", "language": "curl"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]["kind"] == "plan" and body["answer"]["steps"][0]["endpoint"] == CASH_BLOCK
    assert body["trace"] == [{"tool": "search_endpoints", "query": "block funds"}]
    assert body["usage"]["requests"] == 2 and body["usage"]["tool_calls"] == 1
    assert app.state.agent_guard.status()["answers_today"] == 1


def test_chat_retries_an_invented_endpoint(client, agent_on):
    agent_on(as_output("Plan", plan(step={"endpoint": "MadeUpOp"})), as_output("Plan", plan()))
    r = client.post("/api/agent/chat", json={"message": "put a hold on funds"})
    assert r.status_code == 200 and r.json()["answer"]["steps"][0]["endpoint"] == CASH_BLOCK
    assert r.json()["usage"]["requests"] == 2  # the first answer went back to the model


def test_chat_can_ask_a_clarifying_question(client, agent_on):
    agent_on(as_output("Clarification", {"kind": "clarification", "question": "Which account type?",
                                          "choices": ["Savings", "Term deposit"]}))
    answer = client.post("/api/agent/chat", json={"message": "open an account"}).json()["answer"]
    assert answer["kind"] == "clarification" and answer["choices"] == ["Savings", "Term deposit"]


def test_code_route_rewrites_a_plan(client, agent_on):
    agent_on(ModelResponse(parts=[TextPart(json.dumps({"code": "import httpx"}))]))
    r = client.post("/api/agent/code", json={"plan": plan(), "language": "python"})
    assert r.status_code == 200 and r.json()["code"] == "import httpx"


def test_agent_unavailable_without_a_key(client, monkeypatch):
    monkeypatch.setattr(code_agent, "available", lambda: False)
    r = client.post("/api/agent/chat", json={"message": "anything"})
    assert r.status_code == 503 and r.json()["error"] == "agent_unavailable"


def test_rate_limit_and_daily_cap(client, agent_on, monkeypatch, tmp_path):
    agent_on(as_output("Clarification", {"kind": "clarification", "question": "?", "choices": ["a", "b"]}))
    monkeypatch.setattr(app.state, "agent_guard", AgentGuard(tmp_path / "rl", per_hour=1))
    assert client.post("/api/agent/chat", json={"message": "x"}).status_code == 200
    second = client.post("/api/agent/chat", json={"message": "x"})
    assert second.status_code == 429 and second.json()["error"] == "rate_limited"

    monkeypatch.setattr(app.state, "agent_guard", AgentGuard(tmp_path / "cap", daily_usd=0))
    capped = client.post("/api/agent/chat", json={"message": "x"})
    assert capped.status_code == 429 and capped.json()["error"] == "daily_cap"


def test_ledger_survives_a_restart(tmp_path):
    AgentGuard(tmp_path).record(kind="chat", usd=4.9, outcome="ok", seconds=1)
    reloaded = AgentGuard(tmp_path, daily_usd=5)
    assert reloaded.status()["today_usd"] == pytest.approx(4.9)
    AgentGuard(tmp_path).record(kind="chat", usd=0.2, outcome="ok", seconds=1)
    assert AgentGuard(tmp_path, daily_usd=5).check("1.2.3.4")[0] == "daily_cap"


def test_bad_requests(client):
    assert client.post("/api/agent/chat", json={"message": ""}).status_code == 400
    assert client.post("/api/agent/chat", json={"message": "x", "language": "cobol"}).status_code == 400
    assert client.post("/api/agent/code", json={"plan": {}, "language": "java"}).status_code == 400


# ---------------------------------------------------------------- review fixes (decode: retries, truncation)

def test_failed_run_still_counts_toward_the_caps(client, agent_on):
    # a model that keeps calling a tool with no arguments: after decode's tool-retry budget the run fails
    agent_on(*[ModelResponse(parts=[ToolCallPart("search_endpoints", {})], model_name="claude-sonnet-5-5")] * 10)
    r = client.post("/api/agent/chat", json={"message": "x"})
    assert r.status_code == 502 and r.json()["error"] == "agent_error"
    assert app.state.agent_guard.status()["today_usd"] > 0


def test_big_schemas_are_compacted_and_long_lists_say_so(client):
    big = code_agent.get_body_schema(ctx(), "DEPBT0711MtrdDlOnHld_POSTUsingPOST")  # ~107 KB raw
    assert len(json.dumps(big)) <= code_agent.MAX_TOOL_CHARS + 500
    listed = code_agent.list_endpoints(ctx(), "TradeFinance")
    assert listed["shown"] == 80 and listed["total"] > 80


# ---------------------------------------------------------------- milestone 2: the streaming route

def sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines.get("event"), json.loads(lines.get("data", "{}"))))
    return events


def test_stream_sends_steps_then_the_answer(client, agent_on):
    agent_on(
        ModelResponse(parts=[ToolCallPart("search_endpoints", {"query": "block funds"})], model_name="claude-sonnet-5-5"),
        ModelResponse(parts=[ToolCallPart("get_endpoint", {"name": CASH_BLOCK})], model_name="claude-sonnet-5-5"),
        as_output("Plan", plan()),
    )
    r = client.post("/api/agent/stream", json={"message": "put a hold on funds"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert [e for e, _ in events] == ["step", "step", "answer"]
    assert events[0][1] == {"tool": "search_endpoints", "query": "block funds"}
    assert events[2][1]["answer"]["steps"][0]["endpoint"] == CASH_BLOCK and events[2][1]["usage"]["usd"] > 0
    assert app.state.agent_guard.status()["answers_today"] == 1


def test_stream_reports_errors_as_an_event_and_still_prices_them(client, agent_on):
    agent_on(*[ModelResponse(parts=[ToolCallPart("search_endpoints", {})], model_name="claude-sonnet-5-5")] * 10)
    events = sse_events(client.post("/api/agent/stream", json={"message": "x"}).text)
    assert events[-1][0] == "error" and events[-1][1]["error"] == "agent_error"
    assert app.state.agent_guard.status()["today_usd"] > 0


def test_stream_checks_availability_and_caps_before_streaming(client, monkeypatch, tmp_path):
    monkeypatch.setattr(code_agent, "available", lambda: False)
    assert client.post("/api/agent/stream", json={"message": "x"}).status_code == 503
    monkeypatch.setattr(code_agent, "available", lambda: True)
    monkeypatch.setattr(app.state, "agent_guard", AgentGuard(tmp_path, daily_usd=0))
    r = client.post("/api/agent/stream", json={"message": "x"})
    assert r.status_code == 429 and r.json()["error"] == "daily_cap"
