"""The MCP endpoint (/api/mcp/): APIWeaver's tools for any MCP client; writes wait for a person."""

import json

import pytest
from fastmcp import Client

from backend import main

WRITE = {"api_name": "TCS_BaNCS", "endpoint_name": "create_customer",
         "parameters": {"customerType": 1, "customerFullName": "Ada Test"}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def weaver(client, monkeypatch):
    """The app's own weaver (built and registered by the lifespan); bank calls are recorded, never sent."""
    sent = []

    async def fake_call(api_name, endpoint_name, params, ctx=None, bank_token=""):
        sent.append((api_name, endpoint_name, params))
        main.MCP_WEAVER.tokens.append(bank_token)
        return {"created": True}

    monkeypatch.setattr(main.MCP_WEAVER, "_execute_api_call", fake_call)
    main.MCP_WEAVER.sent = sent
    main.MCP_WEAVER.tokens = []
    return main.MCP_WEAVER


async def _call(weaver, args):
    async with Client(weaver.mcp) as c:
        return (await c.call_tool("call_api", args)).data


@pytest.mark.anyio
async def test_a_write_is_held_until_a_person_approves_it(weaver, client):
    held = await _call(weaver, WRITE)

    assert held["status"] == "pending_approval" and held["approval_url"].endswith(held["approval_id"])
    assert weaver.sent == []  # nothing reached the bank
    page = client.get(f"/api/mcp-writes/{held['approval_id']}").json()
    assert page["method"] == "POST" and page["status"] == "pending" and page["params"]["customerFullName"] == "Ada Test"
    assert (await _call(weaver, {**WRITE, "parameters": {"approval_id": held["approval_id"]}}))["status"] == "pending"

    decided = client.post(f"/api/mcp-writes/{held['approval_id']}", json={"approve": True}).json()

    assert decided["status"] == "approved"
    assert weaver.sent == [("TCS_BaNCS", "create_customer", WRITE["parameters"])]  # the stored request, once
    outcome = await _call(weaver, {**WRITE, "parameters": {"approval_id": held["approval_id"]}})
    assert outcome == {"success": True, "status": "approved", "data": {"created": True}}
    assert client.post(f"/api/mcp-writes/{held['approval_id']}", json={"approve": True}).status_code == 409


@pytest.mark.anyio
async def test_a_rejected_write_is_never_sent(weaver, client):
    held = await _call(weaver, WRITE)

    client.post(f"/api/mcp-writes/{held['approval_id']}", json={"approve": False})

    assert weaver.sent == []
    assert (await _call(weaver, {**WRITE, "parameters": {"approval_id": held["approval_id"]}}))["status"] == "rejected"
    assert client.get("/api/mcp-writes/not-a-real-id").status_code == 404


def test_any_mcp_client_can_list_the_tools_over_http(client):
    accept = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    init = client.post("/api/mcp/", headers=accept, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
    assert init.status_code == 200
    session = {**accept, "mcp-session-id": init.headers["mcp-session-id"]}
    client.post("/api/mcp/", headers=session, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    listed = client.post("/api/mcp/", headers=session, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    data = next(json.loads(line[6:]) for line in listed.text.splitlines() if line.startswith("data: "))
    assert {t["name"] for t in data["result"]["tools"]} == {"list_apis", "call_api", "get_api_schema", "search_endpoints"}


def test_the_endpoint_answers_without_a_trailing_slash_and_without_a_redirect(client):
    # Vercel's /api/:path* rewrite drops the slash: /api/mcp itself must be the MCP endpoint.
    init = client.post("/api/mcp", follow_redirects=False,
                       headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
                       json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "test", "version": "1"}}})

    assert init.status_code == 200 and init.headers.get("mcp-session-id")


@pytest.mark.anyio
async def test_call_api_needs_the_callers_bank_token_and_passes_it_on(weaver, monkeypatch):
    from backend.apiweaver import BANK_TOKEN
    monkeypatch.setattr(main.settings, "bancs_token_required", True)
    read = {"api_name": "TCS_BaNCS", "endpoint_name": "get_account_balance",
            "parameters": {"accountReference": "101000000101814"}}

    refused = await _call(weaver, read)
    assert refused["status"] == "token_required" and weaver.sent == []

    BANK_TOKEN.set("visitor-token-1")  # what the Playground does; an MCP client sends the Accesstoken header
    assert (await _call(weaver, read))["success"] is True
    assert weaver.tokens == ["visitor-token-1"]

    held = await _call(weaver, WRITE)  # a held write keeps the token for when it is approved ...
    page = weaver.held_write(held["approval_id"])
    assert "visitor-token-1" not in str(page)  # ... and the approval page never shows it
    await weaver.decide_write(held["approval_id"], True)
    assert weaver.tokens[-1] == "visitor-token-1"
