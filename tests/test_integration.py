# tests/test_integration.py
"""
Integration tests for the TCS BaNCS API Cookbook.
Runs the real app (lifespan included) in-process; no outbound network calls.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

# The app resolves api_specs/, collections/, frontend/ and logs/ relative to cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, settings  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with` runs the lifespan, which loads the knowledge base and APIWeaver
    with TestClient(app) as c:
        yield c


# ===================== Core =====================

def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "BaNCS API Cookbook" in r.text


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["fallback_mode"] is False
    assert data["stats"]["endpoints_loaded"] > 0


def test_list_endpoints(client):
    eps = client.get("/api/endpoints").json()
    assert len(eps) > 0
    assert {"name", "method", "path"} <= eps[0].keys()


def test_cookbook(client):
    recipes = client.get("/api/cookbook").json()
    assert len(recipes) > 0
    assert client.get(f"/api/cookbook/{recipes[0]['id']}").status_code == 200
    assert client.get("/api/cookbook/does-not-exist").status_code == 404


def test_openapi_export(client):
    spec = client.get("/api/openapi").json()
    assert spec["openapi"].startswith("3.")
    assert len(spec["paths"]) > 0


def test_mcp_tools(client):
    data = client.get("/api/mcp/tools").json()
    assert data["enabled"] == settings.mcp_enabled
    if settings.mcp_enabled:
        assert data["total"] > 0
        assert any(t["name"].endswith("get_account_balance") for t in data["tools"])


def test_hoppscotch_export(client):
    data = client.get("/api/hoppscotch/collections").json()
    assert data["total"] > 0 and data["collections"]


# ===================== Knowledge base =====================

def test_knowledge_search(client):
    data = client.post("/api/knowledge/search", json={"query": "account balance", "limit": 3}).json()
    assert 0 < data["total"] <= 3
    assert any("balance" in r["endpoint_name"].lower() for r in data["results"])


def test_knowledge_search_requires_query(client):
    assert client.post("/api/knowledge/search", json={}).status_code == 400


def test_knowledge_add_and_delete(client):
    body = {"endpoint_name": "PytestEP", "data": {"method": "GET", "url": "/pytest", "description": "tmp"}}
    assert client.post("/api/knowledge/add", json=body).status_code == 200
    assert any(e["name"] == "PytestEP" for e in client.get("/api/knowledge/endpoints").json()["endpoints"])
    assert client.delete("/api/knowledge/PytestEP").status_code == 200
    assert client.delete("/api/knowledge/PytestEP").status_code == 404


def test_semantic_search_finds_by_meaning(client):
    if not app.state.rag_engine:
        pytest.skip("RAG_ENABLED is off or sentence-transformers is not installed")
    data = client.post("/api/knowledge/search", json={"query": "loan repayment schedule", "limit": 3}).json()
    assert data["source"] == "rag"
    assert data["results"][0]["endpoint_name"] == "GetLoansRepaymentSchedule"
    assert data["results"][0]["path"] != "/"  # metadata filled from the knowledge base


def test_semantic_search_low_confidence_falls_back_to_keyword(client):
    if not app.state.rag_engine:
        pytest.skip("RAG_ENABLED is off or sentence-transformers is not installed")
    data = client.post("/api/knowledge/search", json={"query": "zzqx nonsense"}).json()
    assert data["source"] == "keyword" and data["total"] == 0


def test_kb_fuzzy_search():
    from backend.knowledge_base import UnifiedAPIKnowledgeBase
    results = UnifiedAPIKnowledgeBase().search_endpoints("GetAccountBalance")
    assert results and results[0][0] == "GetAccountBalance"


# ===================== Chat =====================

@pytest.mark.parametrize("message, expected", [
    ("hi", "Hello"),
    ("show all apis", "API endpoints"),
    ("customer details", "relevant API endpoints"),
    ("python code example for account balance", "```python"),
    ("what mcp tools are available", "MCP"),
    ("", "provide a message"),
])
def test_chat(client, message, expected):
    data = client.post("/api/chat", json={"message": message}).json()
    assert expected in data["response"]


def test_chat_uses_semantic_search(client):
    if not app.state.rag_engine:
        pytest.skip("RAG_ENABLED is off or sentence-transformers is not installed")
    data = client.post("/api/chat", json={"message": "update customer mobile number"}).json()
    assert data["metadata"]["source"] == "rag"
    assert "AmendPhonenumber" in data["api_references"][0]["name"]  # keyword search ranks CreateCustomer first


def test_chat_websocket(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "system"
        ws.send_json({"message": "hi"})
        assert "Hello" in ws.receive_json()["data"]["response"]


# ===================== Security regressions =====================

@pytest.mark.parametrize("url", [
    "https://evil.example.com/steal",
    "http://169.254.169.254/latest/meta-data/",
    "https://demoapps.tcsbancs.com.evil.example.com/",
    "https://demoapps.tcsbancs.com@evil.example.com/",
    "http://demoapps.tcsbancs.com/",           # scheme downgrade
    "https://demoapps.tcsbancs.com:8443/",     # other port
])
def test_playground_blocks_foreign_hosts(client, url):
    with patch.object(httpx.AsyncClient, "request", new=AsyncMock()) as sent:
        r = client.post("/api/playground/execute", json={"type": "api", "method": "GET", "url": url})
    assert r.status_code == 400
    sent.assert_not_called()  # auth headers never leave the process


def test_playground_allows_bancs_host(client):
    fake = httpx.Response(200, json={"balance": 1000},
                          request=httpx.Request("GET", settings.api_base_url))
    with patch.object(httpx.AsyncClient, "request", new=AsyncMock(return_value=fake)) as sent:
        r = client.post("/api/playground/execute",
                        json={"type": "api", "method": "GET", "url": "/Core/accountManagement/x"})
    assert r.status_code == 200 and r.json()["result"]["body"] == {"balance": 1000}
    kwargs = sent.call_args.kwargs
    assert kwargs["url"] == f"{settings.api_base_url}/Core/accountManagement/x"
    assert kwargs["headers"]["entity"] == settings.bancs_entity


def test_playground_validation(client):
    assert client.post("/api/playground/execute", json={"type": "bogus"}).status_code == 422
    assert client.post("/api/playground/execute", json={"type": "api", "url": ""}).status_code == 422


def test_dev_logs_require_debug(client):
    expected = 200 if settings.debug else 403
    assert client.get("/api/developer/logs").status_code == expected


def test_cors_rejects_foreign_origin(client):
    r = client.options("/api/chat", headers={"Origin": "https://evil.example.com",
                                             "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") not in ("*", "https://evil.example.com")
