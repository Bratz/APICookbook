# tests/test_integration.py
"""
Integration tests for the TCS BaNCS API Cookbook.
Runs the real app (lifespan included) in-process; no outbound network calls.
"""

import json
import re
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.main import app, settings  # conftest.py sets cwd and provides the `client` fixture


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


def test_openapi_export(client):
    spec = client.get("/api/openapi").json()
    assert spec["openapi"].startswith("3.")
    assert len(spec["paths"]) > 0


def test_mcp_tools(client):
    data = client.get("/api/mcp/tools").json()
    assert data["enabled"] == settings.mcp_enabled
    if settings.mcp_enabled:
        names = {t["name"] for t in data["tools"]}
        assert {"call_api", "get_api_schema", "search_endpoints", "list_apis"} <= names
        # Each listed tool is one the server knows: calling it never says "Unknown tool".
        for name in names:
            r = client.post("/api/playground/execute", json={"type": "mcp", "tool_name": name, "tool_params": {}})
            assert "Unknown tool" not in r.text, name


# ===================== Knowledge base =====================

def test_knowledge_search(client):
    data = client.post("/api/knowledge/search", json={"query": "account balance", "limit": 3}).json()
    assert 0 < data["total"] <= 3
    assert any("balance" in r["endpoint_name"].lower() for r in data["results"])


def test_knowledge_search_requires_query(client):
    assert client.post("/api/knowledge/search", json={}).status_code == 400


def test_knowledge_add_and_delete(client, monkeypatch):
    monkeypatch.setenv("RUNNER_TOKEN", "op-token-0123456789abcdef0123456789abcdef")
    op = {"Authorization": "Bearer op-token-0123456789abcdef0123456789abcdef"}
    body = {"endpoint_name": "PytestEP", "data": {"method": "GET", "url": "/pytest", "description": "tmp"}}
    assert client.post("/api/knowledge/add", json=body, headers=op).status_code == 200
    assert any(e["name"] == "PytestEP" for e in client.get("/api/endpoints").json())
    assert client.delete("/api/knowledge/PytestEP", headers=op).status_code == 200
    assert client.delete("/api/knowledge/PytestEP", headers=op).status_code == 404


SHARED_STATE_ROUTES = [("POST", "/api/knowledge/add"), ("DELETE", "/api/knowledge/GetAccountBalance"),
                       ("POST", "/api/developer/reload-knowledge"), ("POST", "/api/developer/test-endpoint?endpoint_name=GetAccountBalance"),
                       ("POST", "/api/hoppscotch/import")]


@pytest.mark.parametrize("token", ["", "op-token-0123456789abcdef0123456789abcdef"])
def test_routes_that_change_shared_state_are_operator_only(client, monkeypatch, token):
    # No token configured (""): even a bare "Bearer " must not pass. Token configured: a wrong one must not.
    monkeypatch.setenv("RUNNER_TOKEN", token)
    for method, path in SHARED_STATE_ROUTES:
        for auth in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer wrong"}):
            assert client.request(method, path, headers=auth).status_code == 403, (method, path, auth)
    assert any(e["name"] == "GetAccountBalance" for e in client.get("/api/endpoints").json())


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


# 30 developer questions, each with every endpoint that answers it (Postman and Swagger duplicates).
# Measured: 0.53 top-3 with the old document text, 0.80 with the current one (rag_engine._create_api_document).
RETRIEVAL_CASES = [
    ("check account balance", {"GetAccountBalance", "CBPETGetAccountBalanceUsingGET"}),
    ("how much money is in this account right now", {"GetAccountBalance", "CBPETGetAccountBalanceUsingGET"}),
    ("list all accounts of a customer with their balances", {"CBCET0101FetchCustAcntsUsingGET", "AMET0395GetAcntUsrBlncsUsingGET", "GetAccountListByCustomerID"}),
    ("onboard a new customer", {"CreateCustomer", "CustomerCreate", "RSTBTQuickCustomerOnboardingWthMnrUsingPOST"}),
    ("I need to open a savings account for an existing customer", {"AMBT2020CreateAcntActvUsingPOST", "RSTBTCreateAcntActv_POSTUsingPOST", "AccountCreate", "RSTBT2020CreateAcntActvwthRelationsUsingPOST"}),
    ("update customer mobile number", {"RSTBTAmendPhonenumberUsingPUT", "BRBT0303ModContactDtls_POSTUsingPUT"}),
    ("the customer moved house, change their postal address", {"RSTBTAmendPostalAddressUsingPUT", "BRBTCreateModifyCommDtlsUsingPUT"}),
    ("transfer money between two accounts", {"AMEBTFundTransferWrapperIB_POSTUsingPOST", "AMEBT2021AmeWrapper_POSTUsingPOST"}),
    ("approve a pending fund transfer", {"AMEBT1603AuthAMETxn_POSTUsingPOST"}),
    ("list fund transfers", {"AMEET1612FetchAMETxnList_POSTUsingGET_1"}),
    ("put a hold on some funds so they cannot be withdrawn", {"AMBTCrtCashBlockWrapperUsingPOST", "CashBlock", "INTBT0118CrtCshBlckIntf_POSTUsingPOST"}),
    ("cancel a cash block", {"AMBTCancelCashBlockWrapperUsingDELETE", "ReleaseCancelCashBlock"}),
    ("see all blocks placed on an account", {"AMET0117FtchBlckLstUsingGET"}),
    ("loan repayment schedule", {"CRET2811GetRpymntDtls_POSTUsingGET", "GetLoansRepaymentSchedule"}),
    ("pay out the loan amount to the borrower", {"CRBT4211DownDisbLOSMasterWrapper_POSTUsingPOST", "CRBT1751CreateDisburReqUsingPOST"}),
    ("how much does the customer still owe to close the loan", {"CRET6827FetchLoanWriteoffDtlsUsingGET", "GetLoanDischargeEnquiry"}),
    ("repay a loan by transfer from an account", {"CRBT0315InitRpymnt_POSTUsingPOST"}),
    ("request an account statement", {"BRBTRequestAcntStatmntUsingPOST", "AMET2021AcntsBalStatmnt_POSTUsingPOST"}),
    ("order a new cheque book", {"CBCBTCreateChequeBookRequestUsingPOST", "RSBTCreateChqRequestOnRlsd_POSTUsingPOST"}),
    ("stop payment on a cheque", {"RSTBT11810StopPymntChqUsingPOST", "RSBTCreateStopChequeRequest_POSTUsingPOST"}),
    ("the customer wants to close their account", {"AMTO5202CrtAcntClsReleased_POSTUsingPOST", "RSTBT2010AcntClsrRls_POSTUsingPOST", "AMTO5201ValidateAcntClsDtls_POSTUsingGET"}),
    ("open a fixed deposit", {"DEPBTOpenTDWrapperIB_POSTUsingPOST"}),
    ("break a term deposit before it matures", {"DEPBT0409IBCreateDealErlMtrty_POSTUsingPOST"}),
    ("current currency exchange rate", {"CBCET1401GetEXRTUsingGET", "MIET1210ListOthrPrpsEXRTUsingGET"}),
    ("what interest rate applies to a loan product", {"CBCET0333GetLatestINRT_POSTUsingGET"}),
    ("register a house as security for a loan", {"CreateCollateral-RealState", "CRBT1801CrtTenantOwnedCltrlWrapperUsingPOST"}),
    ("account details including the address", {"AMET2020GetAcntDetails_wrapperUsingGET"}),
    ("look up a customer's profile", {"GetCustomerDetails", "RSTETViewCustomerDetailsUsingGET", "CBPET0106GetBPDetailsUsingGET"}),
    ("show the transaction history of an account", {"GetAccountTransactions", "CBCET2001CalcRNGBalUsingGET"}),
    ("crypto wallet balance", {"FAET2022FetchCryptoBal_POSTUsingGET"}),
]


def test_retrieval_quality_on_developer_questions(client):
    rag = app.state.rag_engine
    if not rag:
        pytest.skip("RAG_ENABLED is off or sentence-transformers is not installed")
    hits = []
    for query, gold in RETRIEVAL_CASES:
        top3 = {doc.metadata["endpoint_name"] for doc, _ in rag.search(query, top_k=3)}
        hits.append(bool(top3 & gold))
    missed = [q for (q, _), ok in zip(RETRIEVAL_CASES, hits) if not ok]
    assert sum(hits) / len(hits) >= 0.75, f"top-3 hit rate {sum(hits)}/{len(hits)}; missed: {missed}"


@pytest.mark.parametrize("query", ["what is the weather in paris", "recipe for chocolate pancakes", "tell me a joke"])
def test_off_topic_questions_get_no_semantic_match(client, query):
    if not app.state.rag_engine:
        pytest.skip("RAG_ENABLED is off or sentence-transformers is not installed")
    assert app.state.rag_engine.search(query, top_k=3) == []


@pytest.mark.parametrize("url", [
    "/accountManagement/account/cashBlock",
    "https://demoapps.tcsbancs.com/Core/accountManagement/account/cashBlock",  # full URL with the basePath
])
def test_body_schema_resolves_for_playground_editor(client, url):
    r = client.get("/api/body-schema", params={"method": "POST", "url": url})
    assert r.status_code == 200
    data = r.json()
    assert data["endpoint"] == "AMBTCrtCashBlockWrapperUsingPOST"
    defs = data["schema"]["definitions"]
    assert data["schema"]["$ref"] == "#/definitions/createCashBlockResource" and "createCashBlockResource" in defs
    # every $ref the schema uses is inlined, so the editor needs nothing else
    refs = re.findall(r'"\$ref": "#/definitions/([^"]+)"', json.dumps(data["schema"]))
    assert set(refs) <= set(defs)


def test_body_schema_matches_path_parameters_and_misses_cleanly(client):
    r = client.get("/api/body-schema", params={"method": "PUT", "url": "/accountManagement/account/cashblocks/123/update"})
    assert r.status_code == 200 and r.json()["endpoint"] == "AMBT0119UpdCashBlckIntfUsingPUT"
    assert client.get("/api/body-schema", params={"method": "GET", "url": "/Core/accountManagement/account/balanceDetails/1"}).status_code == 404
    assert client.get("/api/body-schema", params={"method": "POST", "url": "/no/such/path"}).status_code == 404


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
    assert kwargs["headers"]["Accept"] == "application/json"  # the bank answers XML otherwise


def test_playground_needs_the_visitors_bank_token_and_forwards_it(client, monkeypatch):
    monkeypatch.setattr(settings, "bancs_token_required", True)
    call = {"type": "api", "method": "GET", "url": "/Core/accountManagement/x"}
    fake = httpx.Response(200, json={}, request=httpx.Request("GET", settings.api_base_url))
    with patch.object(httpx.AsyncClient, "request", new=AsyncMock(return_value=fake)) as sent:
        assert client.post("/api/playground/execute", json=call).status_code == 401
        sent.assert_not_called()
        client.post("/api/playground/execute", json=call, headers={"Accesstoken": "visitor-token-1"})
        assert sent.call_args.kwargs["headers"]["Accesstoken"] == "visitor-token-1"
        # An Accesstoken typed into the Playground's header rows wins.
        client.post("/api/playground/execute", json={**call, "headers": {"Accesstoken": "typed"}},
                    headers={"Accesstoken": "visitor-token-1"})
        assert sent.call_args.kwargs["headers"]["Accesstoken"] == "typed"


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


def test_chat_code_example_uses_core_base_path_and_schema_fields(client):
    import ast
    response = client.post("/api/chat", json={"message": "how to update the phone numbers"}).json()["response"]
    code = response.split("```python\n", 1)[1].split("```", 1)[0]
    ast.parse(code)  # the example must be valid Python
    assert f"{settings.api_base_url}/Core/" in code  # Swagger paths sit under the spec's basePath
    assert "Add your request data here" not in code  # the body comes from the endpoint's schema
    assert "Fields from the API spec" in code


@pytest.mark.parametrize("path", ["/api/cookbook", "/api/knowledge/endpoints", "/api/knowledge/search/balance",
                                  "/api/hoppscotch/collections", "/api/playground/history",
                                  "/api/developer/bert-status", "/api/bert-chat/intents"])
def test_removed_routes_stay_gone(client, path):
    assert client.get(path).status_code in (404, 405)


def test_playground_adds_the_core_base_path_and_refuses_unfilled_placeholders(client):
    fake = httpx.Response(200, json={}, request=httpx.Request("GET", settings.api_base_url))
    with patch.object(httpx.AsyncClient, "request", new=AsyncMock(return_value=fake)) as sent:
        client.post("/api/playground/execute",
                    json={"type": "api", "method": "GET", "url": "/accountManagement/account/balanceDetails/1"})
        assert str(sent.call_args.kwargs["url"]).endswith("/Core/accountManagement/account/balanceDetails/1")
        sent.reset_mock()
        r = client.post("/api/playground/execute",
                        json={"type": "api", "method": "GET", "url": "/Core/x/balanceDetails/{accountReference}"})
        assert r.status_code == 400 and "{accountReference}" in r.json()["error"]
        sent.assert_not_called()
