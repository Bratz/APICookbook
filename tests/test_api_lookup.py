"""backend/api_lookup.py: the Swagger lookups behind Build & run's proxy, the endpoint cards and the eval."""

import json

import pytest

from backend import api_lookup

CASH_BLOCK = "AMBTCrtCashBlockWrapperUsingPOST"


def test_search_returns_swagger_ops_with_scores_and_no_cutoff(client):
    kb, rag = client.app.state.kb, client.app.state.rag_engine
    if not rag:
        pytest.skip("RAG_ENABLED is off")
    hits = api_lookup.search_endpoints(kb, rag, "cancel a cash block")
    assert "AMBTCancelCashBlockWrapperUsingDELETE" in [h["endpoint"] for h in hits]
    assert all(kb.endpoints[h["endpoint"]]["source"] == "swagger" for h in hits)
    # the search chat's 0.62 cut-off would return nothing here; an agent still gets candidates to judge
    assert len(api_lookup.search_endpoints(kb, rag, "zzqx nonsense")) == api_lookup.SEARCH_RESULTS


def test_get_endpoint_hides_transport_headers_and_postman_entries(client):
    kb = client.app.state.kb
    info = api_lookup.get_endpoint(kb, CASH_BLOCK)
    assert info["path"] == "/accountManagement/account/cashBlock" and info["has_body"] is True
    assert info["url_template"].endswith("/Core/accountManagement/account/cashBlock")
    assert not set(info["parameters"]) & api_lookup._common(kb)
    assert "error" in api_lookup.get_endpoint(kb, "GetAccountBalance")  # a Postman entry


def test_get_body_schema(client):
    kb = client.app.state.kb
    schema = api_lookup.get_body_schema(kb, CASH_BLOCK)
    assert "createCashBlockResource" in schema["definitions"]
    assert api_lookup.get_body_schema(kb, "CBPETGetAccountBalanceUsingGET") == "This operation takes no request body."


def test_list_endpoints_by_area(client):
    kb = client.app.state.kb
    rows = api_lookup.list_endpoints(kb, "AccountManagement")
    assert rows and all(kb.endpoints[r["endpoint"]]["category"] == "AccountManagement" for r in rows)
    assert "areas" in api_lookup.list_endpoints(kb, "NoSuchArea")


def test_big_schemas_are_compacted_and_long_lists_say_so(client):
    kb = client.app.state.kb
    big = api_lookup.get_body_schema(kb, "DEPBT0711MtrdDlOnHld_POSTUsingPOST")  # ~107 KB raw
    assert len(json.dumps(big)) <= api_lookup.MAX_TOOL_CHARS + 500
    listed = api_lookup.list_endpoints(kb, "TradeFinance")
    assert listed["shown"] == 80 and listed["total"] > 80
