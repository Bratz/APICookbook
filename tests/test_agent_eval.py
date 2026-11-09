"""F5's scoring: gold sequences, mapping real calls to Swagger operations, the report. No model calls."""

from backend import agent_eval
from backend.agent_eval import TASKS, OperationMatcher, Result, body_valid, in_order, python_parses


def test_every_gold_operation_exists_in_the_spec(client):
    kb = client.app.state.kb
    missing = sorted(op for _, _, gold in TASKS for step in gold for op in step
                     if (kb.endpoints.get(op) or {}).get("source") != "swagger")
    assert missing == []  # a typo here would make a task unwinnable for both agents


def test_gold_is_matched_in_order_with_extra_calls_allowed():
    gold = [{"A", "A2"}, {"B"}]

    assert in_order([{"x"}, {"A2"}, {"y"}, {"B"}], gold)  # exploration around the gold steps is fine
    assert not in_order([{"B"}, {"A"}], gold)  # wrong order
    assert not in_order([{"A"}], gold)  # a step missing
    assert in_order([{"A", "B"}, {"B"}], gold)  # one call may be several operations


def test_real_calls_map_to_their_swagger_operations(client):
    matcher = OperationMatcher(client.app.state.kb.endpoints)

    assert "RSTETViewCustomerDetailsUsingGET" in matcher.operations(
        "GET", "/Core/customerManagement/customer/viewDetails?CustomerID=135215")
    assert "AMET0117FtchBlckLstUsingGET" in matcher.operations(
        "GET", "/Core/accountManagement/account/blockList/0101000000102")  # a {param} segment
    assert "RSTBTAmendPhonenumberUsingPUT" in matcher.operations("PUT", "/Core/customerManagement/customer/contact")
    assert matcher.operations("DELETE", "/Core/customerManagement/customer/contact") == set()  # method matters
    assert matcher.operations("GET", "/Core/accountManagement/account/blockList/1/extra") == set()


def test_code_and_body_checks():
    assert python_parses("import requests\nprint(1)\n")
    assert not python_parses("def broken(:\n")
    schema = {"type": "object", "required": ["customerId"], "properties": {"customerId": {"type": "integer"}}}
    assert body_valid(schema, '{"customerId": 7}') is True
    assert body_valid(schema, '{"customerId": "seven"}') is False
    assert body_valid(schema, "not json") is False
    assert body_valid(None, "{}") is None


def test_the_report_puts_both_agents_side_by_side():
    results = [
        Result(agent="code", task=1, kind="read", gold_hit=True, ok=True, usd=0.02, seconds=9),
        Result(agent="build", task=1, kind="read", gold_hit=True, ok=True, gold_get_2xx=True, usd=0.1, seconds=40),
        Result(agent="code", task=12, kind="write", gold_hit=False, ok=False, note="asked a clarifying question"),
        Result(agent="build", task=12, kind="write", gold_hit=True, ok=True, write_bodies_valid=False, usd=0.2),
    ]

    text = agent_eval.report(results)

    assert "| Gold sequence hit, all tasks | 1/2 | 2/2 |" in text
    assert "| … write tasks | 0/1 | 1/1 |" in text
    assert "| A gold GET answered 2xx (read tasks) | - | 1/1 |" in text
    assert "| Held write bodies valid against the schema | - | 0/1 |" in text
    assert "❌ asked a clarifying question" in text
