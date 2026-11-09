"""F5's scoring: gold sequences, mapping real calls to Swagger operations, the report. No model calls."""

from backend import agent_eval
from backend.agent_eval import TASKS, OperationMatcher, Result, body_valid, in_order


def test_every_gold_operation_exists_in_the_spec(client):
    kb = client.app.state.kb
    missing = sorted(op for _, _, gold in TASKS for step in gold for op in step
                     if (kb.endpoints.get(op) or {}).get("source") != "swagger")
    assert missing == []  # a typo here would make a task unwinnable


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


def test_body_checks():
    schema = {"type": "object", "required": ["customerId"], "properties": {"customerId": {"type": "integer"}}}
    assert body_valid(schema, '{"customerId": 7}') is True
    assert body_valid(schema, '{"customerId": "seven"}') is False
    assert body_valid(schema, "not json") is False
    assert body_valid(None, "{}") is None


def test_the_report_summarises_and_lists_every_task():
    results = [
        Result(agent="build", task=1, kind="read", gold_hit=True, ok=True, gold_get_2xx=True, usd=0.1, seconds=40),
        Result(agent="build", task=12, kind="write", gold_hit=False, ok=True, write_bodies_valid=False, usd=0.2,
               note="run ended: failed"),
    ]

    text = agent_eval.report(results)

    assert "| Gold sequence hit, all tasks | 1/2 |" in text
    assert "| … write tasks | 0/1 |" in text
    assert "| A gold GET answered 2xx (read tasks) | 1/1 |" in text
    assert "| Held write bodies valid against the schema | 0/1 |" in text
    assert "❌ run ended: failed" in text
    assert "$0.150/task ($0.30)" in text


def test_an_api_question_passes_only_by_naming_a_gold_operation():
    endpoints = {"RSTBTAmendPhonenumberUsingPUT": {"url": "/customerManagement/customer/contact"}}
    gold = [{"RSTBTAmendPhonenumberUsingPUT"}]

    assert agent_eval.names_an_operation("Use RSTBTAmendPhonenumberUsingPUT with a body…", gold, endpoints)
    assert agent_eval.names_an_operation("PUT /customerManagement/customer/contact takes…", gold, endpoints)
    assert not agent_eval.names_an_operation("Use the contact API.", gold, endpoints)
    assert not agent_eval.names_an_operation("anything", [], endpoints)  # off-topic tasks have no gold


def test_the_report_counts_api_questions_and_off_topic_separately():
    text = agent_eval.report([
        Result(agent="build", task=21, kind="spec", gold_hit=True, ok=True),
        Result(agent="build", task=25, kind="offtopic", gold_hit=False, ok=True, note="used tools"),
    ])

    assert "| … API questions (answered from the spec, no bank calls) | 1/1 |" in text
    assert "| … off-topic (declined, no tools) | 0/1 |" in text
    assert "| 25 | offtopic | Write a quicksort function in Go. | ❌ used tools |" in text
