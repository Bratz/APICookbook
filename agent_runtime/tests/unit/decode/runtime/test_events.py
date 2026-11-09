"""The ``--events-jsonl`` step log (APICookbook fork F2b), through a real agent run offline."""

from __future__ import annotations

import json

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.test import TestModel

from decode.runtime.events import MAX_OUTPUT_CHARS, event_record, event_writer


def test_tool_calls_and_results_become_records_and_text_is_left_out():
    call = event_record(
        FunctionToolCallEvent(ToolCallPart("bash", {"command": "ls"}, tool_call_id="c1"))
    )
    result = event_record(
        FunctionToolResultEvent(ToolReturnPart("bash", "x" * 9000, tool_call_id="c1"))
    )

    assert call == {"type": "tool_call", "id": "c1", "tool": "bash", "args": '{"command":"ls"}'}
    assert result["type"] == "tool_result" and result["ok"] is True and result["id"] == "c1"
    assert len(result["output"]) == MAX_OUTPUT_CHARS  # a whole file never lands in the log
    assert event_record(PartStartEvent(index=0, part=TextPart("thinking"))) is None


async def test_a_run_appends_one_line_per_step(tmp_path):
    agent = Agent(TestModel())

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    log = tmp_path / "run" / "events.jsonl"
    await agent.run("add", event_stream_handler=event_writer(log))

    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["tool_call", "tool_result"]
    assert lines[0]["tool"] == lines[1]["tool"] == "add"
    assert lines[1]["ok"] is True
    assert all(isinstance(line["at"], float) for line in lines)
