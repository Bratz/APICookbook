"""Milestone 1 capstone: one scripted five-step conversation through the FULL real stack.

Proves the whole M1 stack together: real build_agent (flat tool registry, deferred-tool /
permission seam, memory-instructions hook), real Runner + AgentTurnHandler turn lifecycle,
real render_event on every event, and real SessionLog persist + ``--resume`` replay. Swapped/faked:
the model is a scripted FunctionModel (GEMINI_API_KEY faked so build_agent constructs); the session
log is redirected under tmp_path. (APICookbook fork: the web_fetch step and the MEMORY.md write-back
were removed with those features.)
Fully offline — no network, no API key, no skipif.

The five turns: read (auto-allowed) → write (approved) → write (DENIED, file never lands) →
todo_write → ask_user (ungated, fake answer) — covering all three
permission outcomes: auto-allow, human-allow, human-deny.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import SecretStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from rich.console import Console

from decode import render
from decode.agent.deps import AgentDeps
from decode.agent.factory import build_agent
from decode.agent.loop import AgentTurnHandler
from decode.context import session_log
from decode.context.session_log import SessionLog
from decode.entities import events
from decode.entities.permissions import PermissionDecision, PermissionRequest
from decode.harness.runner import Runner
from decode.permissions.gate import PermissionGate

# --- markers the scripted model streams, so assertions read as a transcript -----------------

_READ_TARGET = "notes.txt"
_READ_CONTENTS = "remember to ship milestone one"
_WRITE_OK_PATH = "created.txt"
_WRITE_OK_BODY = "this file was written by the agent"
_WRITE_DENIED_PATH = "should-not-exist.txt"
_WRITE_DENIED_BODY = "this write is denied and must never hit disk"
_TODO_CONTENT = "wire up the capstone test"
_ASK_QUESTION = "which environment should I target?"
_ASK_ANSWER = "target the staging environment"
_FINAL_TEXT = "all five steps are done"


@pytest.fixture(autouse=True)
def _fake_gemini_key(mocker):
    """Let ``build_agent`` construct the Gemini provider offline (the model is overridden)."""
    mocker.patch(
        "decode.agent.factory.settings.gemini_api_key", SecretStr("test-key"), create=False
    )


def _scripted_model(working_dir: Path) -> FunctionModel:
    """A FunctionModel that walks the five-step capstone script, one tool call per leg.

    The model streams a tool call on each *fresh* leg and plain text on each *resume* leg (after
    a tool returned), advancing a step counter so each turn fires exactly one tool then finishes.
    A turn ends when the model streams ``_FINAL_TEXT`` instead of a tool call. The loop streams
    every model node, so the model must stream (yield), not return.
    """
    state = {"step": 0}

    async def stream_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[object]:
        # A leg that follows a tool return is a *resume* leg: end the turn with plain text so the
        # runner reaches its would-stop boundary. Detect it by a ToolReturnPart in the last
        # request (the framework feeds tool results back as the next request's parts).
        if _last_request_has_tool_return(messages):
            yield _FINAL_TEXT
            return

        step = state["step"]
        state["step"] += 1
        yield {0: _CALL_FOR_STEP[step](working_dir)}

    return FunctionModel(stream_function=stream_function)


def _last_request_has_tool_return(messages: list[ModelMessage]) -> bool:
    """True when the most recent request carries a tool result (i.e. this is a resume leg)."""
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            return any(isinstance(part, ToolReturnPart) for part in message.parts)
    return False


# Each step's tool call, as a DeltaToolCall the model streams. Index = turn number (0-based).
_CALL_FOR_STEP = [
    lambda _wd: DeltaToolCall(name="read", json_args=json.dumps({"path": _READ_TARGET})),
    lambda _wd: DeltaToolCall(
        name="write", json_args=json.dumps({"path": _WRITE_OK_PATH, "content": _WRITE_OK_BODY})
    ),
    lambda _wd: DeltaToolCall(
        name="write",
        json_args=json.dumps({"path": _WRITE_DENIED_PATH, "content": _WRITE_DENIED_BODY}),
    ),
    lambda _wd: DeltaToolCall(
        name="todo_write",
        json_args=json.dumps(
            {"tasks": [{"id": "1", "content": _TODO_CONTENT, "status": "in_progress"}]}
        ),
    ),
    lambda _wd: DeltaToolCall(name="ask_user", json_args=json.dumps({"question": _ASK_QUESTION})),
]


class _ScriptedResolvers:
    """The scripted human: an approve/deny verdict per *asked* call, plus a fixed ask_user answer.

    ``permission_verdicts`` is consumed in order — one per call the gate routes to the human
    (under ``default`` mode that is only the two mutating ``write`` calls; the read-only tools
    auto-allow and never reach the resolver, and ``ask_user`` is ungated). Each is a real
    :class:`PermissionDecision`. ``ask_user_answer`` is the line a human would type; the resolver
    returns it straight back as the tool result.
    """

    def __init__(self, *, permission_verdicts: list[PermissionDecision], ask_user_answer: str):
        self._verdicts = list(permission_verdicts)
        self._answer = ask_user_answer
        self.permission_requests: list[PermissionRequest] = []
        self.questions_asked: list[str] = []

    async def resolve_permission(self, request: PermissionRequest) -> PermissionDecision:
        self.permission_requests.append(request)
        return self._verdicts.pop(0)

    async def resolve_user_question(self, question: str) -> str:
        self.questions_asked.append(question)
        return self._answer


async def _run_turn(runner: Runner, prompt: str) -> None:
    """Submit one prompt and drive the runner to idle (one whole turn)."""
    from decode.harness.queue import InputIntent

    await runner.submit(prompt, InputIntent.STEER)
    await runner.wait_idle()


def _tool_returns(history: list[ModelMessage]) -> list[str]:
    """Every tool-return content string in the conversation (what tools fed back to the model)."""
    return [
        str(part.content)
        for message in history
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


async def test_milestone1_capstone_full_stack(tmp_path, monkeypatch):
    """Drive the five-step conversation through the real stack, then assert the M1 guarantees."""
    # --- arrange: a real working tree under tmp, the session log + memory both redirected there
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    (working_dir / _READ_TARGET).write_text(_READ_CONTENTS, encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    # Pin both clocks so the session-log filename and the dated MEMORY.md line are deterministic.
    fixed_now = datetime(2026, 6, 20, 9, 0, tzinfo=UTC)
    monkeypatch.setattr(session_log, "_utc_now", lambda: fixed_now)

    # Every event the turn emits goes through the REAL renderer into a Rich buffer. If any event
    # kind were unhandled, render_event would raise and fail the turn — so this proves the whole
    # render path end to end (not just that the turn ran). The buffer is captured so the test can
    # also assert the rendered transcript carries the surfaced question / task line.
    render_buffer = io.StringIO()
    console = Console(file=render_buffer, force_terminal=False, width=100)

    def on_event(event: events.Event) -> None:
        console.print(render.render_event(event))

    # Under ``default`` mode only the two mutating ``write`` calls reach the human: approve
    # step 2 (write-ok), DENY step 3 (write-denied). The read-only tools (read / todo_write)
    # auto-allow and consume no verdict; ask_user is ungated — so two verdicts only.
    resolvers = _ScriptedResolvers(
        permission_verdicts=[
            PermissionDecision.allow(),  # write (created.txt)
            PermissionDecision.deny(reason="the user denied this write"),  # write (denied)
        ],
        ask_user_answer=_ASK_ANSWER,
    )

    agent = build_agent()
    deps = AgentDeps(
        cwd=working_dir,
        emit=on_event,
        gate=PermissionGate(),
        resolve_permission=resolvers.resolve_permission,
        resolve_user_question=resolvers.resolve_user_question,
    )
    log = SessionLog.create(
        sessions_dir,
        cwd=working_dir,
        now=datetime(2026, 6, 20, 9, 0, tzinfo=UTC),
        session_id=UUID("00000000-0000-0000-0000-00000000ca15"),
    )
    handler = AgentTurnHandler(agent, deps=deps, session_log=log)
    runner = Runner(handler, on_event=on_event)

    # --- act: walk the scripted conversation, one turn per step, through the real model override
    with agent.override(model=_scripted_model(working_dir)):
        await _run_turn(runner, "read the notes file")
        await _run_turn(runner, "write the created file")
        await _run_turn(runner, "write the file that should be denied")
        await _run_turn(runner, "update the task list")
        await _run_turn(runner, "ask me which environment")

    # --- assert: each step did what it should have -----------------------------------------
    returns = _tool_returns(handler.message_history)
    transcript = "\n".join(returns)

    # 1. read returned the file's numbered contents.
    assert any(_READ_CONTENTS in r for r in returns), "read must return the file contents"

    # 2. the approved write actually created the file with the right bytes.
    created = working_dir / _WRITE_OK_PATH
    assert created.is_file(), "the approved write must create the file"
    assert created.read_text(encoding="utf-8") == _WRITE_OK_BODY

    # 3. the denied write left NO file and the model was told (denial fed back as a tool result).
    assert not (working_dir / _WRITE_DENIED_PATH).exists(), "a denied write must not hit disk"
    assert "the user denied this write" in transcript, "the denial must reach the model"

    # 4. todo_write rewrote the per-run task store and emitted a checklist line.
    assert [t.content for t in deps.task_store] == [_TODO_CONTENT]
    assert deps.task_store[0].status == "in_progress"

    # 5. ask_user surfaced the question and the fake answer came back as the tool result.
    assert resolvers.questions_asked == [_ASK_QUESTION]
    assert any(_ASK_ANSWER in r for r in returns), "the ask_user answer must reach the model"

    # Under ``default`` mode only the two mutating ``write`` calls reached the human; the
    # read-only tools (read / todo_write) auto-allowed and ask_user is ungated, so
    # neither produced a permission request (ADR-0003 §1).
    asked_tools = [r.tool_name for r in resolvers.permission_requests]
    assert asked_tools == ["write", "write"]

    # The real renderer ran on every emitted event without raising, and the rendered transcript
    # carries what the user would have seen: the surfaced question and the checklist task line.
    rendered = render_buffer.getvalue()
    assert _ASK_QUESTION in rendered, "the ask_user question must render in the TUI transcript"
    assert _TODO_CONTENT in rendered, "the task checklist must render in the TUI transcript"
    assert _FINAL_TEXT in rendered, "the streamed assistant text must render"

    # --- assert: the session log was written and replays (ADR-0002 §9) ----------------------
    replayed = session_log.load(log.path)
    assert replayed == handler.message_history, "the JSONL log must replay the whole conversation"
    # The log is append-only: a header line + one batch per turn (five turns).
    lines = [ln for ln in log.path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1 + 5, "header + one append per turn"

    # --- assert: ``--resume`` replays the history into a fresh handler (ADR-0002 §9) --------
    resumed = session_log.load_latest(sessions_dir)
    assert resumed == handler.message_history
    fresh_handler = AgentTurnHandler(agent, deps=deps, message_history=resumed)
    assert fresh_handler.message_history == handler.message_history
    # A resumed handler treats the replayed prefix as already-persisted (so resume never
    # re-writes it): a no-op would-stop persists nothing new.
    user_prompts = [
        str(part.content)
        for message in fresh_handler.message_history
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert any("read the notes file" in p for p in user_prompts)
