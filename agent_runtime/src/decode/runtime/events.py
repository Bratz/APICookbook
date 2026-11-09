"""The step log ``decode run --events-jsonl`` writes (APICookbook fork F2b).

One JSON object per line, appended as the run goes, so the sandbox host's runner can stream a run's
steps to the website while it is still working:

    {"type": "tool_call",   "at": ..., "id": ..., "tool": "bash", "args": "{\"command\": ...}"}
    {"type": "tool_result", "at": ..., "id": ..., "tool": "bash", "ok": true, "output": "..."}

Fed by pydantic-ai's ``event_stream_handler`` on the headless ``agent.run``: no change to the loop.
Arguments and outputs are cut (they can be whole files), and a write failure never stops the run.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent

logger = logging.getLogger(__name__)

MAX_ARGS_CHARS = 4000
MAX_OUTPUT_CHARS = 4000


def event_record(event: AgentStreamEvent) -> dict[str, Any] | None:
    """The log line for one stream event; ``None`` for the events the log leaves out."""
    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        return {
            "type": "tool_call",
            "id": part.tool_call_id,
            "tool": part.tool_name,
            "args": part.args_as_json_str()[:MAX_ARGS_CHARS],
        }
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        return {
            "type": "tool_result",
            "id": part.tool_call_id,
            "tool": part.tool_name,
            "ok": part.outcome == "success",
            "output": part.model_response_str()[:MAX_OUTPUT_CHARS],
        }
    return (
        None  # ponytail: text deltas and part starts stay out; the final answer is in the summary
    )


def event_writer(
    path: Path,
) -> Callable[[RunContext[Any], AsyncIterable[AgentStreamEvent]], Awaitable[None]]:
    """An ``event_stream_handler`` that appends each step to ``path`` as it happens."""
    path.parent.mkdir(parents=True, exist_ok=True)

    async def handler(_ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            record = event_record(event)
            if record is None:
                continue
            try:
                with path.open("a", encoding="utf-8") as out:
                    out.write(
                        json.dumps({**record, "at": round(time.time(), 3)}, ensure_ascii=False)
                        + "\n"
                    )
            except OSError:
                logger.warning("could not append to the events log %s", path, exc_info=True)

    return handler
