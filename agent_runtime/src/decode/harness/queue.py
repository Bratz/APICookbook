"""The two interaction queues that back the Priority Gate (ADR-0002 §4).

**steering** is drained before each model-request leg (boundary-inject, never mid-stream);
**follow-up** only at the would-stop boundary, continuing the conversation as the next leg of the
same turn. Both are plain ``asyncio.Queue``s of user-text strings.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field


class InputIntent(enum.Enum):
    """What the user signalled with the submitted line (ADR-0002 §4-5).

    Moved here from the removed ``decode.tui.app`` (APICookbook fork): the web chat sends STEER for a
    normal message (new turn when idle, steer when busy), FOLLOW_UP to queue one, ABORT for Stop.
    """

    STEER = "steer"
    FOLLOW_UP = "follow-up"
    ABORT = "abort"


@dataclass(slots=True)
class InteractionQueues:
    """The steering + follow-up queues for one harness session; the runner owns the drain timing."""

    steering: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    follow_up: asyncio.Queue[str] = field(default_factory=asyncio.Queue)

    def drain_steering(self) -> list[str]:
        """Non-blocking: remove and return every queued steering message, oldest first."""
        return _drain(self.steering)

    def drain_follow_up(self) -> list[str]:
        """Non-blocking: remove and return every queued follow-up message, oldest first."""
        return _drain(self.follow_up)

    def clear(self) -> None:
        """Discard everything in both queues (used on cooperative abort, ADR-0002 §5)."""
        _drain(self.steering)
        _drain(self.follow_up)


def _drain(queue: asyncio.Queue[str]) -> list[str]:
    items: list[str] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return items
