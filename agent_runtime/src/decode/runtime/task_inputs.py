"""Resolve the task and model override for a headless ``decode run`` (ADR-0019 §4).

APICookbook fork: the Kitaru Worker branch (a task read from ``KITARU_TASK_INPUTS`` on a replay) was
removed with Kitaru itself (docs/decode-fork-plan.md). A run now needs an explicit task: the website
always passes one.
"""

from __future__ import annotations

# The one friendly line for "no task".
NO_TASK_MESSAGE = 'decode run needs a TASK to run: pass it as an argument (decode run "<task>").'


class WorkerTaskInputError(RuntimeError):
    """No runnable task was given. (Name kept from upstream so callers and tests read unchanged.)"""


def resolve_task_and_model(task: str | None, model: str | None) -> tuple[str, str | None]:
    """Return ``(task, model-override)``; raise :class:`WorkerTaskInputError` when there is no task."""
    explicit_task = (task or "").strip()
    if not explicit_task:
        raise WorkerTaskInputError(NO_TASK_MESSAGE)
    return explicit_task, _clean_model_override(model)


def _clean_model_override(model: str | None) -> str | None:
    """A Model Override as ``None`` (use the configured model) or a non-blank model id."""
    cleaned = (model or "").strip()
    return cleaned or None
