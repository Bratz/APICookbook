"""The blocking ``ask_user`` tool — the model asks the human a free-form question.

The one blocking tool: the TUI surfaces the question and the human's typed line becomes the tool
result. Not gated — it IS the human-interaction tool, so gating it would double-prompt. It awaits
the answer on ``ctx.deps.resolve_user_question``, the single mid-turn decision channel (never a
second prompt); headless / cancelled requests map to a model-readable
:class:`pydantic_ai.ModelRetry` so the turn never hangs. See ADR-0002 §2,3,7.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext

from decode.agent.deps import AgentDeps
from decode.entities import events

logger = logging.getLogger(__name__)

ASK_USER_TOOL_NAME = "ask_user"

# The message fed back to the model when no human can answer (headless run / cancelled prompt).
_NO_INTERACTIVE_USER_MESSAGE = (
    "No interactive user is attached, so the question could not be answered. "
    "Proceed without asking the user, or make a reasonable assumption and state it."
)


class PlanOption(BaseModel):
    """One choice the human can pick with a click (APICookbook fork)."""

    label: str  # a few words, the action: "Scan a sample of 250"
    detail: str = ""  # its one-line trade-off: "About 250 calls; a partial answer, now"


def plan_choices(options: list[PlanOption] | None) -> list[dict[str, str]]:
    """Two options or none: one option is not a choice. Labels up to 80 characters, details 120."""
    picked = [
        {"label": o.label.strip()[:80], "detail": o.detail.strip()[:120]}
        for o in options or []
        if o.label.strip()
    ][:2]
    return picked if len(picked) == 2 else []


class MoreQuestion(BaseModel):
    """A further question on the same card as the first: an interview asks them together."""

    question: str
    options: list[PlanOption] | None = None


def more_choices(more: list[MoreQuestion] | None) -> list[dict[str, object]]:
    """At most two further questions (three on a card), each with two options or none."""
    return [
        {"question": m.question.strip()[:300], "options": plan_choices(m.options)}
        for m in more or []
        if m.question.strip()
    ][:2]


class NoInteractiveUserError(RuntimeError):
    """Raised by the headless resolver: there is no interactive user to answer ``ask_user``.

    :func:`ask_user` catches it and re-raises a model-readable :class:`pydantic_ai.ModelRetry`
    so an unattended run never hangs waiting for an answer that will never arrive.
    """


async def deny_user_question_resolver(
    question: str,
    options: list[dict[str, str]] | None = None,
    more: list[dict[str, object]] | None = None,
) -> str:
    """The safe headless default: there is no interactive user, so refuse to answer.

    Used when there is no terminal to ask (an unattended / piped run). Raising (rather than
    returning a fake answer) is the honest default — the model is told plainly that no human is
    attached via the :class:`pydantic_ai.ModelRetry` :func:`ask_user` builds from this.
    """
    logger.debug("headless ask_user resolver refusing question=%r", question)
    raise NoInteractiveUserError(_NO_INTERACTIVE_USER_MESSAGE)


async def ask_user(
    ctx: RunContext[AgentDeps],
    question: str,
    options: list[PlanOption] | None = None,
    more_questions: list[MoreQuestion] | None = None,
) -> str:
    """Ask the human ``question`` and return their typed answer (ADR-0002 §2,7).

    ``options``: exactly two choices the human can pick with one click, for example two plans, the
    recommended one first. ``label`` is a few words; ``detail`` is its one-line trade-off. One option
    is not a choice and is ignored. The human can still type their own answer. Leave it out for an
    open question.

    ``more_questions``: at most two further questions shown on the same card and answered together
    (an interview before a larger build), each with its own two ``options`` or none. The answers
    come back as one text, a line per question.

    Emits an :class:`~decode.entities.events.AskUserRequested` event so the TUI renders the
    question, then awaits the human's free-text answer on ``ctx.deps.resolve_user_question`` —
    the single mid-turn decision channel (never a second prompt). The typed line is returned to
    the model as the tool result.

    Not gated: ``ask_user`` is the human-interaction tool itself, so it does **not** raise
    :class:`pydantic_ai.ApprovalRequired` and is registered to skip the permission gate.

    Raises a model-readable :class:`pydantic_ai.ModelRetry` (never hangs) when no interactive
    user is attached (:class:`NoInteractiveUserError`) or the pending request is cancelled
    (turn aborted / REPL shutting down), so the model can proceed without the answer.
    """
    ctx.deps.emit(events.AskUserRequested(tool_call_id=ctx.tool_call_id, question=question))
    logger.debug("ask_user awaiting an answer (question=%r)", question)
    try:
        # APICookbook fork: options go only to a resolver that shows them (the visitor's page), so
        # every other resolver keeps upstream's one-argument shape.
        choices = plan_choices(options)
        more = more_choices(more_questions)
        if more:
            answer = await ctx.deps.resolve_user_question(question, options=choices, more=more)
        elif choices:
            answer = await ctx.deps.resolve_user_question(question, options=choices)
        else:
            answer = await ctx.deps.resolve_user_question(question)
    except NoInteractiveUserError as exc:
        logger.debug("ask_user has no interactive user (question=%r)", question)
        raise ModelRetry(str(exc)) from exc
    except asyncio.CancelledError as exc:
        logger.debug("ask_user request cancelled (question=%r)", question)
        raise ModelRetry(
            "The user dismissed the question without answering; proceed without it."
        ) from exc
    logger.debug("ask_user got an answer (%d chars)", len(answer))
    return answer
