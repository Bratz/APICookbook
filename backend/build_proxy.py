"""Build & run's BaNCS proxy (docs/decode-fork-plan.md, F2a).

The sandbox workers' only network destination (10.0.0.49:7601, private). Every path carries a run's
token, ``/r/<token>/...``, so each call belongs to one run and stops working when the run ends.

    /r/<token>/<BaNCS path>          GET passes through. POST/PUT/PATCH/DELETE are HELD until the
                                     run's visitor approves; reject or a timeout answers 403.
    /r/<token>/_kb/search?q=...      Read-only API knowledge for the agent's tools (backend/api_lookup.py),
    /r/<token>/_kb/endpoint/<name>   over the same knowledge base as the site's endpoint cards.
    /r/<token>/_kb/body-schema/<name>

It runs inside cookbook-api's process (``serve()``, started from the app's lifespan), so the
approvals the main app's routes record and the knowledge base are shared without any IPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from backend import api_lookup
from backend.config import settings

logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
WRITE_TIMEOUT_S = 120.0  # a held write the visitor doesn't answer counts as rejected
TOKEN_NEEDED = ("This call reaches the bank, so it needs your BaNCS access token. Add it under "
                "Authentication & headers on the home page, or send it in the Accesstoken header.")
RUN_TTL_S = 20 * 60  # a token dies with its run, or after this, whichever comes first
MAX_CALLS_PER_RUN = 300
# 403, not 429: a 429 reads as "slow down and retry", and agents then sleep and retry against a limit that never resets.
CALL_LIMIT = (f"Not sent: this run has used its {MAX_CALLS_PER_RUN} bank calls. Do not retry or sleep, this limit does "
              "not reset. Report what you found so far, or ask the visitor with ask_user how to narrow the task.")
QUESTION_TIMEOUT_S = 300.0  # a question the visitor doesn't answer in 5 minutes goes back unanswered
MAX_QUESTIONS_PER_RUN = 3
MAX_FAILED_WRITES_IN_A_ROW = 3  # then the run must ask or stop: it is guessing values against a shared bank
MAX_WRITES_PER_RUN = 12
STOPPED_WRITES = ("Not sent, and not shown to the visitor: {why}. Stop changing values and retrying. If a value only "
                  "the visitor can know is missing (a product, a branch, a customer), ask them with ask_user: one "
                  "short question with the options you found. Otherwise finish and say what you tried and what "
                  "the bank answered.")
MAX_BODY_BYTES = 256 * 1024
_FORWARD_HEADERS = {"content-type", "accept", "entity", "userid", "languagecode", "channeltype", "accesstoken"}


@dataclass
class HeldWrite:
    id: str
    method: str
    path: str
    body_preview: str
    created: float = field(default_factory=time.time)
    decision: Optional[str] = None  # "approved" | "rejected" | "timeout"
    event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "method": self.method, "path": self.path, "body": self.body_preview,
                "decision": self.decision, "waiting_s": round(time.time() - self.created)}


@dataclass
class HeldQuestion:
    """A question the agent asks the visitor (decode's ask_user), held until they answer or time runs out."""
    id: str
    question: str
    created: float = field(default_factory=time.time)
    status: str = "pending"  # "pending" | "answered" | "skipped" | "timeout"
    answer: str = ""
    event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "status": self.status, "answer": self.answer,
                "waiting_s": round(time.time() - self.created), "timeout_s": int(QUESTION_TIMEOUT_S)}


@dataclass
class ProxyRun:
    token: str
    created: float = field(default_factory=time.time)
    ended: bool = False
    writes: dict[str, HeldWrite] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    bank_token: str = field(default="", repr=False)  # the visitor's; the sandbox never sees it
    # (method, URL, body) -> how the visitor and the bank answered it: a repeat is answered the same way, unasked
    answered: dict[tuple, dict[str, Any]] = field(default_factory=dict, repr=False)
    questions: dict[str, HeldQuestion] = field(default_factory=dict)
    visitor: bool = True  # False for operator runs (evals, smoke): nobody watches, so questions are answered at once
    writes_sent: int = 0  # writes the visitor was asked about
    failed_in_a_row: int = 0  # approved writes the bank refused, since the last success or answer

    @property
    def alive(self) -> bool:
        return not self.ended and time.time() - self.created < RUN_TTL_S


RUNS: dict[str, ProxyRun] = {}


# ---------------------------------------------------------------- used by cookbook-api's build routes

def visitor_token(accesstoken: str = Header("")) -> str:
    """The visitor's BaNCS access token. APICookbook never checks it or stores it: it passes it to the bank."""
    return accesstoken.strip()


def require_token(token: str) -> None:
    if settings.bancs_token_required and not token:
        raise HTTPException(status_code=401, detail=TOKEN_NEEDED)


def new_run() -> ProxyRun:
    for token in [t for t, r in RUNS.items() if time.time() - r.created > 2 * RUN_TTL_S]:
        RUNS.pop(token, None)  # ponytail: prune on create; a dict of a few dozen runs a day
    run = ProxyRun(token=secrets.token_urlsafe(24))
    RUNS[run.token] = run
    return run


def end_run(token: str) -> None:
    run = RUNS.get(token)
    if run:
        run.ended = True
        for write in run.writes.values():
            if write.decision is None:
                write.decision = "rejected"
                write.event.set()
        for question in run.questions.values():
            if question.status == "pending":
                question.status = "timeout"
                question.event.set()


def decide(token: str, write_id: str, approve: bool) -> Optional[HeldWrite]:
    run = RUNS.get(token)
    write = run.writes.get(write_id) if run else None
    if write is None or write.decision is not None:
        return None
    write.decision = "approved" if approve else "rejected"
    write.event.set()
    return write


def answer_question(token: str, question_id: str, answer: str) -> Optional[HeldQuestion]:
    """The visitor's answer (empty = skipped); None when the question is not pending."""
    run = RUNS.get(token)
    question = run.questions.get(question_id) if run else None
    if question is None or question.status != "pending":
        return None
    question.answer = answer.strip()[:2000]
    question.status = "answered" if question.answer else "skipped"
    if question.answer:
        run.failed_in_a_row = 0  # new information: the run may try writes again
    question.event.set()
    return question


def questions(token: str) -> list[dict[str, Any]]:
    run = RUNS.get(token)
    return [q.public() for q in run.questions.values()] if run else []


def pending(token: str) -> list[dict[str, Any]]:
    run = RUNS.get(token)
    return [w.public() for w in run.writes.values() if w.decision is None] if run else []


def writes(token: str) -> list[dict[str, Any]]:
    """Every write the run made, held or decided, oldest first."""
    run = RUNS.get(token)
    return [w.public() for w in run.writes.values()] if run else []


def calls(token: str) -> list[dict[str, Any]]:
    run = RUNS.get(token)
    return list(run.calls) if run else []


# ---------------------------------------------------------------- the proxy app (port 7601)

proxy_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _run_or_404(token: str) -> ProxyRun:
    run = RUNS.get(token)
    if run is None or not run.alive:
        raise HTTPException(status_code=404, detail="no such run (or it has ended)")
    return run


def _kb() -> tuple[Any, Any]:
    from backend.main import app  # the live knowledge base and RAG engine, loaded at startup

    if not getattr(app.state, "kb", None):
        raise HTTPException(status_code=503, detail="the knowledge base is still loading")
    return app.state.kb, getattr(app.state, "rag_engine", None)


@proxy_app.get("/r/{token}/_kb/search")
def kb_search(token: str, q: str) -> Any:
    _run_or_404(token)
    kb, rag = _kb()
    return api_lookup.search_endpoints(kb, rag, q[:200])


@proxy_app.get("/r/{token}/_kb/endpoint/{name}")
def kb_endpoint(token: str, name: str) -> Any:
    _run_or_404(token)
    result = api_lookup.get_endpoint(_kb()[0], name)
    if "url_template" in result:  # point the agent at this run's proxy, never the bank's own host
        result["url_template"] = result["url_template"].replace(api_lookup.BASE_URL, f"{public_base(token)}/Core")
    return result


def public_base(token: str) -> str:
    """The run's proxy address as the sandbox sees it (BUILD_PROXY_BASE is the main VM's private address)."""
    return f"{os.getenv('BUILD_PROXY_BASE', 'http://10.0.0.49:7601').rstrip('/')}/r/{token}"


@proxy_app.get("/r/{token}/_kb/body-schema/{name}")
def kb_body_schema(token: str, name: str) -> Any:
    _run_or_404(token)
    return api_lookup.get_body_schema(_kb()[0], name)  # already compacted to the tool size cap


def kb_area(token: str, area: str) -> Any:
    _run_or_404(token)
    return api_lookup.list_endpoints(_kb()[0], area)


@proxy_app.post("/r/{token}/_ask")
async def ask(token: str, request: Request) -> Any:
    """decode's ask_user for this run: hold the question for the visitor; answer when they reply or time runs out."""
    run = _run_or_404(token)
    question = str((await request.json()).get("question") or "").strip()[:1000]
    if not question:
        return {"answer": "", "message": "Ask a question in words."}
    if len(run.questions) >= MAX_QUESTIONS_PER_RUN:
        return {"answer": "", "message": f"You have asked {MAX_QUESTIONS_PER_RUN} questions in this run, the limit. "
                                         "Finish with what you have and say what is still open."}
    if not run.visitor:
        return {"answer": "", "message": "No visitor is watching this run. Decide yourself, state the assumption, "
                                         "or finish and say what is still open."}
    held = HeldQuestion(id=uuid.uuid4().hex[:12], question=question)
    run.questions[held.id] = held
    try:
        await asyncio.wait_for(held.event.wait(), timeout=QUESTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        if held.status == "pending":
            held.status = "timeout"
    messages = {"skipped": "The visitor skipped the question. Do not ask it again: decide yourself, state the "
                           "assumption, or finish and say what is still open.",
                "timeout": "The visitor did not answer in time. Do not ask again: finish with what you tried and "
                           "the open question."}
    return {"answer": held.answer, "message": messages.get(held.status, "")}


@proxy_app.api_route("/r/{token}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def forward(token: str, path: str, request: Request) -> Response:
    run = _run_or_404(token)
    if len(run.calls) >= MAX_CALLS_PER_RUN:
        if len(run.calls) == MAX_CALLS_PER_RUN:  # the first refused call goes in the log, so the visitor sees why
            run.calls.append({"method": request.method.upper(), "path": "/" + path, "at": time.time(), "decision": "limit"})
        return JSONResponse({"error": CALL_LIMIT, "code": "RUN_CALL_LIMIT"}, status_code=403)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"error": "request body too large"}, status_code=413)
    method = request.method.upper()
    target = httpx.URL(f"{settings.api_base_url.rstrip('/')}/{path.lstrip('/')}", params=request.query_params)
    allowed = httpx.URL(settings.api_base_url)
    if (target.scheme, target.host, target.port) != (allowed.scheme, allowed.host, allowed.port):
        return JSONResponse({"error": "only BaNCS paths are allowed"}, status_code=400)

    record: dict[str, Any] = {"method": method, "path": "/" + path, "at": time.time()}
    run.calls.append(record)
    key = (method, str(target), body)
    if method in WRITE_METHODS and key in run.answered:
        # The same write again (a script re-run): the visitor already answered it, so don't ask twice.
        earlier = run.answered[key]
        record.update(repeat_of=earlier["write_id"], decision=earlier["decision"])
        if earlier["decision"] != "approved":
            return JSONResponse({"error": f"This {method} was not sent to the bank: the visitor rejected "
                                          "the same request earlier in this run."}, status_code=403)
        record.update(status=earlier["status"], answer=earlier["answer"])
        return Response(content=earlier["content"], status_code=earlier["status"], media_type=earlier["media_type"])
    if method in WRITE_METHODS and (run.failed_in_a_row >= MAX_FAILED_WRITES_IN_A_ROW or run.writes_sent >= MAX_WRITES_PER_RUN):
        why = (f"the bank refused the last {run.failed_in_a_row} writes in a row" if run.failed_in_a_row >= MAX_FAILED_WRITES_IN_A_ROW
               else f"this run has made {MAX_WRITES_PER_RUN} writes, the limit")
        record["decision"] = "stopped"
        return JSONResponse({"error": STOPPED_WRITES.format(why=why)}, status_code=409)
    if method in WRITE_METHODS:
        run.writes_sent += 1
        write = HeldWrite(id=uuid.uuid4().hex[:12], method=method, path="/" + path,
                          body_preview=body[:4000].decode("utf-8", "replace"))
        run.writes[write.id] = write
        record["write_id"] = write.id
        logger.info("[build-proxy] holding %s /%s for approval (%s)", method, path, write.id)
        try:
            await asyncio.wait_for(write.event.wait(), timeout=WRITE_TIMEOUT_S)
        except asyncio.TimeoutError:
            write.decision = write.decision or "timeout"
        record["decision"] = write.decision
        if write.decision == "rejected":
            run.answered[key] = {"write_id": write.id, "decision": "rejected"}
        if write.decision != "approved":
            reason = "the visitor did not approve it in time" if write.decision == "timeout" else "the visitor rejected it"
            return JSONResponse({"error": f"This {method} was not sent to the bank: {reason}."}, status_code=403)

    headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_HEADERS}
    for name, value in (("entity", settings.bancs_entity), ("userId", str(settings.bancs_user_id)),
                        ("languageCode", str(settings.bancs_language_code))):
        if not any(k.lower() == name.lower() for k in headers):
            headers[name] = value  # the agent's code may omit them; the bank requires them
    # The bank answers XML unless asked for JSON; requests/httpx/curl send */* by default.
    accept = next((k for k in headers if k.lower() == "accept"), None)
    if accept is None or headers[accept].strip() in ("", "*/*"):
        headers.pop(accept, None)
        headers["Accept"] = "application/json"
    if run.bank_token:  # the visitor's token wins over whatever the agent's code sent (it has none)
        for name in [k for k in headers if k.lower() == "accesstoken"]:
            headers.pop(name)
        headers["Accesstoken"] = run.bank_token
    started = time.time()
    try:
        async with _bank_client() as client:
            upstream = await client.request(method, target, headers=headers, content=body or None)
    except httpx.HTTPError as exc:
        record.update(status=502, ms=int((time.time() - started) * 1000))
        return JSONResponse({"error": f"the bank did not answer: {type(exc).__name__}"}, status_code=502)
    record.update(status=upstream.status_code, ms=int((time.time() - started) * 1000))
    if method in WRITE_METHODS:  # the visitor approved it: show them what the bank said, and keep it for a repeat
        run.failed_in_a_row = 0 if upstream.is_success else run.failed_in_a_row + 1
        record["answer"] = upstream.text[:300]
        run.answered[key] = {"write_id": record["write_id"], "decision": "approved", "status": upstream.status_code,
                             "answer": record["answer"], "content": upstream.content,
                             "media_type": upstream.headers.get("content-type")}
    return Response(content=upstream.content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type"))


def _bank_client() -> httpx.AsyncClient:
    """The client for the bank (a seam: tests swap in a fake bank)."""
    return httpx.AsyncClient(timeout=30.0)


class _QuietServer(uvicorn.Server):
    """A second server in the main app's event loop: it must not take over the process's signals."""

    @contextlib.contextmanager
    def capture_signals(self):  # uvicorn >= 0.29
        yield

    def install_signal_handlers(self) -> None:  # older uvicorn
        pass


def server(port: int) -> uvicorn.Server:
    """The proxy's server; the lifespan runs ``serve()`` as a task and sets ``should_exit`` on shutdown."""
    return _QuietServer(uvicorn.Config(proxy_app, host="0.0.0.0", port=port, log_level="warning",
                                       access_log=False, lifespan="off"))
