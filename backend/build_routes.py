"""Build & run routes (docs/decode-fork-plan.md, F2a-F4): start a run on the sandbox host, watch it
live, approve or reject its bank writes, stop it.

Who may do what (F4, decided 2026-11-09: every visitor, with caps):
- **Starting a run** is open to every visitor when ``BUILD_PUBLIC=true``, behind this mode's own caps
  (``BUILD_RATE_PER_HOUR`` per IP, ``BUILD_DAILY_CAP_USD``, ``BUILD_MONTHLY_CAP_USD``, a ledger apart
  from the code agent's). The operator's ``Authorization: Bearer <RUNNER_TOKEN>`` skips the per-IP limit.
- **Everything about one run** needs that run's key: a random secret the start response hands back and
  the page keeps in memory (``X-Run-Key``). Nobody types it; only the browser that started a run can
  watch it, approve its writes or stop it. The operator token works on every run.

    GET    /api/build/status                      {"available", caps}
    POST   /api/build/runs                        {"task", "previous"?: {"id", "key"}}  → {"id", "key"}
           (with "previous": a follow-up, which starts from that run's files and a note on it)
    GET    /api/build/runs/{id}                   the runner's status + held writes + the bank calls
    GET    /api/build/runs/{id}/events            the run live, as server-sent events (F2b):
           step (each tool call / result) · write (held, waiting) · write_decided · bank_call · done
    POST   /api/build/runs/{id}/writes/{write}    {"approve": true|false}
    DELETE /api/build/runs/{id}                   stop it
    GET    /api/build/runs/{id}/files             the files the agent wrote, pointed at the real bank
    GET    /api/build/runs/{id}/files.zip         the same, as a zip with a README
    POST   /api/build/runs/{id}/preview           start the run's files in a sandbox worker (10 minutes): the app
                                                  that preview.json names, when there is one, and a terminal
    POST   /api/build/runs/{id}/preview/term      type into that terminal and read its output
    GET    /api/build/runs/{id}/preview/writes    the bank writes the preview is holding for approval
    POST   /api/build/runs/{id}/preview/writes/{write}   {"approve": true|false}
    POST   /api/build/runs/{id}/preview/http      one request to that app: {"method", "path", ...} in, the app's
                                                  status, content type and body (base64) out, as JSON
    DELETE /api/build/runs/{id}/preview           stop it
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import os
import secrets
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend import api_lookup, build_proxy
from backend.config import settings
from backend.agent_guard import AgentGuard

router = APIRouter(prefix="/api/build")

SETTLE_POLL_S = 10.0  # how often a run's watcher asks the runner whether it has finished
_CAP_MESSAGES = {
    "monthly_cap": "Build & run has reached this month's budget. The code agent still answers.",
    "daily_cap": "Build & run has reached today's budget; it resets at 00:00 UTC. The code agent still answers.",
    "rate_limited": "You have started {n} runs in the last hour; please try again later.",
}


@dataclass
class BuildRun:
    token: str  # the run's proxy token
    key: str  # the run's own secret, held by the browser that started it
    started: float = field(default_factory=time.time)
    settled: bool = False
    operator: bool = False  # started with RUNNER_TOKEN (evals, smoke): kept out of the visitors' caps
    task: str = ""  # what the visitor typed: a follow-up's note quotes it
    preview: Optional[dict[str, str]] = None  # the open preview of this run: {"id", "token"}


# runner run id -> run. ponytail: in memory, so a cookbook-api restart forgets runs in flight (their
# tokens and keys die with them, which fails safe; the runner still stops them at its time limit).
_RUNS: dict[str, BuildRun] = {}
_GUARD: Optional[AgentGuard] = None


def _guard() -> AgentGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = AgentGuard(os.getenv("BUILD_DATA_DIR", "data/build"),
                            per_hour=int(os.getenv("BUILD_RATE_PER_HOUR", "3")),
                            daily_usd=float(os.getenv("BUILD_DAILY_CAP_USD", "5")),
                            monthly_usd=float(os.getenv("BUILD_MONTHLY_CAP_USD", "50")))
    return _GUARD


def _runner() -> tuple[str, str]:
    url, token = os.getenv("RUNNER_URL", "http://10.0.0.223:7700"), os.getenv("RUNNER_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="Build & run is not configured on this server.")
    return url.rstrip("/"), token


def _public() -> bool:
    return os.getenv("BUILD_PUBLIC", "").lower() == "true"


def _is_operator(authorization: str) -> bool:
    _, token = _runner()
    return hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode())


def _run_for(run_id: str, authorization: str, run_key: str) -> BuildRun:
    """The run, if the caller is the operator or holds this run's key; 404 otherwise (no probing)."""
    run = _RUNS.get(run_id)
    if run is None or not (_is_operator(authorization) or hmac.compare_digest(run_key.encode(), run.key.encode())):
        raise HTTPException(status_code=404, detail="no such run")
    return run


async def _call_runner(method: str, path: str, json: Any = None, timeout: float = 15.0) -> httpx.Response:
    url, token = _runner()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, f"{url}{path}", json=json,
                                        headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"the sandbox host did not answer: {type(exc).__name__}")


def _client_ip(request: Request) -> str:
    # Through the site, Caddy sets X-Forwarded-For to the Vercel server that connected to it (it does not
    # trust the incoming header), so that header names one of many Vercel servers, not the visitor.
    # Vercel's own header has the visitor, and Caddy passes it on.
    # ponytail: a caller that skips the site can send any value here and dodge the hourly limit; the
    # USD caps still hold. Check a shared secret from Vercel if that is ever abused.
    forwarded = request.headers.get("x-vercel-forwarded-for") or request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


async def settle(run_id: str, final: dict[str, Any]) -> None:
    """Once per run: record what it cost in the build ledger and end its proxy token."""
    run = _RUNS.get(run_id)
    if run is None or run.settled:
        return
    run.settled = True
    build_proxy.end_run(run.token)
    if run.operator:
        return  # evals and smoke runs report their own cost; the caps are the visitors' budget
    summary = final.get("summary") or {}
    _guard().record(kind="build", usd=float(summary.get("cost_usd") or 0), outcome=str(final.get("status")),
                    seconds=time.time() - run.started, tools=len(build_proxy.calls(run.token)),
                    tokens={k: summary.get(k) for k in ("input_tokens", "output_tokens") if summary.get(k) is not None})


async def _settle_when_done(run_id: str) -> None:
    """The run's watcher: its cost is counted even if nobody watches it to the end."""
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        await asyncio.sleep(SETTLE_POLL_S)
        try:
            response = await _call_runner("GET", f"/runs/{run_id}")
        except HTTPException:
            continue
        if response.status_code == 200 and response.json().get("status") != "running":
            await settle(run_id, response.json())
            return
    await settle(run_id, {"status": "unknown"})


class PreviousRun(BaseModel):
    id: str
    key: str


class StartRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    previous: Optional[PreviousRun] = None  # a follow-up of this run (the page holds its key)


NOTE_TASK_CHARS, NOTE_ANSWER_CHARS = 600, 1500  # a follow-up's note stays small: the files carry the work


async def _follow_up(previous: PreviousRun, task: str, new_token: str) -> tuple[str, list[dict[str, str]]]:
    """The task for a follow-up run (a short note on the run before, then the new ask) and that run's
    files, pointed at the new run's proxy. Each run's context stays one task long; the work carries over."""
    before = _run_for(previous.id, "", previous.key)  # 404 unless this browser started that run
    listing = await _run_files(previous.id, "", previous.key)  # its files, pointed at the real bank
    response = await _call_runner("GET", f"/runs/{previous.id}")
    answer = response.json().get("answer", "") if response.status_code == 200 else ""
    answer = answer.replace(build_proxy.public_base(before.token), api_lookup.BASE_URL.removesuffix("/Core"))
    bank_root, proxy = api_lookup.BASE_URL.removesuffix("/Core"), build_proxy.public_base(new_token)
    files = [{"path": f["path"], "content": f["content"].replace(bank_root, proxy)} for f in listing["files"]]
    names = ", ".join(f["path"] for f in files) or "none"
    note = (f"This task follows on from your previous run in this chat.\n"
            f"Previous task: {before.task[:NOTE_TASK_CHARS]}\n"
            f"What you reported then (trimmed): {answer.strip()[:NOTE_ANSWER_CHARS]}\n"
            f"Your files from that run are already in the workspace: {names}. Read them and change them "
            f"rather than starting over; they already use this run's bank address.\n\nNew task: ")
    return note + task, files


class Decision(BaseModel):
    approve: bool


class Answer(BaseModel):
    answer: str = Field(default="", max_length=2000)  # empty: the visitor skipped the question
    answers: Optional[list[str]] = Field(default=None, max_length=3)  # an interview card: one per question


@router.get("/status")
async def build_status() -> dict[str, Any]:
    try:
        _runner()
        available = _public()
    except HTTPException:
        available = False
    return {"available": available, **_guard().status()}


@router.post("/runs", status_code=202)
async def start(body: StartRequest, request: Request, authorization: str = Header(""),
                accesstoken: str = Header("")) -> Any:
    operator = _is_operator(authorization)
    token = build_proxy.visitor_token(accesstoken)
    if not operator:
        if not _public():
            raise HTTPException(status_code=401, detail="Build & run is not open on this server.")
        build_proxy.require_token(token)
        blocked = _guard().check(_client_ip(request))
        if blocked:
            message = _CAP_MESSAGES.get(blocked[0], blocked[1]).format(n=_guard().per_hour)
            return JSONResponse(status_code=429, content={"error": blocked[0], "message": message})
    proxy = build_proxy.new_run()
    proxy.bank_token = token
    proxy.visitor = not operator  # an eval or smoke run has nobody to answer the agent's questions
    task, files = body.task, []
    if body.previous:
        try:
            task, files = await _follow_up(body.previous, body.task, proxy.token)
        except HTTPException:
            build_proxy.end_run(proxy.token)
            raise
    response = await _call_runner("POST", "/runs", {"task": task, "files": files,
                                                    "proxy_url": build_proxy.public_base(proxy.token)})
    if response.status_code != 202:
        build_proxy.end_run(proxy.token)
        busy = response.status_code == 429
        return JSONResponse(status_code=429 if busy else 502, content={
            "error": "busy" if busy else "runner_refused",
            "message": "Both sandboxes are busy. Try again in a minute." if busy else "The sandbox could not start the run."})
    run_id = response.json()["id"]
    _RUNS[run_id] = BuildRun(token=proxy.token, key=secrets.token_urlsafe(24), operator=operator, task=body.task)
    asyncio.create_task(_settle_when_done(run_id))
    return {"id": run_id, "key": _RUNS[run_id].key}


@router.get("/runs/{run_id}")
async def status(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    response = await _call_runner("GET", f"/runs/{run_id}")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="the runner does not know this run")
    final = response.json()
    if final.get("status") != "running":
        await settle(run_id, final)
    return {**final, "pending_writes": build_proxy.pending(run.token), "bank_calls": build_proxy.calls(run.token)}


@router.post("/runs/{run_id}/writes/{write_id}")
async def decide(run_id: str, write_id: str, body: Decision, authorization: str = Header(""),
                 x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    write = build_proxy.decide(run.token, write_id, body.approve)
    if write is None:
        raise HTTPException(status_code=404, detail="no such pending write (decided, timed out or unknown)")
    return write.public()


@router.post("/runs/{run_id}/questions/{question_id}")
async def answer(run_id: str, question_id: str, body: Answer, authorization: str = Header(""),
                 x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    question = build_proxy.answer_question(run.token, question_id, body.answer, body.answers)
    if question is None:
        raise HTTPException(status_code=404, detail="no such open question (answered, timed out or unknown)")
    return question.public()


@router.delete("/runs/{run_id}")
async def stop(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    build_proxy.end_run(run.token)
    response = await _call_runner("DELETE", f"/runs/{run_id}")
    return {"id": run_id, "stopping": response.status_code == 200}


_README = """Code written by APICookbook Build & run (https://apicookbook.vercel.app).

It calls the TCS BaNCS demo bank directly:
  Base URL   {base}
  Headers    entity: {entity}   userId: {user}   languageCode: {lang}
             Accept: application/json (without it the bank answers in XML)
             Accesstoken: your BaNCS access token; the code reads it from the
             BANCS_ACCESS_TOKEN environment variable
             Content-Type: application/json on a request with a body
The demo bank does not check the token today, so the code also runs with the variable unset;
a bank that validates tokens needs it. There is no other sign-in.
The demo data is shared: a write you send changes what every user sees.
"""


async def _run_files(run_id: str, authorization: str, x_run_key: str) -> dict[str, Any]:
    """The run's files from the runner, with the private proxy address swapped for the bank's, so the
    code runs on the visitor's machine as it is."""
    run = _run_for(run_id, authorization, x_run_key)
    response = await _call_runner("GET", f"/runs/{run_id}/files")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="the runner does not know this run")
    listing = response.json()
    private = build_proxy.public_base(run.token)
    bank_root = api_lookup.BASE_URL.removesuffix("/Core")
    listing["files"] = [
        {**f, "content": f["content"].replace(f"{private}/Core", api_lookup.BASE_URL).replace(private, bank_root)}
        for f in listing.get("files", [])
        if not f["path"].startswith("/") and ".." not in f["path"].split("/")
    ]
    # The page builds the zip itself (a .zip request is blocked on some networks), so it needs the README.
    listing["readme"] = _README.format(base=api_lookup.BASE_URL, entity=settings.bancs_entity,
                                       user=settings.bancs_user_id, lang=settings.bancs_language_code)
    return listing


@router.get("/runs/{run_id}/files")
async def files(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> dict[str, Any]:
    return await _run_files(run_id, authorization, x_run_key)


@router.get("/runs/{run_id}/files.zip")
async def files_zip(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> Response:
    listing = await _run_files(run_id, authorization, x_run_key)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in listing["files"]:
            archive.writestr(f"bancs-run/{f['path']}", f["content"])
        archive.writestr("bancs-run/README.txt", listing["readme"])
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="bancs-run-{run_id[:8]}.zip"'})


# ---------------------------------------------------------------- preview: the built app, live, for this run's page only
# The app runs in a sandbox worker with the same isolation as a run. Nothing is published: every request
# to it comes through here with the run's key, goes to the runner, and is made inside the container.
# The answer goes back as JSON (never as a page from this origin), and the page shows it in a sandboxed frame.
PREVIEW_START_TIMEOUT_S = 40.0  # the runner waits up to 20 s for the app to answer on its port
PREVIEW_CALL_TIMEOUT_S = 165.0  # above the runner's 150 s for one request, which may hold a write 120 s
PREVIEW_QUICK_S = 10.0  # an answer within this goes back as plain JSON
PREVIEW_PING_S = 15.0  # a slower one: a space this often, so no hop between here and the browser gives up
PREVIEW_TERM_TIMEOUT_S = 35.0  # above the runner's 25 s wait for terminal output
PREVIEW_KINDS = ("page", "api", "script")


class PreviewCall(BaseModel):
    method: str = Field(default="GET", max_length=10)
    path: str = Field(max_length=2000)
    content_type: str = Field(default="", max_length=100)
    body_b64: str = Field(default="", max_length=1_400_000)


class PreviewTerm(BaseModel):
    input_b64: str = Field(default="", max_length=14_000)
    seen: int = Field(default=0, ge=0)
    wait_ms: int = Field(default=0, ge=0, le=25_000)


def _preview_plan(listing: dict[str, Any]) -> dict[str, Any]:
    """What preview.json asks for: {"kind", "command", "port", "routes"}. With no usable server in it
    (or no preview.json at all) the preview is a terminal on the run's files."""
    wanted = next((f["content"] for f in listing["files"] if f["path"] == "preview.json"), "")
    try:
        described = json.loads(wanted)
        described = described if isinstance(described, dict) else {}
    except ValueError:
        described = {}
    try:
        command, port = str(described["command"]).strip(), int(described["port"])
    except (ValueError, KeyError, TypeError):
        command, port = None, None
    kind = described.get("kind") if described.get("kind") in PREVIEW_KINDS else "page"
    routes = []
    for route in (described.get("routes") if isinstance(described.get("routes"), list) else [])[:12]:
        if isinstance(route, dict) and isinstance(route.get("path"), str) and route["path"].startswith("/"):
            routes.append({"method": str(route.get("method", "GET")).upper()[:10], "path": route["path"][:500],
                           "label": str(route.get("label", ""))[:80], "body": str(route.get("body", ""))[:4000]})
    return {"kind": kind if command else "script", "command": command or None, "port": port, "routes": routes}


async def _slow_json(work: Any) -> Response:
    """The answer of ``work`` as JSON. One that takes long (a bank write waits up to 120 s for the
    visitor) goes out as a stream that starts with spaces, which JSON allows, so the connection never
    sits silent. It is labelled as an event stream because Caddy flushes those and buffers the rest."""
    task = asyncio.ensure_future(work)
    done, _ = await asyncio.wait({task}, timeout=PREVIEW_QUICK_S)
    if done:
        return JSONResponse(task.result())

    async def body() -> AsyncIterator[str]:
        while not (await asyncio.wait({task}, timeout=PREVIEW_PING_S))[0]:
            yield " "
        try:
            yield json.dumps(task.result())
        except HTTPException as exc:  # the status line has gone out: say it in the app-answer shape
            yield json.dumps({"status": 410 if exc.status_code == 404 else 502, "content_type": "text/plain",
                              "body_b64": base64.b64encode(str(exc.detail).encode()).decode()})

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _stop_preview(run: BuildRun) -> None:
    if run.preview:
        preview, run.preview = run.preview, None
        build_proxy.end_run(preview["token"])
        try:
            await _call_runner("DELETE", f"/previews/{preview['id']}")
        except HTTPException:
            pass  # the runner stops it at its time limit anyway


@router.post("/runs/{run_id}/preview")
async def preview_start(run_id: str, authorization: str = Header(""), x_run_key: str = Header(""),
                        accesstoken: str = Header("")) -> Any:
    run = _run_for(run_id, authorization, x_run_key)
    listing = await _run_files(run_id, authorization, x_run_key)
    if listing.get("status") == "running":
        raise HTTPException(status_code=409, detail="the run has not finished")
    if not listing["files"]:
        raise HTTPException(status_code=422, detail="this run left no files to preview")
    plan = _preview_plan(listing)
    token = build_proxy.visitor_token(accesstoken)
    if not _is_operator(authorization):
        build_proxy.require_token(token)
    await _stop_preview(run)  # one preview for a run
    proxy = build_proxy.new_run()
    proxy.bank_token, proxy.visitor = token, False  # a write from the preview waits for approval, as in a run
    bank_root, base = api_lookup.BASE_URL.removesuffix("/Core"), build_proxy.public_base(proxy.token)
    files = [{"path": f["path"], "content": f["content"].replace(bank_root, base)} for f in listing["files"]]
    response = await _call_runner("POST", "/previews", {"files": files, "command": plan["command"],
                                                         "port": plan["port"] if plan["command"] else None},
                                  timeout=PREVIEW_START_TIMEOUT_S)
    if response.status_code != 201:
        build_proxy.end_run(proxy.token)
        if response.status_code == 429:
            return JSONResponse(status_code=429, content={
                "error": "busy", "message": "Both preview slots are in use. Try again in a few minutes."})
        if response.status_code == 422:
            log = str(response.json().get("log", "")).replace(base, bank_root)[-1500:]
            return JSONResponse(status_code=422, content={
                "error": "did_not_start", "message": "The app did not start in the sandbox.", "log": log})
        return JSONResponse(status_code=502, content={
            "error": "runner_refused", "message": "The sandbox could not start the preview."})
    made = response.json()
    run.preview = {"id": made["id"], "token": proxy.token}
    return {"expires_in": made["expires_in"], "kind": plan["kind"], "routes": plan["routes"]}


@router.post("/runs/{run_id}/preview/http")
async def preview_http(run_id: str, call: PreviewCall, authorization: str = Header(""),
                       x_run_key: str = Header("")) -> Any:
    run = _run_for(run_id, authorization, x_run_key)
    if not run.preview:
        raise HTTPException(status_code=404, detail="no preview is open for this run")
    return await _slow_json(_preview_relay(run, "http", call.model_dump(), PREVIEW_CALL_TIMEOUT_S))


async def _preview_relay(run: BuildRun, what: str, payload: dict[str, Any], timeout: float) -> Any:
    response = await _call_runner("POST", f"/previews/{run.preview['id']}/{what}", payload, timeout=timeout)
    if response.status_code == 404:  # its 10 minutes are over
        await _stop_preview(run)
        raise HTTPException(status_code=404, detail="the preview has ended")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="the sandbox refused the preview request")
    return response.json()


@router.post("/runs/{run_id}/preview/term")
async def preview_term(run_id: str, body: PreviewTerm, authorization: str = Header(""),
                       x_run_key: str = Header("")) -> Any:
    run = _run_for(run_id, authorization, x_run_key)
    if not run.preview:
        raise HTTPException(status_code=404, detail="no preview is open for this run")
    return await _preview_relay(run, "term", body.model_dump(), PREVIEW_TERM_TIMEOUT_S)


@router.get("/runs/{run_id}/preview/writes")
async def preview_writes(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    if not run.preview:
        raise HTTPException(status_code=404, detail="no preview is open for this run")
    return {"writes": build_proxy.writes(run.preview["token"])}


@router.post("/runs/{run_id}/preview/writes/{write_id}")
async def preview_decide(run_id: str, write_id: str, body: Decision, authorization: str = Header(""),
                         x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    write = build_proxy.decide(run.preview["token"], write_id, body.approve) if run.preview else None
    if write is None:
        raise HTTPException(status_code=404, detail="no such pending write (decided, timed out or unknown)")
    return write.public()


@router.delete("/runs/{run_id}/preview")
async def preview_stop(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> dict[str, Any]:
    run = _run_for(run_id, authorization, x_run_key)
    await _stop_preview(run)
    return {"stopped": True}


STREAM_POLL_S = 1.0
STREAM_MAX_S = 20 * 60  # past the runner's own 15-minute limit
STREAM_PING_S = 15.0


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_run(run_id: str, token: str) -> AsyncIterator[str]:
    """One run, live: its steps from the runner and its bank traffic from the proxy, until it ends."""
    cursor, writes_seen, decided_seen, calls_sent = 0, set(), set(), 0
    asked_seen, settled_seen = set(), set()
    started = last_sent = time.monotonic()
    while time.monotonic() - started < STREAM_MAX_S:
        response = await _call_runner("GET", f"/runs/{run_id}/events?after={cursor}")
        page = response.json() if response.status_code == 200 else {"status": "unknown", "events": [], "next": cursor}
        out = [_sse("step", e) for e in page["events"]]
        cursor = page["next"]
        for write in build_proxy.writes(token):
            if write["id"] not in writes_seen:
                writes_seen.add(write["id"])
                out.append(_sse("write", write))
            if write["decision"] and write["id"] not in decided_seen:
                decided_seen.add(write["id"])
                out.append(_sse("write_decided", write))
        for question in build_proxy.questions(token):
            if question["id"] not in asked_seen:
                asked_seen.add(question["id"])
                out.append(_sse("question", question))
            if question["status"] != "pending" and question["id"] not in settled_seen:
                settled_seen.add(question["id"])
                out.append(_sse("question_answered", question))
        calls = build_proxy.calls(token)
        # in order, once each has an outcome: the bank's status, or a write that was not sent
        while calls_sent < len(calls) and ("status" in calls[calls_sent]
                                           or calls[calls_sent].get("decision") not in (None, "approved")):
            out.append(_sse("bank_call", calls[calls_sent]))  # in order, once each has an outcome
            calls_sent += 1
        for chunk in out:
            yield chunk
        if out:
            last_sent = time.monotonic()
        if page["status"] not in ("running", "unknown") and not page["events"]:
            final = (await _call_runner("GET", f"/runs/{run_id}")).json()
            await settle(run_id, final)
            yield _sse("done", {k: final.get(k) for k in ("status", "exit_code", "answer", "summary")})
            return
        if time.monotonic() - last_sent > STREAM_PING_S:
            yield ": ping\n\n"  # keeps Vercel's and Caddy's idle timers from closing the stream
            last_sent = time.monotonic()
        await asyncio.sleep(STREAM_POLL_S)
    yield _sse("done", {"status": "stream_timeout"})


@router.get("/runs/{run_id}/events")
async def events(run_id: str, authorization: str = Header(""), x_run_key: str = Header("")) -> StreamingResponse:
    run = _run_for(run_id, authorization, x_run_key)
    return StreamingResponse(stream_run(run_id, run.token), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
