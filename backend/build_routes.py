"""Build & run routes (docs/decode-fork-plan.md, F2a): start a run on the sandbox host, see its held
bank writes, approve or reject them.

F2a is operator-only: every route needs ``Authorization: Bearer <RUNNER_TOKEN>``, the secret the
runner already requires. Visitors arrive in F3 (UI) behind F4's caps; until then nobody without the
token can start a run.

    POST   /api/build/runs                        {"task"}  → {"id"}
    GET    /api/build/runs/{id}                   the runner's status + held writes + the bank calls
    POST   /api/build/runs/{id}/writes/{write}    {"approve": true|false}
    DELETE /api/build/runs/{id}                   stop it
"""

from __future__ import annotations

import hmac
import os
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from backend import build_proxy

router = APIRouter(prefix="/api/build")

# runner run id -> this run's proxy token. ponytail: in memory, so a cookbook-api restart forgets
# runs in flight (their tokens die with them, which fails safe); persisted when F3 needs history.
_RUNS: dict[str, str] = {}


def _runner() -> tuple[str, str]:
    url, token = os.getenv("RUNNER_URL", "http://10.0.0.223:7700"), os.getenv("RUNNER_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="Build & run is not configured (RUNNER_TOKEN)")
    return url.rstrip("/"), token


def _authorize(authorization: str) -> None:
    _, token = _runner()
    if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


async def _call_runner(method: str, path: str, json: Any = None) -> httpx.Response:
    url, token = _runner()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            return await client.request(method, f"{url}{path}", json=json,
                                        headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"the sandbox host did not answer: {type(exc).__name__}")


def _token(run_id: str) -> str:
    token = _RUNS.get(run_id)
    if token is None:
        raise HTTPException(status_code=404, detail="no such run")
    return token


class StartRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    approve: bool


@router.post("/runs", status_code=202)
async def start(body: StartRequest, authorization: str = Header("")) -> dict[str, Any]:
    _authorize(authorization)
    proxy = build_proxy.new_run()
    response = await _call_runner("POST", "/runs",
                                  {"task": body.task, "proxy_url": build_proxy.public_base(proxy.token)})
    if response.status_code != 202:
        build_proxy.end_run(proxy.token)
        raise HTTPException(status_code=response.status_code, detail=response.json().get("error", "runner refused"))
    run_id = response.json()["id"]
    _RUNS[run_id] = proxy.token
    return {"id": run_id}


@router.get("/runs/{run_id}")
async def status(run_id: str, authorization: str = Header("")) -> dict[str, Any]:
    _authorize(authorization)
    token = _token(run_id)
    response = await _call_runner("GET", f"/runs/{run_id}")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="the runner does not know this run")
    run = response.json()
    if run.get("status") != "running":
        build_proxy.end_run(token)  # the token dies with the run: no more bank calls on it
    return {**run, "pending_writes": build_proxy.pending(token), "bank_calls": build_proxy.calls(token)}


@router.post("/runs/{run_id}/writes/{write_id}")
async def decide(run_id: str, write_id: str, body: Decision, authorization: str = Header("")) -> dict[str, Any]:
    _authorize(authorization)
    write = build_proxy.decide(_token(run_id), write_id, body.approve)
    if write is None:
        raise HTTPException(status_code=404, detail="no such pending write (decided, timed out or unknown)")
    return write.public()


@router.delete("/runs/{run_id}")
async def stop(run_id: str, authorization: str = Header("")) -> dict[str, Any]:
    _authorize(authorization)
    token = _token(run_id)
    build_proxy.end_run(token)
    response = await _call_runner("DELETE", f"/runs/{run_id}")
    return {"id": run_id, "stopping": response.status_code == 200}
