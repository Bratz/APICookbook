"""F5: the code agent and Build & run on the same BaNCS tasks, scored the same way.

Runs in a one-off container from the cookbook-api image (.github/workflows/agent-eval.yml), never inside
the live one: it loads its own knowledge base and RAG index for the code agent, and drives the LIVE
Build & run routes (http://cookbook-api:7600 on the Docker network) with the operator token:

    python -m backend.agent_eval [--agents code,build] [--only 1,5,12] [--out /app/data/evals]

Each task has a GOLD sequence: the Swagger operations a correct solution must use, in order (each
step a set of acceptable alternatives). Both agents are scored on whether they hit the gold sequence:

- **code agent:** the endpoints named in its plan's steps (its Python must also parse);
- **Build & run:** the operations its code really called, mapped from each bank call's method and path
  through the Swagger path templates. Every write is held and REJECTED here, so evals never change
  demo data; a held write still counts as "used" (the agent got the operation right) and its body is
  checked against the operation's schema.

Also recorded: the cost, time and requests of each answer, and for Build & run whether the run
completed and whether a gold GET returned 2xx (its code really worked against the bank).

ponytail: no LLM judge yet; gold sequences and real execution decide. Add one if scores disagree with
reading the answers.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import jsonschema

TASKS: list[tuple[str, str, list[set[str]]]] = [
    # (kind, task, gold) — kind: "read" (GETs only) or "write" (needs a change; held and rejected)
    ("read", "Check the balance of an account.",
     [{"CBPETGetAccountBalanceUsingGET", "AMET0395GetAcntUsrBlncsUsingGET"}]),
    ("read", "List all accounts of a customer with their balances.",
     [{"CBCET0101FetchCustAcntsUsingGET", "AMET0395GetAcntUsrBlncsUsingGET"}]),
    ("read", "See all the blocks placed on an account.", [{"AMET0117FtchBlckLstUsingGET"}]),
    ("read", "Show the repayment schedule of a loan.", [{"CRET2811GetRpymntDtls_POSTUsingGET"}]),
    ("read", "How much does the customer still owe to close their loan?", [{"CRET6827FetchLoanWriteoffDtlsUsingGET"}]),
    ("read", "List fund transfers.", [{"AMEET1612FetchAMETxnList_POSTUsingGET_1"}]),
    ("read", "What is the current currency exchange rate between two currencies?",
     [{"CBCET1401GetEXRTUsingGET", "MIET1210ListOthrPrpsEXRTUsingGET"}]),
    ("read", "What interest rate applies to a loan product?", [{"CBCET0333GetLatestINRT_POSTUsingGET"}]),
    ("read", "Show an account's details including its address.", [{"AMET2020GetAcntDetails_wrapperUsingGET"}]),
    ("read", "Look up a customer's profile.", [{"RSTETViewCustomerDetailsUsingGET", "CBPET0106GetBPDetailsUsingGET"}]),
    ("read", "Check an account's balance and then list the blocks placed on that account.",
     [{"CBPETGetAccountBalanceUsingGET", "AMET0395GetAcntUsrBlncsUsingGET"}, {"AMET0117FtchBlckLstUsingGET"}]),
    ("write", "Put a hold of 500 on an account so the money can't be withdrawn.",
     [{"AMBTCrtCashBlockWrapperUsingPOST", "INTBT0118CrtCshBlckIntf_POSTUsingPOST"}]),
    ("write", "Cancel a cash block on an account.", [{"AMBTCancelCashBlockWrapperUsingDELETE"}]),
    ("write", "A customer moved house: update their postal address.",
     [{"RSTBTAmendPostalAddressUsingPUT", "BRBTCreateModifyCommDtlsUsingPUT"}]),
    ("write", "Update a customer's mobile number.",
     [{"RSTBTAmendPhonenumberUsingPUT", "BRBT0303ModContactDtls_POSTUsingPUT"}]),
    ("write", "Transfer money between two accounts.",
     [{"AMEBTFundTransferWrapperIB_POSTUsingPOST", "AMEBT2021AmeWrapper_POSTUsingPOST"}]),
    ("write", "Order a new cheque book for an account.",
     [{"CBCBTCreateChequeBookRequestUsingPOST", "RSBTCreateChqRequestOnRlsd_POSTUsingPOST"}]),
    ("write", "Stop payment on a cheque.",
     [{"RSTBT11810StopPymntChqUsingPOST", "RSBTCreateStopChequeRequest_POSTUsingPOST"}]),
    ("write", "Request an account statement.",
     [{"BRBTRequestAcntStatmntUsingPOST", "AMET2021AcntsBalStatmnt_POSTUsingPOST"}]),
    ("write", "Look up a customer's profile, then update their mobile number.",
     [{"RSTETViewCustomerDetailsUsingGET", "CBPET0106GetBPDetailsUsingGET"},
      {"RSTBTAmendPhonenumberUsingPUT", "BRBT0303ModContactDtls_POSTUsingPUT"}]),
]

BUILD_URL = os.getenv("EVAL_BUILD_URL", "http://cookbook-api:7600/api/build")  # the LIVE api
BUILD_CONCURRENCY = 2  # the sandbox host's slots


# ---------------------------------------------------------------- pure scoring (unit-tested)

def in_order(used: list[set[str]], gold: list[set[str]]) -> bool:
    """Each gold step is matched by some later-used operation, in the gold order (extra calls allowed)."""
    i = 0
    for want in gold:
        while i < len(used) and not (used[i] & want):
            i += 1
        if i == len(used):
            return False
        i += 1
    return True


class OperationMatcher:
    """Maps a real call (method, path) to the Swagger operations it could be."""

    def __init__(self, endpoints: dict[str, dict[str, Any]], base_path: str = "/Core"):
        self.base = base_path
        self.routes = []
        for name, ep in endpoints.items():
            if ep.get("source") != "swagger" or not ep.get("url"):
                continue
            pattern = "^" + re.sub(r"\\\{[^/]+?\\\}", "[^/]+", re.escape(ep["url"].rstrip("/"))) + "/?$"
            self.routes.append((ep["method"].upper(), re.compile(pattern), name))

    def operations(self, method: str, path: str) -> set[str]:
        path = path.split("?", 1)[0]
        if path.startswith(self.base + "/"):
            path = path[len(self.base):]
        return {name for m, rx, name in self.routes if m == method.upper() and rx.match(path)}


def python_parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def body_valid(schema: Optional[dict], body: str) -> Optional[bool]:
    """Whether a held write's body satisfies the operation's schema; None when there is nothing to check."""
    if not schema:
        return None
    try:
        jsonschema.Draft4Validator(schema).validate(json.loads(body or "null"))
        return True
    except (ValueError, jsonschema.ValidationError, jsonschema.SchemaError):
        return False


# ---------------------------------------------------------------- one task, two agents

@dataclass
class Result:
    agent: str
    task: int
    kind: str
    gold_hit: bool = False
    used: list[list[str]] = field(default_factory=list)
    usd: float = 0.0
    seconds: float = 0.0
    requests: Optional[int] = None
    ok: Optional[bool] = None  # code agent: Python parses · Build & run: run completed
    gold_get_2xx: Optional[bool] = None  # Build & run, read tasks: a gold GET answered 2xx
    write_bodies_valid: Optional[bool] = None  # Build & run: every held write matched its schema
    note: str = ""


async def eval_code_agent(n: int, kind: str, task: str, gold: list[set[str]], kb, rag) -> Result:
    from pydantic_ai import capture_run_messages

    from backend import code_agent

    r = Result(agent="code", task=n, kind=kind)
    started = time.time()
    with capture_run_messages() as messages:
        try:
            result, _deps = await code_agent.answer(task, "python", kb, rag)
            out = result.output
            if getattr(out, "kind", "") == "plan":
                r.used = [[s.endpoint] for s in out.steps]
                r.ok = python_parses(out.code)
            else:
                r.note = "asked a clarifying question"
                r.ok = False
            r.requests = result.usage().requests
        except Exception as exc:  # a failed answer scores 0, it does not stop the eval
            r.note = f"{type(exc).__name__}: {exc}"[:200]
            r.ok = False
        r.usd = round(code_agent.usd(messages), 5)
    r.seconds = round(time.time() - started, 1)
    r.gold_hit = in_order([set(u) for u in r.used], gold)
    return r


async def eval_build(n: int, kind: str, task: str, gold: list[set[str]], kb, matcher: OperationMatcher,
                     token: str) -> Result:
    r = Result(agent="build", task=n, kind=kind)
    started = time.time()
    auth = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=1500)) as client:
        for attempt in range(60):  # both sandboxes busy: wait for a slot
            response = await client.post(f"{BUILD_URL}/runs", json={"task": task}, headers=auth)
            if response.status_code != 429:
                break
            await asyncio.sleep(15)
        if response.status_code != 202:
            r.note = f"start failed: {response.status_code} {response.text[:150]}"
            return r
        run_id = response.json()["id"]
        calls, writes, final = [], [], {}
        async with client.stream("GET", f"{BUILD_URL}/runs/{run_id}/events", headers=auth) as stream:
            event = None
            async for line in stream.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event:
                    data = json.loads(line[6:])
                    if event == "write":  # evals never change demo data
                        writes.append(data)
                        await client.post(f"{BUILD_URL}/runs/{run_id}/writes/{data['id']}",
                                          json={"approve": False}, headers=auth)
                    elif event == "bank_call":
                        calls.append(data)
                    elif event == "done":
                        final = data
                        break
                    event = None
    used = [matcher.operations(c["method"], c["path"]) for c in calls]
    r.used = [sorted(u) or [f"{c['method']} {c['path']}"] for u, c in zip(used, calls)]
    r.gold_hit = in_order(used, gold)
    r.ok = final.get("status") == "completed"
    gold_ops = set().union(*gold)
    if kind == "read":
        r.gold_get_2xx = any(c["method"] == "GET" and 200 <= int(c.get("status") or 0) < 300 and (u & gold_ops)
                             for u, c in zip(used, calls))
    checks = []
    for w in writes:
        for op in matcher.operations(w["method"], w["path"]):
            ok = body_valid(kb.resolved_body_schema(op), w.get("body") or "")
            if ok is not None:
                checks.append(ok)
                break
    r.write_bodies_valid = all(checks) if checks else None
    summary = final.get("summary") or {}
    r.usd = float(summary.get("cost_usd") or 0)
    r.requests = summary.get("requests")
    r.seconds = round(time.time() - started, 1)
    if not r.ok:
        r.note = f"run ended: {final.get('status')}"
    return r


# ---------------------------------------------------------------- the report

def report(results: list[Result]) -> str:
    """The side-by-side table, then one row per task."""
    by = {a: [x for x in results if x.agent == a] for a in ("code", "build")}

    def rate(rows: list[Result], attr: str) -> str:
        vals = [getattr(x, attr) for x in rows if getattr(x, attr) is not None]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "-"

    def cost(rows: list[Result]) -> str:
        if not rows:
            return "-"
        usd, secs = sum(x.usd for x in rows), sorted(x.seconds for x in rows)
        return f"${usd / len(rows):.3f}/task (${usd:.2f}), median {secs[len(secs) // 2]:.0f} s"

    lines = ["| | Code agent | Build & run |", "|---|---|---|"]
    for label, kind in (("Gold sequence hit, all tasks", None), ("… read tasks", "read"), ("… write tasks", "write")):
        cells = [rate([x for x in by[a] if kind in (None, x.kind)], "gold_hit") for a in ("code", "build")]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines += [
        f"| Code parses (code) · run completed (build) | {rate(by['code'], 'ok')} | {rate(by['build'], 'ok')} |",
        f"| A gold GET answered 2xx (read tasks) | - | {rate(by['build'], 'gold_get_2xx')} |",
        f"| Held write bodies valid against the schema | - | {rate(by['build'], 'write_bodies_valid')} |",
        f"| Cost and time | {cost(by['code'])} | {cost(by['build'])} |",
        "", "| # | Kind | Task | Code agent | Build & run |", "|---|---|---|---|---|",
    ]
    for n, (kind, task, _gold) in enumerate(TASKS, 1):
        cells = []
        for a in ("code", "build"):
            x = next((x for x in by[a] if x.task == n), None)
            cells.append("-" if x is None else ("✅" if x.gold_hit else "❌") + (f" {x.note}" if x.note else ""))
        lines.append(f"| {n} | {kind} | {task} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="code,build")
    parser.add_argument("--only", default="", help="task numbers, e.g. 1,12")
    parser.add_argument("--out", default="/app/data/evals")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    os.environ["BUILD_PROXY_PORT"] = "0"  # this process only reads the spec; the live api serves the proxy

    from backend.main import app

    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app):  # loads the KB and RAG
        kb, rag = app.state.kb, app.state.rag_engine
    matcher = OperationMatcher(kb.endpoints)
    missing = sorted(op for _, _, gold in TASKS for step in gold for op in step if op not in kb.endpoints)
    if missing:
        print("gold operations not in the spec:", ", ".join(missing), file=sys.stderr)
    picked = [int(x) for x in args.only.split(",") if x] or list(range(1, len(TASKS) + 1))
    agents = args.agents.split(",")
    results: list[Result] = []

    if "code" in agents:
        for n in picked:
            kind, task, gold = TASKS[n - 1]
            results.append(await eval_code_agent(n, kind, task, gold, kb, rag))
            print(f"code  {n:2}: {'hit ' if results[-1].gold_hit else 'miss'} ${results[-1].usd:.3f}", flush=True)
    if "build" in agents:
        token = os.environ["RUNNER_TOKEN"]
        slots = asyncio.Semaphore(BUILD_CONCURRENCY)

        async def one(n: int) -> Result:
            async with slots:
                kind, task, gold = TASKS[n - 1]
                res = await eval_build(n, kind, task, gold, kb, matcher, token)
                print(f"build {n:2}: {'hit ' if res.gold_hit else 'miss'} ${res.usd:.3f} {res.seconds:.0f}s {res.note}",
                      flush=True)
                return res

        results += await asyncio.gather(*(one(n) for n in picked))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"eval-{stamp}.json").write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    print("\n" + report(results))


if __name__ == "__main__":
    asyncio.run(main())
