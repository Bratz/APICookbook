"""F5: Build & run on 20 BaNCS tasks with gold call sequences (docs/decode-fork-plan.md).

Runs in a one-off container from the cookbook-api image (.github/workflows/agent-eval.yml), never inside
the live one: it reads the spec, and drives the LIVE Build & run routes (http://cookbook-api:7600 on the
Docker network) with the operator token, so these runs stay out of the visitors' caps:

    python -m backend.agent_eval [--only 1,5,12] [--out /app/data/evals]

Each task has a GOLD sequence: the Swagger operations a correct solution must use, in order (each step a
set of acceptable alternatives). A run hits it when the operations its code really called (each bank
call's method and path mapped through the Swagger path templates) contain the gold sequence in order.
Every write is held and REJECTED here, so evals never change demo data; a held write still counts as
"used", and its body is checked against the operation's schema. Also recorded: whether the run
completed, whether a gold GET answered 2xx on read tasks, and each run's cost, time and requests.

Two kinds of task check the agent's judgement rather than its building: an API question ("spec") passes
when the answer names a gold operation and no bank call was made; an off-topic request passes when it
was declined without any tool call.

The code agent this was first compared against was retired in F6 after Build & run scored 20/20 twice.
"""

from __future__ import annotations

import argparse
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
    # "spec": a question ABOUT the API. Pass: the answer names a gold operation (id or path), no bank calls.
    ("spec", "Which API do I use to update a customer's mobile number, and what does its request body look like?",
     [{"RSTBTAmendPhonenumberUsingPUT", "BRBT0303ModContactDtls_POSTUsingPUT"}]),
    ("spec", "What parameters does the account balance API take?",
     [{"CBPETGetAccountBalanceUsingGET", "AMET0395GetAcntUsrBlncsUsingGET"}]),
    ("spec", "Is there an API to stop payment on a cheque? What does it need?",
     [{"RSTBT11810StopPymntChqUsingPOST", "RSBTCreateStopChequeRequest_POSTUsingPOST"}]),
    ("spec", "How do I get the repayment schedule of a loan through the API?", [{"CRET2811GetRpymntDtls_POSTUsingGET"}]),
    # "offtopic": nothing to do with BaNCS. Pass: declined with no tool calls and no bank calls.
    ("offtopic", "Write a quicksort function in Go.", []),
    ("offtopic", "What is the capital of France?", []),
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


def names_an_operation(answer: str, gold: list[set[str]], endpoints: dict[str, dict[str, Any]]) -> bool:
    """A spec answer passes when it names a gold operation: its id, or its Swagger path."""
    for op in set().union(*gold) if gold else set():
        path = (endpoints.get(op) or {}).get("url", "")
        if op in answer or (path and path in answer):
            return True
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
    ok: Optional[bool] = None  # the run completed
    gold_get_2xx: Optional[bool] = None  # Build & run, read tasks: a gold GET answered 2xx
    write_bodies_valid: Optional[bool] = None  # Build & run: every held write matched its schema
    tool_calls: int = 0
    answer: str = ""  # the start of the answer, for reading misses
    note: str = ""


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
                    elif event == "step" and data.get("type") == "tool_call":
                        r.tool_calls += 1
                    elif event == "done":
                        final = data
                        break
                    event = None
    used = [matcher.operations(c["method"], c["path"]) for c in calls]
    r.used = [sorted(u) or [f"{c['method']} {c['path']}"] for u, c in zip(used, calls)]
    r.answer = (final.get("answer") or "")[:600]
    if kind == "spec":  # answered from the spec, without touching the bank
        r.gold_hit = not calls and names_an_operation(r.answer, gold, kb.endpoints)
    elif kind == "offtopic":  # declined without using any tool
        r.gold_hit = not calls and r.tool_calls == 0
    else:
        r.gold_hit = in_order(used, gold)
    r.ok = final.get("status") == "completed"
    gold_ops = set().union(*gold) if gold else set()
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
    """The summary table, then one row per task."""
    def rate(attr: str, kind: Optional[str] = None) -> str:
        vals = [getattr(x, attr) for x in results if kind in (None, x.kind) and getattr(x, attr) is not None]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "-"

    usd, secs = sum(x.usd for x in results), sorted(x.seconds for x in results)
    cost = f"${usd / len(results):.3f}/task (${usd:.2f}), median {secs[len(secs) // 2]:.0f} s" if results else "-"
    lines = ["| | Build & run |", "|---|---|",
             f"| Gold sequence hit, all tasks | {rate('gold_hit')} |",
             f"| … read tasks | {rate('gold_hit', 'read')} |",
             f"| … write tasks | {rate('gold_hit', 'write')} |",
             f"| … API questions (answered from the spec, no bank calls) | {rate('gold_hit', 'spec')} |",
             f"| … off-topic (declined, no tools) | {rate('gold_hit', 'offtopic')} |",
             f"| Run completed | {rate('ok')} |",
             f"| A gold GET answered 2xx (read tasks) | {rate('gold_get_2xx')} |",
             f"| Held write bodies valid against the schema | {rate('write_bodies_valid')} |",
             f"| Cost and time | {cost} |",
             "", "| # | Kind | Task | Build & run |", "|---|---|---|---|"]
    for n, (kind, task, _gold) in enumerate(TASKS, 1):
        x = next((x for x in results if x.task == n), None)
        cell = "-" if x is None else ("✅" if x.gold_hit else "❌") + (f" {x.note}" if x.note else "")
        lines.append(f"| {n} | {kind} | {task} | {cell} |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="task numbers, e.g. 1,12")
    parser.add_argument("--out", default="/app/data/evals")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    from backend.knowledge_base import UnifiedAPIKnowledgeBase

    kb = UnifiedAPIKnowledgeBase()
    await kb.load_specifications()
    matcher = OperationMatcher(kb.endpoints)
    missing = sorted(op for _, _, gold in TASKS for step in gold for op in step if op not in kb.endpoints)
    if missing:
        print("gold operations not in the spec:", ", ".join(missing), file=sys.stderr)
    picked = [int(x) for x in args.only.split(",") if x] or list(range(1, len(TASKS) + 1))
    token = os.environ["RUNNER_TOKEN"]
    slots = asyncio.Semaphore(BUILD_CONCURRENCY)

    async def one(n: int) -> Result:
        async with slots:
            kind, task, gold = TASKS[n - 1]
            res = await eval_build(n, kind, task, gold, kb, matcher, token)
            print(f"build {n:2}: {'hit ' if res.gold_hit else 'miss'} ${res.usd:.3f} {res.seconds:.0f}s {res.note}",
                  flush=True)
            return res

    results = list(await asyncio.gather(*(one(n) for n in picked)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"eval-{stamp}.json").write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    print("\n" + report(results))


if __name__ == "__main__":
    asyncio.run(main())
