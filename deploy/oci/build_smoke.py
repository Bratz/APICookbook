"""Drive one Build & run task end to end through cookbook-api (docs/decode-fork-plan.md, F2a/F2b).

Runs INSIDE the cookbook-api container (stdlib only; .github/workflows/build-smoke.yml pipes it
in):  python - "<task>" reject|approve
Reads the run's live event stream, prints every step, and approves or rejects each held bank write
the moment it appears. Exit 0 if the run completed.
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7600/api/build"
AUTH = {"Authorization": f"Bearer {os.environ['RUNNER_TOKEN']}", "Content-Type": "application/json"}


def call(method, path, body=None, timeout=30):
    request = urllib.request.Request(BASE + path, method=method, headers=AUTH,
                                     data=json.dumps(body).encode() if body is not None else None)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {path} -> {error.code}: {error.read().decode()[:300]}")


def events(run_id):
    """(event, data) pairs from the run's server-sent events, as they arrive."""
    kind = None
    with call("GET", f"/runs/{run_id}/events", timeout=1500) as stream:
        for raw in stream:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: ") and kind:
                yield kind, json.loads(line[6:])
                kind = None


task, decision = sys.argv[1], sys.argv[2]
with call("POST", "/runs", {"task": task}) as response:
    run_id = json.loads(response.read())["id"]
print(f"run {run_id} started; held writes will be {decision.upper()}ED\n")

final = {}
for kind, data in events(run_id):
    if kind == "step":
        detail = data.get("args") if data["type"] == "tool_call" else ("ok" if data.get("ok") else "FAILED")
        print(f"  {data['type']:11} {data['tool']:17} {str(detail)[:150]}")
    elif kind == "write":
        print(f"\n  HELD   {data['method']} {data['path']}\n         {data['body'][:300]}")
        with call("POST", f"/runs/{run_id}/writes/{data['id']}", {"approve": decision == "approve"}) as r:
            print("         ->", json.loads(r.read())["decision"])
    elif kind == "write_decided":
        pass  # already printed with our decision
    elif kind == "bank_call":
        print(f"  BANK   {data['method']:6} {data['path'][:90]} {data.get('status', '-')} {data.get('decision', '')}")
    elif kind == "done":
        final = data

summary = final.get("summary") or {}
print(f"\n=== {final.get('status')} (exit {final.get('exit_code')}): {summary.get('requests')} requests, "
      f"${summary.get('cost_usd')}")
print("--- answer\n" + (final.get("answer") or "")[-3000:])
sys.exit(0 if final.get("status") == "completed" else 1)
