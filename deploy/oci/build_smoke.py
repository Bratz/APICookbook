"""Drive one Build & run task end to end through cookbook-api (docs/decode-fork-plan.md, F2a).

Runs INSIDE the cookbook-api container (stdlib only; .github/workflows/build-smoke.yml pipes it
in):  python - "<task>" reject|approve
Every held bank write is printed, then approved or rejected as asked. Exit 0 if the run completed.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7600/api/build"
AUTH = {"Authorization": f"Bearer {os.environ['RUNNER_TOKEN']}", "Content-Type": "application/json"}


def call(method, path, body=None):
    request = urllib.request.Request(BASE + path, method=method, headers=AUTH,
                                     data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {path} -> {error.code}: {error.read().decode()[:300]}")


task, decision = sys.argv[1], sys.argv[2]
run_id = call("POST", "/runs", {"task": task})["id"]
print(f"run {run_id} started; held writes will be {decision.upper()}ED")
decided = set()
for _ in range(200):  # ~17 min, past the runner's own limit
    time.sleep(5)
    status = call("GET", f"/runs/{run_id}")
    for write in status["pending_writes"]:
        if write["id"] not in decided:
            decided.add(write["id"])
            print(f"\n--- held write {write['id']}: {write['method']} {write['path']}\n{write['body'][:600]}")
            print(" ->", call("POST", f"/runs/{run_id}/writes/{write['id']}", {"approve": decision == "approve"})["decision"])
    if status["status"] != "running":
        break

summary = status.get("summary") or {}
print(f"\n=== {status['status']} (exit {status['exit_code']}): {summary.get('requests')} requests, "
      f"${summary.get('cost_usd')}")
print("--- bank calls")
for c in status["bank_calls"]:
    print(f"  {c['method']:6} {c['path'][:90]:90} {c.get('status', '-')} {c.get('decision', '')}")
print("--- answer\n" + status["answer"][-3000:])
sys.exit(0 if status["status"] == "completed" else 1)
