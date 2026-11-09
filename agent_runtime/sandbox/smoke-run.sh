#!/usr/bin/env bash
# One end-to-end run through the sandbox host's runner (decode fork F1).
# Run as root on the host, with {"task": "..."} on stdin:  sudo bash smoke-run.sh < task.json
# Exit 0 only if the run completed and left no worker container behind.
set -euo pipefail
set -a; . /etc/decode/runner.env; set +a   # RUNNER_TOKEN; never printed

url="http://$(hostname -I | awk '{print $1}'):7700"
auth="Authorization: Bearer $RUNNER_TOKEN"

id=$(curl -sf -H "$auth" -H "Content-Type: application/json" --data-binary @- "$url/runs" | jq -r .id)
echo "run $id started"
status=running
for _ in $(seq 1 100); do   # up to ~17 min, past the runner's own 15 min limit
  sleep 10
  body=$(curl -sf -H "$auth" "$url/runs/$id")
  status=$(jq -r .status <<<"$body")
  [ "$status" = running ] || break
done

jq '{status, exit_code, summary: (.summary // {} | {exit_reason, error, requests, input_tokens, output_tokens, cost_usd})}' <<<"$body"
echo "--- answer";        jq -r .answer <<<"$body"
echo "--- stderr (tail)"; jq -r .stderr_tail <<<"$body" | tail -n 20
leftover=$(docker ps -q --filter label=decode.workspace | wc -l)
echo "--- leftover workers: $leftover"
[ "$status" = completed ] && [ "$leftover" -eq 0 ]
