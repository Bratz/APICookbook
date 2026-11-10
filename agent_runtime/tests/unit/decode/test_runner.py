"""The Build & run runner (APICookbook fork F1): run table, slot cap, auth and request checks.

``decode run`` is swapped for a tiny fake child (prints an answer, writes a summary), and the docker
reap is a no-op, so this runs anywhere with no daemon and no model.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from decode import runner as runner_mod

TOKEN = "t" * 40

_FAKE_DECODE = (
    "import json, os, sys, time\n"
    "args = sys.argv[1:]\n"
    "if 'sleep' in args[1]: time.sleep(30)\n"
    "summary = args[args.index('--summary-json') + 1]\n"
    "open(summary, 'w').write(json.dumps({'exit_reason': 'completed', 'task': args[1]}))\n"
    "events = open(args[args.index('--events-jsonl') + 1], 'w')\n"
    "events.write(json.dumps({'type': 'tool_call', 'tool': 'bash'}) + chr(10))\n"
    "events.write(json.dumps({'type': 'tool_result', 'tool': 'bash', 'ok': True}) + chr(10))\n"
    "events.close()\n"
    "print(('answer for ' + args[1] + ' via ' + os.environ.get('BANCS_PROXY_URL', '')).strip())\n"
)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_decode_cmd", lambda: [sys.executable, "-c", _FAKE_DECODE])
    monkeypatch.setattr(runner_mod, "reap_workers", lambda run_dir: None)
    return runner_mod.Runner(tmp_path, max_runs=1, timeout_s=20, max_requests=5)


def _wait(runner, run_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = runner.get(run_id)
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_a_run_completes_with_its_answer_and_summary(runner):
    run = runner.start("list accounts")

    body = _wait(runner, run.id)

    assert body["status"] == "completed"
    assert body["exit_code"] == 0
    assert body["answer"].strip() == "answer for list accounts via"  # no proxy given
    assert body["summary"] == {"exit_reason": "completed", "task": "list accounts"}


def test_a_start_past_the_slot_cap_is_refused(runner):
    first = runner.start("sleep a while")

    assert runner.start("second") is None  # max_runs=1: the caller gets 429 and retries

    runner._runs[first.id].process.kill()
    _wait(runner, first.id)
    assert runner.start("now there is room") is not None


@pytest.fixture
def server(runner):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), runner_mod.make_handler(runner, TOKEN))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _call(url, method="GET", body=None, token=TOKEN):
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body else None
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())


def test_health_needs_no_token_but_everything_else_does(server):
    assert _call(f"{server}/health", token=None)[0] == 200
    assert _call(f"{server}/runs", "POST", {"task": "x"}, token=None)[0] == 401
    assert _call(f"{server}/runs", "POST", {"task": "x"}, token="wrong")[0] == 401


def test_bad_requests_are_refused_before_any_run_starts(server, runner):
    assert _call(f"{server}/runs", "POST", {"task": ""})[0] == 400
    assert _call(f"{server}/runs", "POST", {"task": "x" * 9000})[0] == 400
    assert _call(f"{server}/runs", "POST", {"task": "x", "model": "a b; rm -rf /"})[0] == 400
    assert _call(f"{server}/runs/../../etc/passwd")[0] == 404
    assert runner.active() == 0


def test_a_run_round_trips_over_http(server, runner):
    code, body = _call(f"{server}/runs", "POST", {"task": "get balance"})
    assert code == 202

    _wait(runner, body["id"])
    code, result = _call(f"{server}/runs/{body['id']}")

    assert code == 200
    assert result["status"] == "completed"
    assert "get balance" in result["answer"]


def test_only_the_cookbook_proxys_per_run_address_is_passed_on(server, runner):
    good = "http://10.0.0.49:7601/r/" + "a" * 32
    for bad in (
        "http://evil.example/r/" + "a" * 32,
        good + "/../x",
        "http://10.0.0.49:7601/r/short",
    ):
        assert _call(f"{server}/runs", "POST", {"task": "x", "proxy_url": bad})[0] == 400

    code, body = _call(f"{server}/runs", "POST", {"task": "use the bank", "proxy_url": good})
    assert code == 202
    assert _wait(runner, body["id"])["answer"].strip() == f"answer for use the bank via {good}"


def test_a_runs_steps_stream_through_a_cursor(server, runner):
    code, body = _call(f"{server}/runs", "POST", {"task": "two steps"})
    _wait(runner, body["id"])

    code, first = _call(f"{server}/runs/{body['id']}/events")
    assert code == 200
    assert [e["type"] for e in first["events"]] == ["tool_call", "tool_result"]
    assert first["next"] == 2 and first["status"] == "completed"

    _, rest = _call(f"{server}/runs/{body['id']}/events?after=2")
    assert rest["events"] == [] and rest["next"] == 2  # nothing new past the cursor


def test_a_finished_runs_text_files_come_back_but_not_links_hidden_or_binary(runner):
    run = runner.start("list accounts")
    _wait(runner, run.id)
    workspace = run.directory / ".decode" / "sandbox"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "main.py").write_bytes(b"print('hi')\n")
    (workspace / "pkg" / "util.py").write_text("X = 1\n")
    (workspace / ".env").write_text("SECRET=1\n")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("[core]\n")
    (workspace / "blob.bin").write_bytes(bytes([0, 1, 2]))
    outside = run.directory / "stderr.txt"  # a host file next to the Workspace
    outside.write_text("runner secret")
    with contextlib.suppress(OSError):  # no symlink rights: the rest still checks regular files
        (workspace / "link.txt").symlink_to(outside)

    body = runner.files(run.id)

    assert [f["path"] for f in body["files"]] == ["main.py", "pkg/util.py"]
    assert body["files"][0]["content"] == "print('hi')\n"
    assert all("runner secret" not in f["content"] for f in body["files"])
    assert runner.files("0" * 32) is None


def test_files_come_back_over_http_only_after_the_run(server, runner):
    code, started = _call(f"{server}/runs", "POST", {"task": "sleep a while"})
    assert code == 202
    run = runner._runs[started["id"]]
    (run.directory / ".decode" / "sandbox").mkdir(parents=True)
    (run.directory / ".decode" / "sandbox" / "main.py").write_bytes(b"print(1)\n")

    assert _call(f"{server}/runs/{run.id}/files")[1]["files"] == []  # still running

    run.process.kill()
    _wait(runner, run.id)
    code, body = _call(f"{server}/runs/{run.id}/files")
    assert code == 200
    assert body["files"] == [{"path": "main.py", "content": "print(1)\n"}]
    assert _call(f"{server}/runs/{run.id}/files", token="wrong")[0] == 401


def test_a_follow_up_starts_from_the_files_it_is_given(server, runner):
    seed = [
        {"path": "td.py", "content": "print(1)\n"},
        {"path": "lib/client.py", "content": "X = 1\n"},
    ]

    code, started = _call(f"{server}/runs", "POST", {"task": "add a column", "files": seed})

    assert code == 202
    workspace = runner._runs[started["id"]].directory / ".decode" / "sandbox"
    assert (workspace / "td.py").read_bytes() == b"print(1)\n"
    assert (workspace / "lib" / "client.py").read_bytes() == b"X = 1\n"


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "../escape.py", "content": "x"}],
        [{"path": "/etc/cron.d/x", "content": "x"}],
        [{"path": ".bashrc", "content": "x"}],
        [{"path": "a.py"}],
        [{"path": "a.py", "content": "x" * 1_000_001}],
        "not a list",
    ],
)
def test_seed_files_outside_the_rules_are_refused_before_any_run_starts(server, runner, files):
    assert _call(f"{server}/runs", "POST", {"task": "x", "files": files})[0] == 400
    assert runner.active() == 0
