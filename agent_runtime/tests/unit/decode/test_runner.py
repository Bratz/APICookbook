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


# ------------------------------------------------------------------ previews: a built app, tried in its sandbox
NL, TAB = chr(10).encode(), chr(9).encode()
APP = [
    {"path": "app.py", "content": "print(1)"},
    {"path": "templates/index.html", "content": "<p>hi</p>"},
]


class FakeDocker:
    """Stands in for the docker CLI: a container id for ``run``, an HTTP answer for a relayed request."""

    def __init__(self):
        self.calls, self.ready, self.body = [], True, b'{"ok": true}'

    def __call__(self, *args, stdin=None, timeout=60):
        import subprocess

        self.calls.append((args, stdin))
        out, code = b"", 0
        if args[0] == "run":
            out = b"c0ffee" + NL
        elif "curl" in args and "-w" in args:
            out = self.body + NL + b"200" + TAB + b"application/json"
        elif "curl" in args:
            code = 0 if self.ready else 7
        elif "tail" in args:
            out = b"ModuleNotFoundError: No module named 'flask_x'"
        return subprocess.CompletedProcess(args, code, out, b"")


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(runner_mod, "_docker", fake)
    monkeypatch.setattr(runner_mod, "_preview_run_args", lambda workspace: ["run", "-d", "image"])
    monkeypatch.setattr(runner_mod, "_PREVIEW_START_WAIT_S", 0.6)
    return fake


def test_a_preview_starts_the_app_in_a_worker_and_relays_requests_inside_it(runner, docker):
    preview = runner.start_preview(APP, "python app.py", 5000)

    workspace = preview.directory / runner_mod.settings.sandbox_workspace_dir
    assert (workspace / "templates/index.html").read_text() == "<p>hi</p>"
    started = next(args for args, _ in docker.calls if args[:2] == ("exec", "-d"))
    assert started[2] == "c0ffee" and "exec python app.py" in started[-1]

    answer = runner.preview_http(preview.id, "GET", "/api/x?y=1", "", b"")
    assert answer["status"] == 200 and answer["content_type"] == "application/json"
    assert json.loads(__import__("base64").b64decode(answer["body_b64"])) == {"ok": True}
    relayed, _ = docker.calls[-1]
    # made INSIDE the container, on its loopback: no published port, and the path is one argv element
    assert (
        relayed[:3] == ("exec", "-i", "c0ffee") and relayed[-1] == "http://127.0.0.1:5000/api/x?y=1"
    )

    runner.preview_http(preview.id, "POST", "/api/budgets", "application/json", b'{"a": 1}')
    posted, stdin = docker.calls[-1]
    assert (
        stdin == b'{"a": 1}'
        and "--data-binary" in posted
        and "Content-Type: application/json" in posted
    )


def test_a_preview_whose_app_never_answers_reports_its_log_and_leaves_nothing(runner, docker):
    docker.ready = False

    result = runner.start_preview(APP, "python app.py", 5000)

    assert "No module named 'flask_x'" in result  # the reason, for the visitor
    assert any(args[:2] == ("rm", "-f") for args, _ in docker.calls)
    assert runner._previews == {} and not list(runner.runs_dir.iterdir())


def test_previews_have_their_own_slot_cap_and_an_end(runner, docker, monkeypatch):
    first = runner.start_preview(APP, "python app.py", 5000)
    second = runner.start_preview(APP, "python app.py", 5001)
    assert runner.start_preview(APP, "python app.py", 5002) is None  # MAX_PREVIEWS

    assert runner.stop_preview(first.id) and not first.directory.exists()
    assert runner.preview_http(first.id, "GET", "/", "", b"") is None  # stopped
    assert not runner.stop_preview(first.id)

    second.expires = time.time() - 1  # its 10 minutes are over
    assert runner.preview_http(second.id, "GET", "/", "", b"") is None
    assert runner.start_preview(APP, "python app.py", 5003) is not None  # the expired one made room


def test_previews_over_http_need_the_token_and_a_well_formed_request(server, runner, docker):
    assert _call(f"{server}/previews", "POST", {"files": APP}, token=None)[0] == 401
    good = {"files": APP, "command": "python app.py", "port": 5000}
    for bad in (
        {**good, "files": []},
        {**good, "command": "python app.py" + chr(10) + "curl evil"},
        {**good, "port": 80},
        {**good, "files": [{"path": "../x", "content": ""}]},
    ):
        assert _call(f"{server}/previews", "POST", bad)[0] == 400
    assert runner._previews == {}

    code, made = _call(f"{server}/previews", "POST", good)
    assert code == 201 and made["expires_in"] == 600
    relay = f"{server}/previews/{made['id']}/http"

    assert _call(relay, "POST", {"method": "GET", "path": "/api/x"})[1]["status"] == 200
    assert _call(relay, "POST", {"method": "GET", "path": "/a b"})[0] == 400  # not a clean path
    assert _call(relay, "POST", {"method": "TRACE", "path": "/"})[0] == 400
    assert (
        _call(f"{server}/previews/{'0' * 32}/http", "POST", {"method": "GET", "path": "/"})[0]
        == 404
    )
    assert _call(f"{server}/previews/{made['id']}", "DELETE")[0] == 200
    assert _call(relay, "POST", {"method": "GET", "path": "/"})[0] == 404


def test_a_build_with_no_server_gets_a_worker_and_no_app(runner, docker):
    preview = runner.start_preview(APP, None, None)

    assert preview.port is None and preview.id in runner._previews
    assert not any(args[:2] == ("exec", "-d") for args, _ in docker.calls)  # nothing to start
    answer = runner.preview_http(preview.id, "GET", "/", "", b"")
    assert answer["status"] == 502  # there is no app to ask


ECHO_SHELL = "import sys; [sys.stdout.write('got ' + line.strip() + chr(10)) for line in sys.stdin]"


@pytest.fixture
def shell(monkeypatch):
    """A stand-in for the shell in the container: it answers each line typed into it."""
    import subprocess

    opened = []

    def open_shell(container):
        opened.append(container)
        return subprocess.Popen(
            [sys.executable, "-u", "-c", ECHO_SHELL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        )  # fmt: skip

    monkeypatch.setattr(runner_mod, "_open_shell", open_shell)
    return opened


def _text(answer):
    return __import__("base64").b64decode(answer["data_b64"]).decode()


def test_a_preview_terminal_takes_input_and_gives_output_by_offset(runner, docker, shell):
    preview = runner.start_preview(APP, None, None)

    first = runner.terminal(preview.id, b"hello" + NL, 0, 10)
    assert shell == ["c0ffee"]  # the shell runs in the preview's own container
    assert first["from"] == 0 and "got hello" in _text(first) and first["alive"]
    seen = len(_text(first).encode())

    quiet = runner.terminal(preview.id, b"", seen, 0.05)  # nothing new: an empty read, not a repeat
    assert quiet == {"from": seen, "data_b64": "", "alive": True}
    more = runner.terminal(preview.id, b"again" + NL, seen, 10)
    assert more["from"] == seen and _text(more).strip() == "got again"
    assert "got hello" in _text(runner.terminal(preview.id, b"", 0, 0))  # a reader that starts over

    old = preview.terminal.process
    old.stdin.close()  # the shell exits (the visitor typed "exit")
    old.wait(timeout=10)
    assert runner.terminal(preview.id, b"new" + NL, 0, 0)["alive"]  # the next use opens a new one
    assert len(shell) == 2

    process = preview.terminal.process
    runner.stop_preview(preview.id)
    assert process.wait(timeout=10) is not None  # stopped with the preview
    assert runner.terminal(preview.id, b"", 0, 0) is None
    deadline = time.time() + 10
    while not (old.stdout.closed and process.stdout.closed) and time.time() < deadline:
        time.sleep(0.02)
    assert old.stdout.closed and process.stdout.closed  # no pipe is left open


def test_the_terminal_over_http_needs_a_well_formed_request(server, runner, docker, shell):
    _, made = _call(f"{server}/previews", "POST", {"files": APP})  # no command, no port: a terminal
    term = f"{server}/previews/{made['id']}/term"

    code, answer = _call(term, "POST", {"input_b64": "aGkK", "seen": 0, "wait_ms": 10_000})
    assert code == 200 and "got hi" in _text(answer)
    for bad in ({"input_b64": "!!"}, {"seen": -1}, {"seen": "0"}, {"wait_ms": "1"}):
        assert _call(term, "POST", bad)[0] == 400
    assert _call(term, "POST", {}, token=None)[0] == 401
    assert _call(f"{server}/previews/{'0' * 32}/term", "POST", {})[0] == 404
    assert _call(f"{server}/previews", "POST", {"files": APP, "command": "python app.py"})[0] == 400


@pytest.mark.skipif(sys.platform == "win32", reason="the docker backend is POSIX-only")
def test_a_preview_worker_gets_the_same_isolation_as_a_runs_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod.settings, "sandbox_docker_runtime", "runsc")
    monkeypatch.setattr(runner_mod.settings, "sandbox_docker_network", "bancs-only")
    monkeypatch.setattr(runner_mod.settings, "sandbox_docker_hardened", True)

    args = runner_mod._preview_run_args(tmp_path)

    assert args[args.index("--runtime") : args.index("--runtime") + 2] == ["--runtime", "runsc"]
    assert args[args.index("--network") : args.index("--network") + 2] == [
        "--network",
        "bancs-only",
    ]
    assert "--read-only" in args and "--cap-drop" in args
    assert not any(a in ("-p", "--publish", "-P") for a in args)  # nothing is published
