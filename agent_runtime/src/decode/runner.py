"""The Build & run runner on the sandbox host (APICookbook fork F1; docs/decode-fork-plan.md).

A small private HTTP service that cookbook-api calls to run ``decode run`` for a visitor's task.
Each run is its own ``decode run`` subprocess in its own directory (its harness home, so its session
log and Workspace are separate), because decode keeps per-process state (the bash executor memo).

    POST   /runs          {"task", "model"?, "proxy_url"?}  → 202 {"id"} · 429 when all slots are busy
    GET    /runs/<id>     → {"id", "status", "exit_code", "answer", "summary", "stderr_tail"}
    GET    /runs/<id>/events?after=N  → {"status", "events": [...from line N], "next"}  (F2b)
    GET    /runs/<id>/files  → {"status", "files": [{"path", "content"}], "skipped"}  (text files the agent
           wrote; empty while the run is still going)
    DELETE /runs/<id>     → stop it (process group killed, its worker container removed)
    GET    /health        → {"ok", "active", "max"}  (no token: a liveness probe)

Every other call needs ``Authorization: Bearer <RUNNER_TOKEN>``. The host firewall also admits only
cookbook-api's address. The Anthropic key lives in this process's env and is inherited by
``decode run``; the worker containers never see it (decode passes them no env but an opt-in git
token, which is unset here).

ponytail: stdlib ``http.server`` + one thread per run. A start past ``RUNNER_MAX_RUNS`` gets 429 and
cookbook-api retries, which is the queue. Live per-step events arrive in F2 with approvals.
"""

from decode.logging import init_logger

init_logger()

import hmac  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

from decode.config.settings import settings  # noqa: E402

logger = logging.getLogger(__name__)

MAX_TASK_CHARS = 4000
_STOP_GRACE_S = 5.0
_KEEP_RUN_DIRS_S = 86_400.0
_TAIL_CHARS = 4000
_RUN_PATH = re.compile(r"^/runs/([0-9a-f]{32})$")
_EVENTS_PATH = re.compile(r"^/runs/([0-9a-f]{32})/events(?:\?after=(\d{1,6}))?$")
_MAX_EVENTS_PER_READ = 200
_FILES_PATH = re.compile(r"^/runs/([0-9a-f]{32})/files$")
# ponytail: caps sized for scripts, not projects; a zip of the raw folder when someone needs more
_MAX_FILES, _MAX_FILE_BYTES, _MAX_FILES_BYTES = 50, 200_000, 1_000_000
_SKIP_DIRS = {"__pycache__", "node_modules", "venv"}


@dataclass
class Run:
    id: str
    directory: Path
    process: subprocess.Popen[bytes]
    status: str = "running"  # running | completed | failed | timeout | stopped
    exit_code: int | None = None
    stopped: bool = field(default=False, repr=False)


class Runner:
    """The run table and its lifecycle; the HTTP handler is a thin shell over it."""

    def __init__(self, runs_dir: Path, *, max_runs: int, timeout_s: float, max_requests: int):
        self.runs_dir = runs_dir
        self.max_runs = max_runs
        self.timeout_s = timeout_s
        self.max_requests = max_requests
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def active(self) -> int:
        with self._lock:
            return sum(run.status == "running" for run in self._runs.values())

    def start(self, task: str, model: str | None = None, proxy_url: str = "") -> Run | None:
        """Start a run, or return ``None`` when every slot is busy."""
        with self._lock:
            if sum(run.status == "running" for run in self._runs.values()) >= self.max_runs:
                return None
            self._prune_old_dirs()
            run_id = uuid.uuid4().hex
            directory = self.runs_dir / run_id
            directory.mkdir(parents=True)
            argv = [*_decode_cmd(), "run", task, "--max-requests", str(self.max_requests)]
            argv += ["--summary-json", str(directory / "summary.json")]
            argv += ["--events-jsonl", str(directory / "events.jsonl")]
            if model:
                argv += ["--model", model]
            with (
                (directory / "answer.txt").open("wb") as out,
                (directory / "stderr.txt").open("wb") as err,
            ):
                process = subprocess.Popen(
                    argv,
                    cwd=directory,  # the run's harness home: its sessions and Workspace live here
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    env={
                        **os.environ,
                        "DECODE_LOG_FILE": str(directory / "decode.log"),
                        "BANCS_PROXY_URL": proxy_url,  # this run's proxy address (F2a)
                    },
                    start_new_session=True,  # its own process group, so a stop kills the whole tree
                )
            run = Run(id=run_id, directory=directory, process=process)
            self._runs[run_id] = run
        threading.Thread(target=self._watch, args=(run,), daemon=True).start()
        logger.info("[runner] started %s (pid %s)", run_id, process.pid)
        return run

    def stop(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None:
            return False
        if run.status == "running":
            run.stopped = True
            _kill_group(run.process)
        return True

    def get(self, run_id: str) -> dict[str, object] | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        summary_path = run.directory / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        )
        return {
            "id": run.id,
            "status": run.status,
            "exit_code": run.exit_code,
            "answer": _read(run.directory / "answer.txt"),
            "summary": summary,
            "stderr_tail": _read(run.directory / "stderr.txt")[-_TAIL_CHARS:],
        }

    def events(self, run_id: str, after: int = 0) -> dict[str, object] | None:
        """The run's step log from line ``after`` on (a cursor: pass back ``next``)."""
        run = self._runs.get(run_id)
        if run is None:
            return None
        # Status first: a run that is already finished has written its last line.
        status = run.status
        lines = _read(run.directory / "events.jsonl").splitlines()
        page = lines[after : after + _MAX_EVENTS_PER_READ]
        events = []
        for line in page:
            try:
                events.append(json.loads(line))
            except ValueError:
                break  # a line still being written: stop here and send it next time
        return {"status": status, "events": events, "next": after + len(events)}

    def files(self, run_id: str) -> dict[str, object] | None:
        """The text files the agent left in its Workspace, once the run has finished.

        Read on the host, so: only after the run (its worker container is gone, nothing can swap a file),
        only regular files (a symlink could point at this process's secrets), no hidden paths.
        """
        run = self._runs.get(run_id)
        if run is None:
            return None
        if run.status == "running":
            return {"status": run.status, "files": [], "skipped": 0}
        root = run.directory / settings.sandbox_workspace_dir
        files: list[dict[str, str]] = []
        skipped = total = 0
        if root.is_dir() and not root.is_symlink():
            # os.walk does not follow links (followlinks=False): a linked folder is not entered
            for here, dirs, names in os.walk(root):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS)
                for name in sorted(names):
                    path = Path(here) / name
                    info = path.lstat()
                    if name.startswith(".") or not stat.S_ISREG(info.st_mode):
                        skipped += 1
                        continue
                    if (
                        len(files) >= _MAX_FILES
                        or info.st_size > _MAX_FILE_BYTES
                        or total + info.st_size > _MAX_FILES_BYTES
                    ):
                        skipped += 1
                        continue
                    data = path.read_bytes()
                    if b"\0" in data:  # binary: not code to show
                        skipped += 1
                        continue
                    files.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "content": data.decode("utf-8", errors="replace"),
                        }
                    )
                    total += len(data)
        return {"status": run.status, "files": files, "skipped": skipped}

    def _watch(self, run: Run) -> None:
        try:
            run.exit_code = run.process.wait(timeout=self.timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(run.process)
            run.exit_code = run.process.wait()
            timed_out = True
        reap_workers(run.directory)
        if run.stopped:
            run.status = "stopped"
        elif timed_out:
            run.status = "timeout"
        else:
            run.status = "completed" if run.exit_code == 0 else "failed"
        logger.info("[runner] %s %s (exit %s)", run.id, run.status, run.exit_code)

    def _prune_old_dirs(self) -> None:
        cutoff = time.time() - _KEEP_RUN_DIRS_S
        for directory in self.runs_dir.glob("*"):
            if directory.is_dir() and directory.stat().st_mtime < cutoff:
                self._runs.pop(directory.name, None)
                shutil.rmtree(directory, ignore_errors=True)


def reap_workers(run_dir: Path) -> None:
    """Remove any worker container this run left (a killed ``decode run`` never reaches destroy)."""
    from decode.sandbox.docker_backend import WORKSPACE_LABEL  # POSIX-only module (signals)

    workspace = (run_dir / settings.sandbox_workspace_dir).resolve()
    found = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={WORKSPACE_LABEL}={workspace}"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = found.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, check=False)
        logger.info("[runner] removed %d leftover worker(s) for %s", len(ids), run_dir.name)


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """SIGTERM the run's process group, then SIGKILL it after a grace period."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=_STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _decode_cmd() -> list[str]:
    """``decode`` from this interpreter's environment (the console script sits next to python)."""
    return [sys.executable, "-m", "decode"]


def _valid_proxy_url(url: object) -> bool:
    """Only the cookbook proxy's per-run address: the configured prefix plus a token, nothing else."""
    prefix = settings.runner_proxy_prefix
    return (
        isinstance(url, str)
        and url.startswith(prefix)
        and bool(re.fullmatch(r"[A-Za-z0-9_-]{16,64}", url[len(prefix) :]))
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def make_handler(runner: Runner, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "decode-runner"

        def log_message(self, fmt: str, *args: object) -> None:
            logger.info("[runner] %s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: dict[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            given = self.headers.get("Authorization", "")
            if hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "active": runner.active(), "max": runner.max_runs})
                return
            if not self._authorized():
                return
            events = _EVENTS_PATH.match(self.path)
            listing = _FILES_PATH.match(self.path)
            if events:
                body = runner.events(events.group(1), int(events.group(2) or 0))
            elif listing:
                body = runner.files(listing.group(1))
            else:
                match = _RUN_PATH.match(self.path)
                body = runner.get(match.group(1)) if match else None
            self._send(200, body) if body else self._send(404, {"error": "no such run"})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path != "/runs":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(min(length, 64_000)) or b"{}")
                task = str(payload.get("task", "")).strip()
                model = payload.get("model")
            except (ValueError, AttributeError):
                self._send(400, {"error": "body must be JSON with a task"})
                return
            if not task or len(task) > MAX_TASK_CHARS:
                self._send(400, {"error": f"task must be 1-{MAX_TASK_CHARS} characters"})
                return
            if model is not None and not (
                isinstance(model, str) and re.fullmatch(r"[\w.:-]{1,80}", model)
            ):
                self._send(400, {"error": "bad model id"})
                return
            proxy_url = payload.get("proxy_url") or ""
            if proxy_url and not _valid_proxy_url(proxy_url):
                self._send(400, {"error": "proxy_url must be the cookbook proxy's per-run address"})
                return
            run = runner.start(task, model, proxy_url)
            if run is None:
                self._send(429, {"error": "all sandboxes are busy", "max": runner.max_runs})
                return
            self._send(202, {"id": run.id})

        def do_DELETE(self) -> None:
            if not self._authorized():
                return
            match = _RUN_PATH.match(self.path)
            if match and runner.stop(match.group(1)):
                self._send(200, {"id": match.group(1), "stopping": True})
            else:
                self._send(404, {"error": "no such run"})

    return Handler


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build & run runner (decode fork F1)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7700)
    args = parser.parse_args()
    token = settings.runner_token.get_secret_value()
    if len(token) < 32:
        raise SystemExit("decode.runner: set RUNNER_TOKEN (at least 32 characters) to start.")
    runner = Runner(
        settings.runner_runs_dir,
        max_runs=settings.runner_max_runs,
        timeout_s=settings.runner_run_timeout_s,
        max_requests=settings.runner_max_requests,
    )
    settings.runner_runs_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner, token))
    logger.info("[runner] listening on %s:%s (max %d runs)", args.host, args.port, runner.max_runs)
    server.serve_forever()


if __name__ == "__main__":
    main()
