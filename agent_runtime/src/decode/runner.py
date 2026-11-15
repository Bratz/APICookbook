"""The Build & run runner on the sandbox host (APICookbook fork F1; docs/decode-fork-plan.md).

A small private HTTP service that cookbook-api calls to run ``decode run`` for a visitor's task.
Each run is its own ``decode run`` subprocess in its own directory (its harness home, so its session
log and Workspace are separate), because decode keeps per-process state (the bash executor memo).

    POST   /runs          {"task", "model"?, "proxy_url"?, "files"?}  → 202 {"id"} · 429 when all slots are busy
           ("files": [{"path", "content"}] written into the new Workspace first: a follow-up run)
    GET    /runs/<id>     → {"id", "status", "exit_code", "answer", "summary", "stderr_tail"}
    GET    /runs/<id>/events?after=N  → {"status", "events": [...from line N], "next"}  (F2b)
    GET    /runs/<id>/files  → {"status", "files": [{"path", "content"}], "skipped"}  (text files the agent
           wrote; empty while the run is still going)
    DELETE /runs/<id>     → stop it (process group killed, its worker container removed)
    POST   /previews      {"files", "command", "port"}  → 201 {"id", "expires_in"} · 429 · 422
           (a built web app, started in a worker with the same isolation as a run's; it lives 10 minutes)
    POST   /previews/<id>/http  {"method", "path", "content_type"?, "body_b64"?}
           → {"status", "content_type", "body_b64"}  (one request to the app, made INSIDE its container
           with ``docker exec``: no port is published and nothing new can reach the sandbox)
    DELETE /previews/<id> → stop it
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

import base64  # noqa: E402
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

MAX_TASK_CHARS = 8000  # the visitor's 4000 plus a follow-up's note on the run before
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
_SEED_PATH = re.compile(r"(?:[\w-][\w.-]*/){0,8}[\w-][\w.-]*")  # relative, no "..", no hidden names
# A preview: a visitor's built web app, running for a short time so its page can be tried.
_PREVIEW_PATH = re.compile(r"^/previews/([0-9a-f]{32})$")
_PREVIEW_HTTP_PATH = re.compile(r"^/previews/([0-9a-f]{32})/http$")
MAX_PREVIEWS = 2
PREVIEW_TTL_S = 600.0
_PREVIEW_START_WAIT_S = 20.0
_PREVIEW_REQUEST_S = 45  # one request to the app; the app may make several slow bank calls for it
_MAX_PREVIEW_BODY = 1_000_000  # one request or one response
_PREVIEW_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_PREVIEW_URL_PATH = re.compile(r"/[\x21-\x7e]{0,2000}")  # starts with "/", printable, no spaces


@dataclass
class Run:
    id: str
    directory: Path
    process: subprocess.Popen[bytes]
    status: str = "running"  # running | completed | failed | timeout | stopped
    exit_code: int | None = None
    stopped: bool = field(default=False, repr=False)


@dataclass
class Preview:
    id: str
    directory: Path
    container: str
    port: int
    expires: float


class Runner:
    """The run table and its lifecycle; the HTTP handler is a thin shell over it."""

    def __init__(self, runs_dir: Path, *, max_runs: int, timeout_s: float, max_requests: int):
        self.runs_dir = runs_dir
        self.max_runs = max_runs
        self.timeout_s = timeout_s
        self.max_requests = max_requests
        self._runs: dict[str, Run] = {}
        self._previews: dict[str, Preview] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ previews
    def start_preview(
        self, files: list[dict[str, str]], command: str, port: int
    ) -> Preview | str | None:
        """Start the built app in a worker of its own. ``None``: every preview slot is busy.
        A string: the app did not start, and this is why (the end of its log)."""
        with self._lock:
            for old in [p for p in self._previews.values() if p.expires <= time.time()]:
                self._drop_preview(old)
            if len(self._previews) >= MAX_PREVIEWS:
                return None
            preview_id = uuid.uuid4().hex
            directory = self.runs_dir / preview_id
            workspace = (directory / settings.sandbox_workspace_dir).resolve()
            for seed in files:
                target = workspace / seed["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(seed["content"].encode())
            started = _docker(*_preview_run_args(workspace))
            container = started.stdout.decode().strip()
            if started.returncode != 0 or not container:
                shutil.rmtree(directory, ignore_errors=True)
                return "the sandbox could not start a preview worker"
            preview = Preview(
                id=preview_id,
                directory=directory,
                container=container,
                port=port,
                expires=time.time() + PREVIEW_TTL_S,
            )
            self._previews[preview_id] = preview
        # The app: the visitor's own command, inside the same sandbox their run used.
        _docker(
            "exec", "-d", container, "bash", "-c",
            f"cd /workspace && exec {command} > /tmp/preview.log 2>&1",
        )  # fmt: skip
        deadline = time.time() + _PREVIEW_START_WAIT_S
        while time.time() < deadline:
            probe = _docker(
                "exec", container, "curl", "-s", "-o", "/dev/null", "--max-time", "2",
                f"http://127.0.0.1:{port}/",
            )  # fmt: skip
            if probe.returncode == 0:
                timer = threading.Timer(PREVIEW_TTL_S, self.stop_preview, [preview_id])
                timer.daemon = True
                timer.start()
                logger.info("[runner] preview %s is up on port %s", preview_id, port)
                return preview
            time.sleep(0.5)
        log = _docker("exec", container, "tail", "-c", "1500", "/tmp/preview.log")
        self.stop_preview(preview_id)
        return log.stdout.decode(errors="replace").strip() or "the app did not answer on its port"

    def preview_http(
        self, preview_id: str, method: str, path: str, content_type: str, body: bytes
    ) -> dict[str, object] | None:
        """One request to the app, made inside its container; ``None`` when there is no such preview."""
        preview = self._previews.get(preview_id)
        if preview is None or preview.expires <= time.time():
            return None
        # Every value is its own argv element: nothing here is read by a shell.
        argv = [
            "exec", "-i", preview.container, "curl", "-s", "-o", "-", "-X", method,
            "--max-time", str(_PREVIEW_REQUEST_S), "--max-filesize", str(_MAX_PREVIEW_BODY),
            "-w", "\n%{http_code}\t%{content_type}",
        ]  # fmt: skip
        if body:
            argv += [
                "--data-binary",
                "@-",
                "-H",
                f"Content-Type: {content_type or 'application/json'}",
            ]
        argv.append(f"http://127.0.0.1:{preview.port}{path}")
        done = _docker(*argv, stdin=body or None, timeout=_PREVIEW_REQUEST_S + 10)
        payload, _, trailer = done.stdout.rpartition(b"\n")
        status, _, kind = trailer.decode(errors="replace").partition("\t")
        if done.returncode != 0 or not status.isdigit() or status == "000":
            return {"status": 502, "content_type": "text/plain", "body_b64": base64.b64encode(
                b"The app in the preview did not answer.").decode()}  # fmt: skip
        return {
            "status": int(status),
            "content_type": kind or "application/octet-stream",
            "body_b64": base64.b64encode(payload[:_MAX_PREVIEW_BODY]).decode(),
        }

    def stop_preview(self, preview_id: str) -> bool:
        with self._lock:
            preview = self._previews.get(preview_id)
            if preview is None:
                return False
            self._drop_preview(preview)
        return True

    def _drop_preview(self, preview: Preview) -> None:
        self._previews.pop(preview.id, None)
        _docker("rm", "-f", preview.container)
        shutil.rmtree(preview.directory, ignore_errors=True)
        logger.info("[runner] preview %s stopped", preview.id)

    def active(self) -> int:
        with self._lock:
            return sum(run.status == "running" for run in self._runs.values())

    def start(
        self,
        task: str,
        model: str | None = None,
        proxy_url: str = "",
        files: list[dict[str, str]] | None = None,
    ) -> Run | None:
        """Start a run, or return ``None`` when every slot is busy."""
        with self._lock:
            if sum(run.status == "running" for run in self._runs.values()) >= self.max_runs:
                return None
            self._prune_old_dirs()
            run_id = uuid.uuid4().hex
            directory = self.runs_dir / run_id
            directory.mkdir(parents=True)
            workspace = directory / settings.sandbox_workspace_dir
            for seed in files or []:  # a follow-up starts from the files the run before left
                target = workspace / seed["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(seed["content"].encode())
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


def _docker(
    *args: str, stdin: bytes | None = None, timeout: float = 60
) -> subprocess.CompletedProcess[bytes]:
    """One ``docker`` command, never through a shell; a hang or a missing docker reads as a failure."""
    try:
        return subprocess.run(
            ["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[runner] docker %s failed: %s", args[0], type(exc).__name__)
        return subprocess.CompletedProcess(["docker", *args], 1, b"", b"")


def _preview_run_args(workspace: Path) -> list[str]:
    """``docker run`` for a preview worker: exactly the argv a run's worker gets (gVisor, the locked
    network, read-only root, no capabilities), so a preview is never less isolated than a run."""
    from decode.sandbox.docker_backend import DockerBackend  # POSIX-only module (signals)

    return DockerBackend()._run_args(workspace)


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


def _valid_seed(files: object) -> bool:
    """Seed files for a follow-up: plain relative paths, text, inside the same caps as the listing."""
    if not isinstance(files, list) or len(files) > _MAX_FILES:
        return False
    total = 0
    for seed in files:
        if not (
            isinstance(seed, dict)
            and isinstance(seed.get("path"), str)
            and isinstance(seed.get("content"), str)
            and _SEED_PATH.fullmatch(seed["path"])
            and ".." not in seed["path"]
        ):
            return False
        total += len(seed["content"].encode())
    return total <= _MAX_FILES_BYTES


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
            relay = _PREVIEW_HTTP_PATH.match(self.path)
            if self.path not in ("/runs", "/previews") and not relay:
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(min(length, 3 * _MAX_FILES_BYTES)) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError
                task = str(payload.get("task", "")).strip()
                model = payload.get("model")
            except (ValueError, AttributeError):
                self._send(400, {"error": "body must be a JSON object"})
                return
            if relay:
                self._preview_http(relay.group(1), payload)
                return
            if self.path == "/previews":
                self._preview_start(payload)
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
            files = payload.get("files") or []
            if not _valid_seed(files):
                self._send(
                    400, {"error": "files must be a list of {path, content} within the caps"}
                )
                return
            run = runner.start(task, model, proxy_url, files)
            if run is None:
                self._send(429, {"error": "all sandboxes are busy", "max": runner.max_runs})
                return
            self._send(202, {"id": run.id})

        def _preview_start(self, payload: dict[str, object]) -> None:
            files, command, port = payload.get("files"), payload.get("command"), payload.get("port")
            if not (
                _valid_seed(files)
                and files
                and isinstance(command, str)
                and 0 < len(command.strip()) <= 300
                and len(command.splitlines()) == 1
                and isinstance(port, int)
                and 1024 <= port <= 65535
            ):
                self._send(400, {"error": "a preview needs files, a one-line command and a port"})
                return
            preview = runner.start_preview(files, command.strip(), port)  # type: ignore[arg-type]
            if preview is None:
                self._send(429, {"error": "all preview slots are busy", "max": MAX_PREVIEWS})
            elif isinstance(preview, str):
                self._send(422, {"error": "the app did not start", "log": preview})
            else:
                self._send(201, {"id": preview.id, "expires_in": int(PREVIEW_TTL_S)})

        def _preview_http(self, preview_id: str, payload: dict[str, object]) -> None:
            method, path = str(payload.get("method", "GET")).upper(), payload.get("path")
            try:
                body = base64.b64decode(str(payload.get("body_b64") or ""), validate=True)
            except ValueError:
                body = None
            if (
                method not in _PREVIEW_METHODS
                or not isinstance(path, str)
                or not _PREVIEW_URL_PATH.fullmatch(path)
                or body is None
                or len(body) > _MAX_PREVIEW_BODY
            ):
                self._send(400, {"error": "bad preview request"})
                return
            kind = str(payload.get("content_type") or "")[:100]
            if not re.fullmatch(r"[\w!#$&^.+*/; =-]*", kind):
                kind = ""
            answer = runner.preview_http(preview_id, method, path, kind, body)
            self._send(200, answer) if answer else self._send(404, {"error": "no such preview"})

        def do_DELETE(self) -> None:
            if not self._authorized():
                return
            gone = _PREVIEW_PATH.match(self.path)
            if gone:
                if runner.stop_preview(gone.group(1)):
                    self._send(200, {"id": gone.group(1), "stopped": True})
                else:
                    self._send(404, {"error": "no such preview"})
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
