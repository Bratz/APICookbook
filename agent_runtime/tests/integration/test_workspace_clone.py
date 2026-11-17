"""The Workspace = ``git clone`` real-infra smokes: ``--repo`` populates ``/workspace`` (ADR-0012 §3,4).

Task 082's living proof that a user-supplied repo, cloned host-side into ``.decode/sandbox`` by
:func:`decode.sandbox.workspace.prepare_workspace`, is the isolated ``/workspace`` a sandbox's ``bash``
and file tools operate on. A ``skipif``-guarded real-infra docker smoke, using the SAME
predicate as the executor's own integration tests, plus a host-side check that the clone is a real repo
with a recoverable origin (the substrate task 083's git hand-back ships from).

Hermetic ``--repo`` source: a throwaway **local** git repo under ``tmp_path`` — no network, no remote.
Each smoke reaps its container / remote sandbox in a ``finally`` (self-reap), so no infra litter is left
even on failure, and asserts the specific resource it created is gone (leaving any pre-existing manual
keeper alone).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from decode.sandbox.docker_backend import DockerBackend
from decode.sandbox.executor import SandboxExecutor
from decode.sandbox.workspace import prepare_workspace


def _docker_available() -> bool:
    """True if a local docker daemon answers a fast ``docker info`` probe (else the docker smoke SKIPS)."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5.0, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


_DOCKER_AVAILABLE = _docker_available()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_local_git_repo(path: Path) -> Path:
    """A throwaway local git repo with one committed file — the hermetic ``--repo`` source."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@decode.local")
    _git(path, "config", "user.name", "decode test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("cloned-into-workspace\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _container_gone(container_id: str, timeout_s: float = 5.0) -> bool:
    """Poll (bounded) until the daemon no longer lists ``container_id`` — the self-reap proof."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"id={container_id}"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if not result.stdout.strip():
            return True
        time.sleep(0.2)
    return False


def test_cloned_workspace_is_a_real_repo_with_a_recoverable_origin(tmp_path: Path) -> None:
    """Host-side (no infra): ``--repo`` lands a REAL clone at ``.decode/sandbox`` with origin recoverable.

    The always-run anchor of the clone substrate (ADR-0012 §8): the Workspace is host-visible at the
    canonical ``.decode/sandbox`` path, a working ``.git`` clone, and its ``origin`` + cloned HEAD are
    recoverable — the exact facts task 083's hand-back branches, secures, and pushes.
    """
    source = _make_local_git_repo(tmp_path / "source")

    workspace = prepare_workspace(tmp_path / "home", repo=str(source))

    assert workspace == (tmp_path / "home" / ".decode" / "sandbox").resolve()
    assert (workspace / "README.md").read_text(encoding="utf-8") == "cloned-into-workspace\n"
    assert (workspace / ".git").is_dir()
    assert _git_out(workspace, "remote", "get-url", "origin") == str(source)  # where 083 pushes
    assert _git_out(workspace, "rev-parse", "HEAD") == _git_out(source, "rev-parse", "HEAD")


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="the docker daemon is not reachable")
async def test_real_docker_repo_clone_is_visible_in_workspace(tmp_path: Path) -> None:
    """docker: a ``--repo`` clone is the ``/workspace`` ``bash`` + the file backend see — else SKIP (§3,4).

    Clone a local repo host-side into ``.decode/sandbox``, start a real container over the bind mount,
    and prove the cloned file is visible BOTH via ``bash`` (``cat``) and via the file backend
    (``read_bytes``) — the one truthful tree ADR-0012 §4 promises. Reaps its container in a ``finally``
    and asserts that specific container is gone (self-reap), leaving any pre-existing keeper alone.
    """
    source = _make_local_git_repo(tmp_path / "source")
    workspace = prepare_workspace(tmp_path, repo=str(source))  # == <cwd>/.decode/sandbox
    executor = SandboxExecutor(DockerBackend())
    container_id: str | None = None
    try:
        await executor.start(workspace)  # bind-mount the cloned Workspace at /workspace
        container_id = executor._backend._container_id
        assert container_id is not None

        # (1) bash sees the cloned file under /workspace (== deps.cwd) ...
        seen = await executor.run("cat README.md", cwd=workspace, timeout_s=30.0)
        assert seen.exit_code == 0
        assert "cloned-into-workspace" in seen.stdout
        # (2) ... and so does the file backend (the read tool's byte transport) — one truthful tree.
        assert (
            await executor._backend.read_bytes("README.md")
        ).decode() == "cloned-into-workspace\n"
        # (3) the clone is a real repo inside the mount too — the .git dir rode in with the bind mount
        #     (checked with `test -d`, since the slim worker image ships no git binary).
        git_probe = await executor.run(
            "test -d .git && echo has-git", cwd=workspace, timeout_s=30.0
        )
        assert git_probe.stdout.strip() == "has-git"

        await executor.aclose()
        assert _container_gone(container_id), "aclose must stop + remove the session container"
    finally:
        await executor.aclose()  # idempotent safety net — no leaked keeper
