"""The Workspace = ``git clone`` real-infra smokes: ``--repo`` populates ``/workspace`` (ADR-0012 §3,4).

Task 082's living proof that a user-supplied repo, cloned host-side into ``.decode/sandbox`` by
:func:`decode.sandbox.workspace.prepare_workspace`, is the isolated ``/workspace`` a sandbox's ``bash``
and file tools operate on. A ``skipif``-guarded real-infra smoke (modal), using the SAME
predicates as the executors' own integration tests, plus a host-side check that the clone is a real repo
with a recoverable origin (the substrate task 083's git hand-back ships from).

Hermetic ``--repo`` source: a throwaway **local** git repo under ``tmp_path`` — no network, no remote.
Each smoke reaps its container / remote sandbox in a ``finally`` (self-reap), so no infra litter is left
even on failure, and asserts the specific resource it created is gone (leaving any pre-existing manual
keeper alone).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from decode.sandbox.executor import SandboxExecutor
from decode.sandbox.modal_backend import ModalBackend
from decode.sandbox.workspace import prepare_workspace

# A real remote-sandbox cold start (image pull + spawn + upload) can take a while.
_MODAL_TIMEOUT_S = 120.0


def _modal_credentials_present() -> bool:
    """True if modal account credentials are present (else the modal smoke SKIPS) — presence only."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return (Path.home() / ".modal.toml").exists()


_MODAL_AVAILABLE = _modal_credentials_present()


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


@pytest.mark.skipif(not _MODAL_AVAILABLE, reason="modal account credentials are not present")
async def test_real_modal_repo_clone_is_uploaded_into_workspace(tmp_path: Path) -> None:
    """modal: a ``--repo`` clone is bootstrap-uploaded into the remote ``/workspace`` — else SKIP (§3,5).

    Clone a local repo host-side, start a real remote sandbox (which bootstrap-uploads the cloned
    Workspace), and prove ``bash`` sees the cloned file remotely. Reaps the remote sandbox in a
    ``finally`` (``aclose`` = export + terminate) and asserts it was cleared (no leaked remote sandbox).
    """
    source = _make_local_git_repo(tmp_path / "source")
    workspace = prepare_workspace(tmp_path, repo=str(source))
    executor = SandboxExecutor(ModalBackend())
    try:
        await executor.start(workspace)  # spawn + bootstrap-upload the cloned Workspace
        seen = await executor.run(
            "cat /workspace/README.md", cwd=workspace, timeout_s=_MODAL_TIMEOUT_S
        )
        assert seen.stdout.strip() == "cloned-into-workspace"
        # The file backend reads the same cloned file back directly (no mirror).
        assert (
            await executor._backend.read_bytes("README.md")
        ).decode() == "cloned-into-workspace\n"
    finally:
        await executor.aclose()  # export + terminate the remote sandbox (no leak)

    assert executor._backend._sandbox is None  # aclose cleared it — a later run creates a fresh one
