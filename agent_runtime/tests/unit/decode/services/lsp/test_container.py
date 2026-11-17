"""The Language Server in the worker container (APICookbook fork).

In docker mode the code's packages are in the worker image, so the server is started there with
``docker exec -i`` and the client speaks the container's paths (``/workspace/...``) while it reads
the files through the host's bind mount.
"""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import Any

from support.lsp_fakes import FakeLanguageServer

from decode.config.settings import settings
from decode.services.lsp import service as lsp_service
from decode.services.lsp.client import LspClient
from decode.services.lsp.types import Diagnostic, Location


def _request_for(fake: FakeLanguageServer, method: str) -> dict[str, Any]:
    return next(req for req in fake.requests if req.get("method") == method)


def _seed(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("import os\n", encoding="utf-8")


async def test_a_client_with_a_container_root_speaks_container_paths(tmp_path: Path):
    _seed(tmp_path)
    where = {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 3}}
    fake = FakeLanguageServer(
        {"textDocument/definition": {"uri": "file:///workspace/pkg/other.py", "range": where}}
    )
    client = LspClient(fake, tmp_path, PurePosixPath("/workspace"))

    await client.initialize()
    found = await client.definition("pkg/a.py", line=1, column=8)

    init = _request_for(fake, "initialize")["params"]
    assert init["rootUri"] == "file:///workspace"
    assert init["processId"] is None  # the host pid means nothing inside the container
    opened = _request_for(fake, "textDocument/didOpen")["params"]["textDocument"]
    assert opened["uri"] == "file:///workspace/pkg/a.py"
    assert opened["text"] == "import os\n"  # read on the host, through the bind mount
    assert found == Location(path="pkg/other.py", line=3, column=1)  # back to a Workspace path


class _Backend:
    def __init__(self, process: Any) -> None:
        self.process, self.spawned = process, []

    async def spawn(self, *args: str) -> Any:
        self.spawned.append(args)
        return self.process


class _Executor:
    def __init__(self, backend: _Backend) -> None:
        self.backend, self.asked = backend, []

    async def file_backend(self, cwd: Path) -> _Backend:
        self.asked.append(cwd)
        return self.backend


async def test_docker_mode_starts_the_server_in_the_worker(tmp_path: Path, monkeypatch):
    _seed(tmp_path)
    diagnostic = {
        "range": {"start": {"line": 0, "character": 7}, "end": {"line": 0, "character": 9}},
        "severity": 1,
        "message": "bad",
    }
    fake = FakeLanguageServer({"textDocument/diagnostic": {"kind": "full", "items": [diagnostic]}})
    executor = _Executor(_Backend(fake))
    monkeypatch.setattr(settings, "sandbox_mode", "docker")
    monkeypatch.setattr("decode.tools.bash.active_executor", lambda: executor)

    result = await lsp_service.diagnostics(tmp_path, "pkg/a.py")

    assert executor.backend.spawned == [(settings.lsp_server_command, *settings.lsp_server_args)]
    assert executor.asked == [tmp_path.resolve()]  # the container exists before the server starts
    assert _request_for(fake, "initialize")["params"]["rootUri"] == "file:///workspace"
    assert result == [Diagnostic(line=1, column=8, severity=1, message="bad")]


async def test_host_mode_keeps_host_paths(tmp_path: Path, mocker, monkeypatch):
    _seed(tmp_path)
    fake = FakeLanguageServer()
    monkeypatch.setattr(settings, "sandbox_mode", "none")
    mocker.patch.object(asyncio, "create_subprocess_exec", return_value=fake)

    await lsp_service.diagnostics(tmp_path, "pkg/a.py")

    init = _request_for(fake, "initialize")["params"]
    assert init["rootUri"] == tmp_path.resolve().as_uri() and init["processId"] is not None
