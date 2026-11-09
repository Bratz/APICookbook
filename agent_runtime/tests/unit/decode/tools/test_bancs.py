"""The BaNCS knowledge tools and prompt part (APICookbook fork F2a): proxy calls, offline."""

from __future__ import annotations

import json

import httpx
import pytest

from decode.config.settings import settings
from decode.tools import bancs

PROXY = "http://10.0.0.49:7601/r/tok_abcdefghijklmnop"


@pytest.fixture
def seen(monkeypatch):
    """Route the tools' HTTP to a fake proxy; return the requests it saw."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/_kb/endpoint/missing"):
            return httpx.Response(404, json={"detail": "no such run"})
        return httpx.Response(
            200, json={"path": request.url.path, "q": request.url.params.get("q")}
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        bancs.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(settings, "bancs_proxy_url", PROXY)
    return requests


async def test_the_tools_read_the_runs_proxy(seen):
    found = json.loads(await bancs.search_endpoints(None, "open account"))
    endpoint = json.loads(await bancs.get_endpoint(None, "createAccount"))
    schema = json.loads(await bancs.get_body_schema(None, "a/b c"))

    assert found == {"path": "/r/tok_abcdefghijklmnop/_kb/search", "q": "open account"}
    assert endpoint["path"].endswith("/_kb/endpoint/createAccount")
    assert schema["path"].endswith("/_kb/body-schema/a/b c")  # decoded by the server...
    assert seen[-1].url.raw_path.endswith(b"/_kb/body-schema/a%2Fb%20c")  # ...but one path segment
    assert all(str(r.url).startswith(PROXY) for r in seen)


async def test_a_proxy_error_is_text_for_the_model_not_an_exception(seen):
    result = await bancs.get_endpoint(None, "missing")

    assert result.startswith("The API knowledge service answered 404")


async def test_without_a_proxy_the_tools_say_so_and_the_prompt_has_no_bancs_part(monkeypatch):
    monkeypatch.setattr(settings, "bancs_proxy_url", "")

    assert await bancs.search_endpoints(None, "x") == bancs._NOT_CONFIGURED
    assert bancs.instructions() == ""


def test_the_prompt_names_the_proxy_base_and_the_approval_rule(monkeypatch):
    monkeypatch.setattr(settings, "bancs_proxy_url", PROXY + "/")

    text = bancs.instructions()

    assert f"{PROXY}/Core" in text  # the only address the code may call
    assert "WAITS" in text and "403" in text  # writes are held; a refusal is final
    # F5 finding: an unattended run must find real records and act, never stop to ask.
    assert "do not call ask_user" in text and "find a REAL one yourself" in text
    assert "MAKE the call" in text
