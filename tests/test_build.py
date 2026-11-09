"""Build & run's BaNCS proxy and operator routes (docs/decode-fork-plan.md, F2a).

The proxy runs in-process against a fake bank, in ONE event loop (held writes wait on an
asyncio.Event, exactly as in production, where the proxy shares the app's loop).
"""

import asyncio

import httpx
import pytest

from backend import build_proxy, build_routes

BANK_CALLS: list[httpx.Request] = []


def _fake_bank(request: httpx.Request) -> httpx.Response:
    BANK_CALLS.append(request)
    return httpx.Response(200, json={"bank": "ok", "path": request.url.path})


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def proxy(monkeypatch):
    BANK_CALLS.clear()
    monkeypatch.setattr(build_proxy, "_bank_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(_fake_bank)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=build_proxy.proxy_app), base_url="http://proxy")
    run = build_proxy.new_run()
    yield client, run
    build_proxy.end_run(run.token)


async def _until_pending(token: str) -> dict:
    for _ in range(100):
        held = build_proxy.pending(token)
        if held:
            return held[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no write was held")


@pytest.mark.anyio
async def test_a_get_passes_straight_through_with_the_bank_headers(proxy):
    client, run = proxy
    response = await client.get(f"/r/{run.token}/Core/accountManagement/account/balanceDetails/42")

    assert response.status_code == 200
    assert response.json()["path"] == "/Core/accountManagement/account/balanceDetails/42"
    sent = BANK_CALLS[-1]
    assert sent.url.host == httpx.URL(build_proxy.settings.api_base_url).host  # only ever the bank
    assert {"entity", "userid", "languagecode"} <= {k.lower() for k in sent.headers}


@pytest.mark.anyio
async def test_a_write_is_held_until_approved(proxy):
    client, run = proxy
    call = asyncio.create_task(client.post(f"/r/{run.token}/Core/customer", json={"name": "A"}))

    held = await _until_pending(run.token)
    assert held["method"] == "POST" and held["path"] == "/Core/customer" and '"A"' in held["body"]
    assert not BANK_CALLS  # nothing reached the bank while it waits

    build_proxy.decide(run.token, held["id"], approve=True)
    response = await call

    assert response.status_code == 200
    assert BANK_CALLS[-1].method == "POST"


@pytest.mark.anyio
async def test_a_rejected_write_never_reaches_the_bank(proxy):
    client, run = proxy
    call = asyncio.create_task(client.delete(f"/r/{run.token}/Core/customer/7"))

    held = await _until_pending(run.token)
    build_proxy.decide(run.token, held["id"], approve=False)
    response = await call

    assert response.status_code == 403
    assert "was not sent to the bank" in response.json()["error"]
    assert not BANK_CALLS


@pytest.mark.anyio
async def test_an_unanswered_write_times_out_as_rejected(proxy, monkeypatch):
    client, run = proxy
    monkeypatch.setattr(build_proxy, "WRITE_TIMEOUT_S", 0.05)

    response = await client.put(f"/r/{run.token}/Core/customer/7", json={})

    assert response.status_code == 403
    assert "in time" in response.json()["error"]
    assert not BANK_CALLS


@pytest.mark.anyio
async def test_a_token_stops_working_when_its_run_ends(proxy):
    client, run = proxy
    build_proxy.end_run(run.token)

    assert (await client.get(f"/r/{run.token}/Core/x")).status_code == 404
    assert (await client.get("/r/not-a-token/Core/x")).status_code == 404
    assert not BANK_CALLS


@pytest.mark.anyio
async def test_ending_a_run_rejects_its_held_writes(proxy):
    client, run = proxy
    call = asyncio.create_task(client.post(f"/r/{run.token}/Core/customer", json={}))
    await _until_pending(run.token)

    build_proxy.end_run(run.token)

    assert (await call).status_code == 403
    assert not BANK_CALLS


@pytest.mark.anyio
async def test_the_knowledge_endpoints_point_the_agent_at_its_proxy(client, proxy):
    # `client` (conftest) runs the app's lifespan, which loads the knowledge base the proxy reads.
    proxy_client, run = proxy
    hits = (await proxy_client.get(f"/r/{run.token}/_kb/search", params={"q": "account balance"})).json()
    assert hits and "endpoint" in hits[0]

    endpoint = (await proxy_client.get(f"/r/{run.token}/_kb/endpoint/{hits[0]['endpoint']}")).json()

    assert endpoint["url_template"].startswith(build_proxy.public_base(run.token) + "/Core")
    assert "demoapps" not in endpoint["url_template"]  # never the bank's own host


def test_the_build_routes_need_the_runner_token(client, monkeypatch):
    monkeypatch.setenv("RUNNER_TOKEN", "x" * 40)

    assert client.post("/api/build/runs", json={"task": "t"}).status_code == 401
    assert client.post("/api/build/runs", json={"task": "t"},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/build/runs/abc", headers={"Authorization": "Bearer " + "x" * 40}).status_code == 404


def test_the_build_routes_are_off_without_a_runner_token(client, monkeypatch):
    monkeypatch.delenv("RUNNER_TOKEN", raising=False)

    assert client.post("/api/build/runs", json={"task": "t"}).status_code == 503
    assert build_routes._RUNS == {}


@pytest.mark.anyio
async def test_the_stream_merges_steps_writes_and_bank_calls_then_ends(monkeypatch):
    run = build_proxy.new_run()
    held = build_proxy.HeldWrite(id="w1", method="POST", path="/Core/customer", body_preview="{}")
    run.writes["w1"] = held
    run.calls.append({"method": "GET", "path": "/Core/a", "status": 200})
    run.calls.append({"method": "POST", "path": "/Core/customer", "write_id": "w1"})  # still waiting
    pages = iter([
        {"status": "running", "events": [{"type": "tool_call", "tool": "bash"}], "next": 1},
        {"status": "completed", "events": [], "next": 1},
    ])

    async def fake_runner(method, path, json=None):
        if path.endswith("/events?after=0") or path.endswith("/events?after=1"):
            page = next(pages)
            if page["status"] == "completed":  # the visitor approved in between
                build_proxy.decide(run.token, "w1", approve=True)
                run.calls[1].update(decision="approved", status=201)
            return httpx.Response(200, json=page)
        return httpx.Response(200, json={"status": "completed", "exit_code": 0, "answer": "done", "summary": {}})

    monkeypatch.setattr(build_routes, "_call_runner", fake_runner)
    monkeypatch.setattr(build_routes, "STREAM_POLL_S", 0)

    chunks = [c async for c in build_routes.stream_run("r1", run.token)]
    kinds = [c.split("\n", 1)[0].removeprefix("event: ") for c in chunks]

    assert kinds == ["step", "write", "bank_call", "write_decided", "bank_call", "done"]
    assert '"answer": "done"' in chunks[-1]
    assert not run.alive  # the stream ends the run's token with it
