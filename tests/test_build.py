"""Build & run's BaNCS proxy and operator routes (docs/decode-fork-plan.md, F2a).

The proxy runs in-process against a fake bank, in ONE event loop (held writes wait on an
asyncio.Event, exactly as in production, where the proxy shares the app's loop).
"""

import asyncio
import io
import pathlib
import tempfile
import zipfile

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


RUN_ID = "r" * 32


@pytest.fixture
def open_build(monkeypatch, tmp_path):
    """Build & run open to visitors, against a fake runner, with a fresh ledger (1 run/hour/IP)."""
    monkeypatch.setenv("RUNNER_TOKEN", "x" * 40)
    monkeypatch.setenv("BUILD_PUBLIC", "true")
    monkeypatch.setattr(build_routes, "_GUARD",
                        build_routes.AgentGuard(tmp_path, per_hour=1, daily_usd=5, monthly_usd=50))
    build_routes._RUNS.clear()

    async def fake_runner(method, path, json=None):
        if method == "POST":
            return httpx.Response(202, json={"id": RUN_ID})
        return httpx.Response(200, json={"status": "running"})

    async def no_watcher(run_id):
        return None

    monkeypatch.setattr(build_routes, "_call_runner", fake_runner)
    monkeypatch.setattr(build_routes, "_settle_when_done", no_watcher)
    return build_routes._GUARD


def test_a_visitor_starts_a_run_and_gets_its_own_hidden_key(client, open_build):
    started = client.post("/api/build/runs", json={"task": "list customers"})

    assert started.status_code == 202
    run_id, key = started.json()["id"], started.json()["key"]
    assert run_id == RUN_ID and len(key) >= 32
    assert client.get(f"/api/build/runs/{run_id}", headers={"X-Run-Key": key}).status_code == 200
    # Another tab, or someone who only guessed the id, sees nothing.
    assert client.get(f"/api/build/runs/{run_id}").status_code == 404
    assert client.get(f"/api/build/runs/{run_id}", headers={"X-Run-Key": "x" * 32}).status_code == 404
    assert client.delete(f"/api/build/runs/{run_id}", headers={"X-Run-Key": "nope"}).status_code == 404


def test_the_per_ip_cap_answers_in_build_and_runs_own_words(client, open_build):
    assert client.post("/api/build/runs", json={"task": "a"}).status_code == 202

    refused = client.post("/api/build/runs", json={"task": "b"})

    assert refused.status_code == 429
    assert refused.json()["message"] == "You have started 1 runs in the last hour; please try again later."
    # The operator skips the per-IP limit (the USD caps still bind everyone through the ledger).
    assert client.post("/api/build/runs", json={"task": "c"},
                       headers={"Authorization": "Bearer " + "x" * 40}).status_code == 202


def test_starting_is_closed_unless_the_mode_is_public(client, open_build, monkeypatch):
    monkeypatch.setenv("BUILD_PUBLIC", "false")

    assert client.post("/api/build/runs", json={"task": "t"}).status_code == 401
    assert client.get("/api/build/status").json()["available"] is False


def test_the_build_routes_are_off_without_a_runner_token(client, monkeypatch):
    monkeypatch.delenv("RUNNER_TOKEN", raising=False)

    assert client.post("/api/build/runs", json={"task": "t"}).status_code == 503
    assert client.get("/api/build/status").json()["available"] is False


@pytest.mark.anyio
async def test_a_runs_cost_is_recorded_once_and_its_token_ends(open_build):
    proxy = build_proxy.new_run()
    build_routes._RUNS["s" * 32] = build_routes.BuildRun(token=proxy.token, key="k" * 32)
    final = {"status": "completed", "summary": {"cost_usd": 0.21, "input_tokens": 9, "output_tokens": 1}}

    await build_routes.settle("s" * 32, final)
    await build_routes.settle("s" * 32, final)  # the stream and the watcher may both finish it

    assert open_build.status()["today_usd"] == 0.21
    assert len(open_build.ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert not proxy.alive


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

    monkeypatch.setattr(build_routes, "_GUARD", build_routes.AgentGuard(pathlib.Path(tempfile.mkdtemp())))
    build_routes._RUNS["r1"] = build_routes.BuildRun(token=run.token, key="k" * 32)
    chunks = [c async for c in build_routes.stream_run("r1", run.token)]
    kinds = [c.split("\n", 1)[0].removeprefix("event: ") for c in chunks]

    assert kinds == ["step", "write", "bank_call", "write_decided", "bank_call", "done"]
    assert '"answer": "done"' in chunks[-1]
    assert not run.alive  # the stream ends the run's token with it


def test_a_runs_files_come_back_pointed_at_the_real_bank(client, open_build, monkeypatch):
    started = client.post("/api/build/runs", json={"task": "list customers"}).json()
    run_id, key = started["id"], started["key"]
    private = build_proxy.public_base(build_routes._RUNS[run_id].token)
    code = f'BASE = "{private}/Core"\nrequests.get(BASE + "/customerManagement/customer/list/1")\n'

    async def fake_runner(method, path, json=None):
        assert path == f"/runs/{run_id}/files"
        return httpx.Response(200, json={"status": "completed", "skipped": 1, "files": [
            {"path": "list.py", "content": code}, {"path": "../escape.py", "content": "x"}]})

    monkeypatch.setattr(build_routes, "_call_runner", fake_runner)

    listing = client.get(f"/api/build/runs/{run_id}/files", headers={"X-Run-Key": key}).json()
    assert [f["path"] for f in listing["files"]] == ["list.py"]  # nothing that climbs out of the folder
    assert listing["files"][0]["content"].startswith('BASE = "https://demoapps.tcsbancs.com/Core"')
    assert "/r/" not in listing["files"][0]["content"]  # the run's private proxy token is gone

    zipped = client.get(f"/api/build/runs/{run_id}/files.zip", headers={"X-Run-Key": key})
    assert zipped.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
        assert sorted(archive.namelist()) == ["bancs-run/README.txt", "bancs-run/list.py"]
        assert "entity: GPRDTTSTOU" in archive.read("bancs-run/README.txt").decode()
    assert client.get(f"/api/build/runs/{run_id}/files").status_code == 404  # no key, nothing


def test_a_follow_up_carries_the_runs_files_and_a_short_note(client, open_build, monkeypatch):
    operator = {"Authorization": "Bearer " + "x" * 40}  # skips the 1-run-per-hour test cap
    first = client.post("/api/build/runs", json={"task": "list deposits"}, headers=operator).json()
    old_private = build_proxy.public_base(build_routes._RUNS[first["id"]].token)
    sent = {}

    async def fake_runner(method, path, json=None):
        if method == "POST":
            sent.update(json)
            return httpx.Response(202, json={"id": "f" * 32})
        if path.endswith("/files"):
            return httpx.Response(200, json={"status": "completed", "skipped": 0, "files": [
                {"path": "td.py", "content": f'B = "{old_private}/Core"\n'}]})
        return httpx.Response(200, json={"status": "completed", "answer": f"Wrote td.py against {old_private}/Core."})

    monkeypatch.setattr(build_routes, "_call_runner", fake_runner)

    second = client.post("/api/build/runs", headers=operator, json={
        "task": "add the interest rate", "previous": {"id": first["id"], "key": first["key"]}})

    assert second.status_code == 202
    new_private = build_proxy.public_base(build_routes._RUNS["f" * 32].token)
    assert sent["files"] == [{"path": "td.py", "content": f'B = "{new_private}/Core"\n'}]  # the new run's proxy
    assert sent["task"].endswith("New task: add the interest rate")
    assert "Previous task: list deposits" in sent["task"] and "td.py" in sent["task"]
    assert old_private not in sent["task"]  # the old run's token never reaches the new run
    assert build_routes._RUNS["f" * 32].task == "add the interest rate"

    stranger = client.post("/api/build/runs", headers=operator, json={
        "task": "t", "previous": {"id": first["id"], "key": "not-the-key"}})
    assert stranger.status_code == 404  # only the browser that ran it can follow it up


@pytest.mark.anyio
@pytest.mark.parametrize(("sent", "expected"), [(None, "application/json"), ("*/*", "application/json"),
                                                ("application/xml", "application/xml")])
async def test_the_proxy_asks_the_bank_for_json(proxy, sent, expected):
    # The bank answers XML to no Accept or */*, which is what requests, httpx and curl send.
    client, run = proxy
    headers = {"Accept": sent} if sent else {}
    await client.get(f"/r/{run.token}/Core/accountManagement/account/balanceDetails/42", headers=headers)
    assert BANK_CALLS[-1].headers["accept"] == expected


@pytest.mark.anyio
@pytest.mark.parametrize("agent_sent", [None, "", "something-else"])
async def test_the_proxy_sends_the_visitors_token_whatever_the_code_sent(proxy, agent_sent):
    client, run = proxy
    run.bank_token = "visitor-token-1"
    headers = {"Accesstoken": agent_sent} if agent_sent is not None else {}
    await client.get(f"/r/{run.token}/Core/accountManagement/account/balanceDetails/42", headers=headers)
    assert BANK_CALLS[-1].headers["accesstoken"] == "visitor-token-1"
    assert "visitor-token-1" not in repr(run)


def test_a_visitor_run_needs_a_bank_token_and_the_run_carries_it(client, open_build, monkeypatch):
    monkeypatch.setattr(build_proxy.settings, "bancs_token_required", True)

    refused = client.post("/api/build/runs", json={"task": "t"})
    assert refused.status_code == 401 and "Accesstoken" in refused.json()["error"]

    started = client.post("/api/build/runs", json={"task": "t"}, headers={"Accesstoken": "visitor-token-1"})
    assert started.status_code == 202
    assert build_proxy.RUNS[build_routes._RUNS[RUN_ID].token].bank_token == "visitor-token-1"
    # The operator (evals, smoke runs) does not need one.
    assert client.post("/api/build/runs", json={"task": "c"},
                       headers={"Authorization": "Bearer " + "x" * 40}).status_code == 202
