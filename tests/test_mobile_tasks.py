"""The phone's task and protocol API (app/mobile/server.py + service.py on a
real TaskManager): authentication and CSRF on every new route, retry-safe
submissions, desktop-side validation of protocol requests, cancellation,
what the phone may see, and the PWA shell routes.

In-process aiohttp test server on 127.0.0.1; a real Agent with fake tools
and a gated/scripted LLM (no model, desktop or network)."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.core.events import EventBus
from app.mobile.network import parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.server import COOKIE_NAME, create_app
from app.mobile.service import MobileService
from app.tasks.manager import TaskManager
from app.tasks.protocols import Protocol, ProtocolStore, Step
from tests.mobile_fakes import GatedLLM, make_agent, make_voice, wait_until
from tests.test_tasks_and_protocols import FakeAppTool, FakeWebsiteTool

POST = {"X-Garvis-Client": "1"}


class _Phone:
    def __init__(self, client, service, manager, store, pairing, llm, web):
        self.client, self.service, self.manager, self.store = client, service, manager, store
        self.pairing, self.llm, self.web = pairing, llm, web
        self.token = ""

    async def pair(self) -> None:
        code = self.pairing.new_pairing_code()
        response = await self.client.post("/api/pair", json={"code": code.code}, headers=POST)
        self.token = response.cookies[COOKIE_NAME].value
        self.client.session.cookie_jar.clear()

    def auth(self, post: bool = True) -> dict:
        headers = {"Cookie": f"{COOKIE_NAME}={self.token}"}
        return {**headers, **POST} if post else headers


@pytest.fixture
async def phone(tmp_path):
    events = EventBus()
    llm = GatedLLM(gated=False)
    web = FakeWebsiteTool()
    agent = make_agent(tmp_path, llm, events=events, tools=[web, FakeAppTool()])
    store = ProtocolStore(tmp_path / "protocols")
    manager = TaskManager(
        agent, events, tmp_path / "tasks.json", store, available_tools=lambda: {"open_website", "open_application"}
    )
    manager.start()
    service = MobileService(
        agent,
        events,
        make_voice(),
        tasks=manager,
        protocols=store,
        available_tools=lambda: {"open_website", "open_application"},
    )
    service.start()
    pairing = PairingManager(tmp_path / "devices.json")
    app = create_app(
        service,
        pairing,
        allowed_networks=parse_networks(""),
        allowed_hosts=None,
        ca_cert_der=None,
        max_audio_bytes=1_000_000,
    )
    client = TestClient(TestServer(app, host="127.0.0.1"))
    await client.start_server()
    p = _Phone(client, service, manager, store, pairing, llm, web)
    await p.pair()
    yield p
    await service.stop()
    await manager.stop()
    await client.close()


def _protocol(store: ProtocolStore, **fields) -> Protocol:
    steps = fields.pop("steps", [Step(type="open_website", params={"url": "https://weather.example"})])
    return store.save(Protocol(name=fields.pop("name", "Weather"), steps=steps, **fields))


async def test_every_new_route_requires_a_paired_device(phone):
    protocol = _protocol(phone.store)
    for method, path in (
        ("GET", "/api/tasks"),
        ("GET", "/api/tasks/abcdefabcdef"),
        ("GET", "/api/protocols"),
        ("POST", "/api/tasks/abcdefabcdef/cancel"),
        ("POST", f"/api/protocols/{protocol.id}/run"),
    ):
        response = await phone.client.request(method, path, headers=POST, json={})
        assert response.status == 401, path
    assert phone.manager.list() == []


async def test_posts_without_the_client_header_are_refused(phone):
    protocol = _protocol(phone.store)
    response = await phone.client.post(
        f"/api/protocols/{protocol.id}/run", json={}, headers={"Cookie": f"{COOKIE_NAME}={phone.token}"}
    )
    assert response.status == 403
    response = await phone.client.post(
        f"/api/protocols/{protocol.id}/run", json={}, headers={**phone.auth(), "Origin": "https://evil.example"}
    )
    assert response.status == 403
    assert phone.manager.list() == []


async def test_instruction_becomes_a_task_and_a_retry_does_not_run_it_twice(phone):
    body = {"text": "open the weather", "request_id": "abcd1234efgh5678"}
    first = await phone.client.post("/api/messages", json=body, headers=phone.auth())
    again = await phone.client.post("/api/messages", json=body, headers=phone.auth())
    assert first.status == again.status == 202
    turn_id = (await first.json())["turn_id"]
    assert (await again.json())["turn_id"] == turn_id
    await wait_until(lambda: phone.manager.get(turn_id).terminal)
    detail = await (await phone.client.get(f"/api/tasks/{turn_id}", headers=phone.auth(post=False))).json()
    assert detail["status"] == "completed" and detail["result"] == "reply to: open the weather"
    assert len(phone.llm.calls) == 1
    bad = await phone.client.post("/api/messages", json={"text": "x", "request_id": "no spaces!"}, headers=phone.auth())
    assert bad.status == 400


async def test_tasks_list_and_detail_carry_status_steps_and_what_was_shown(phone):
    protocol = _protocol(phone.store)
    response = await phone.client.post(
        f"/api/protocols/{protocol.id}/run", json={"request_id": "run-request-0001"}, headers=phone.auth()
    )
    assert response.status == 202
    task = await response.json()
    await wait_until(lambda: phone.manager.get(task["id"]).terminal)
    listing = await (await phone.client.get("/api/tasks", headers=phone.auth(post=False))).json()
    assert listing["tasks"][0]["id"] == task["id"] and listing["tasks"][0]["status"] == "completed"
    assert listing["version"] == phone.manager.version
    detail = await (await phone.client.get(f"/api/tasks/{task['id']}", headers=phone.auth(post=False))).json()
    assert detail["steps"][0]["status"] == "ok"
    assert detail["opened"][0]["title"] == "Forecast - Browser"
    assert "handle" not in json.dumps(detail["opened"])  # window handles stay on the PC
    assert phone.web.calls == ["https://weather.example"]


async def test_protocol_requests_are_validated_on_the_desktop(phone):
    disabled = _protocol(phone.store, name="Off", enabled=False)
    response = await phone.client.post(f"/api/protocols/{disabled.id}/run", json={}, headers=phone.auth())
    assert response.status == 409 and "disabled" in (await response.json())["error"]
    gone = await phone.client.post(f"/api/protocols/{'a' * 16}/run", json={}, headers=phone.auth())
    assert gone.status == 404
    traversal = await phone.client.post("/api/protocols/..%2F..%2Fsecrets/run", json={}, headers=phone.auth())
    assert traversal.status in (400, 404)
    assert phone.manager.list() == [] and phone.web.calls == []


async def test_protocol_listing_shows_only_what_the_phone_needs(phone):
    _protocol(
        phone.store,
        name="Project",
        steps=[Step(type="open_path", params={"path": "C:\\private\\stuff"}, label="Open my project")],
    )
    data = await (await phone.client.get("/api/protocols", headers=phone.auth(post=False))).json()
    entry = data["protocols"][0]
    assert entry["name"] == "Project" and entry["steps"] == ["Open my project"]
    assert "private" not in json.dumps(data) and "params" not in entry


async def test_cancel_from_the_phone_stops_the_pending_desktop_action(phone):
    from tests.conftest import tool_call_response

    phone.llm.gate.clear()
    phone.llm._responses = [tool_call_response("open_website", {"url": "https://never.example"})]
    response = await phone.client.post(
        "/api/messages", json={"text": "slow", "request_id": "slow-request-01"}, headers=phone.auth()
    )
    task_id = (await response.json())["turn_id"]
    await wait_until(lambda: phone.llm.in_flight == 1)
    cancel = await phone.client.post(f"/api/tasks/{task_id}/cancel", headers=phone.auth())
    assert cancel.status == 202 and (await cancel.json())["status"] == "cancelling"
    phone.llm.release()
    await wait_until(lambda: phone.manager.get(task_id).terminal)
    assert phone.manager.get(task_id).status == "cancelled" and phone.web.calls == []
    again = await phone.client.post(f"/api/tasks/{task_id}/cancel", headers=phone.auth())
    assert again.status == 409
    unknown = await phone.client.post("/api/tasks/zzzzzzzzzzzz/cancel", headers=phone.auth())
    assert unknown.status == 404


async def test_task_changes_wake_the_phones_long_poll(phone):
    snapshot = await (await phone.client.get("/api/events?after=0", headers=phone.auth(post=False))).json()
    version = snapshot["tasks_version"]
    phone.manager.submit_instruction("hello", origin="mobile")
    data = await (
        await phone.client.get(f"/api/events?after={snapshot['last_seq']}&wait=2", headers=phone.auth(post=False))
    ).json()
    assert data["tasks_version"] > version


async def test_pwa_shell_is_served_with_a_policy_that_allows_it(phone):
    sw = await phone.client.get("/sw.js")
    assert sw.status == 200 and "javascript" in sw.headers["Content-Type"]
    assert "/api/" in await sw.text()  # it explicitly never caches the API
    manifest = await phone.client.get("/manifest.webmanifest")
    assert manifest.status == 200 and (await manifest.json())["display"] == "standalone"
    icon = await phone.client.get("/static/icon-180.png")
    assert icon.status == 200 and icon.headers["Content-Type"] == "image/png"
    csp = (await phone.client.get("/")).headers["Content-Security-Policy"]
    assert "worker-src 'self'" in csp and "manifest-src 'self'" in csp and "script-src 'self'" in csp
