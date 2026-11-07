"""HTTP layer of the mobile interface: authentication, CSRF, network/host
restrictions, request validation and the chat/voice/confirmation routes.
Runs the aiohttp app in-process (aiohttp's test server on 127.0.0.1)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.core.events import EventBus
from app.core.exceptions import ConfigurationError
from app.mobile.network import is_client_allowed, parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.server import COOKIE_NAME, SECURITY_HEADERS, create_app
from app.mobile.service import MobileService
from tests.conftest import final_response, tool_call_response
from tests.mobile_fakes import FakeSTT, FakeTTS, GatedLLM, HighRiskTool, make_agent, make_voice, make_wav, wait_until

POST_HEADERS = {"X-Garvis-Client": "1"}


class _Harness:
    def __init__(self, client: TestClient, service: MobileService, pairing: PairingManager, stt, tts, llm, tool):
        self.client = client
        self.service = service
        self.pairing = pairing
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.tool = tool
        self.token: str | None = None

    async def pair(self) -> str:
        code = self.pairing.new_pairing_code()
        response = await self.client.post("/api/pair", json={"code": code.code}, headers=POST_HEADERS)
        assert response.status == 200
        self.token = response.cookies[COOKIE_NAME].value
        self.client.session.cookie_jar.clear()  # every request below sends the cookie explicitly
        return self.token

    def auth(self, **extra) -> dict[str, str]:
        return {"Cookie": f"{COOKIE_NAME}={self.token}", **POST_HEADERS, **extra}


async def _start(tmp_path, *, llm=None, stt=None, tts=None, networks="", hosts=None, max_audio_bytes=1_000_000):
    stt = stt or FakeSTT()
    tts = tts or FakeTTS()
    tool = HighRiskTool()
    llm = llm or GatedLLM(gated=False)
    events = EventBus()
    agent = make_agent(tmp_path, llm, events=events, tools=[tool])
    service = MobileService(agent, events, make_voice(stt, tts), max_audio_seconds=10)
    service.start()
    pairing = PairingManager(tmp_path / "devices.json")
    app = create_app(
        service,
        pairing,
        allowed_networks=parse_networks(networks),
        allowed_hosts=hosts,
        ca_cert_der=b"0\x82fake-der",
        max_audio_bytes=max_audio_bytes,
    )
    client = TestClient(TestServer(app, host="127.0.0.1"))
    await client.start_server()
    return _Harness(client, service, pairing, stt, tts, llm, tool)


@pytest.fixture
async def harness(tmp_path):
    h = await _start(tmp_path)
    yield h
    await h.service.stop()
    await h.client.close()


async def _events(h: _Harness, after: int = 0) -> dict:
    response = await h.client.get(f"/api/events?after={after}", headers=h.auth())
    assert response.status == 200
    return await response.json()


# ---------------------------------------------------------- pages & headers
async def test_index_and_static_assets_are_served_with_security_headers(harness):
    response = await harness.client.get("/")
    assert response.status == 200
    assert "text/html" in response.headers["Content-Type"]
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    for name in ("app.js", "style.css", "recorder-worklet.js"):
        assert (await harness.client.get(f"/static/{name}")).status == 200


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/index.py", "/static/%2e%2e/tls.py", "/api/nothing"])
async def test_no_other_files_or_routes_are_reachable(harness, path):
    response = await harness.client.get(path)
    assert response.status == 404


async def test_ca_certificate_download(harness):
    response = await harness.client.get("/garvis-ca.crt")
    assert response.status == 200
    assert response.headers["Content-Type"] == "application/x-x509-ca-cert"
    assert await response.read() == b"0\x82fake-der"


# ----------------------------------------------------------- authentication
async def test_api_requires_a_paired_device(harness):
    assert (await harness.client.get("/api/events")).status == 401
    response = await harness.client.post("/api/messages", json={"text": "hi"}, headers=POST_HEADERS)
    assert response.status == 401
    response = await harness.client.get("/api/events", headers={"Cookie": f"{COOKIE_NAME}=forged"})
    assert response.status == 401
    assert (await harness.client.get("/api/turns/abc/speech")).status == 401
    response = await harness.client.post(
        "/api/confirmations/00000000-0000-0000-0000-000000000000", json={"approved": True}, headers=POST_HEADERS
    )
    assert response.status == 401
    assert harness.llm.calls == []


async def test_pairing_sets_a_hardened_cookie(harness):
    assert (await (await harness.client.get("/api/session")).json()) == {"paired": False}
    code = harness.pairing.new_pairing_code()
    response = await harness.client.post("/api/pair", json={"code": code.code}, headers=POST_HEADERS)
    assert response.status == 200
    set_cookie = response.headers["Set-Cookie"]
    for attribute in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/"):
        assert attribute in set_cookie
    token = response.cookies[COOKIE_NAME].value
    response = await harness.client.get("/api/session", headers={"Cookie": f"{COOKIE_NAME}={token}"})
    assert (await response.json()) == {"paired": True}


async def test_wrong_pairing_code_and_malformed_pairing_requests(harness):
    harness.pairing.new_pairing_code()
    response = await harness.client.post("/api/pair", json={"code": "not-it"}, headers=POST_HEADERS)
    assert response.status == 403
    assert "Wrong" in (await response.json())["error"]
    assert (await harness.client.post("/api/pair", json={"code": 123}, headers=POST_HEADERS)).status == 400
    assert (await harness.client.post("/api/pair", data="code=1", headers=POST_HEADERS)).status == 400


async def test_unpair_revokes_the_device(harness):
    await harness.pair()
    response = await harness.client.post("/api/unpair", headers=harness.auth())
    assert response.status == 200
    assert (await harness.client.get("/api/events", headers=harness.auth())).status == 401


async def test_post_without_client_header_or_from_another_origin_is_refused(harness):
    await harness.pair()
    cookie = {"Cookie": f"{COOKIE_NAME}={harness.token}"}
    response = await harness.client.post("/api/messages", json={"text": "hi"}, headers=cookie)
    assert response.status == 403
    response = await harness.client.post(
        "/api/messages", json={"text": "hi"}, headers=harness.auth(Origin="https://evil.example")
    )
    assert response.status == 403
    origin = f"http://{harness.client.host}:{harness.client.port}"
    response = await harness.client.post("/api/messages", json={"text": "hi"}, headers=harness.auth(Origin=origin))
    assert response.status == 202
    code = harness.pairing.new_pairing_code()
    response = await harness.client.post("/api/pair", json={"code": code.code})
    assert response.status == 403


# --------------------------------------------------- network & host limits
async def test_clients_outside_the_allowed_networks_are_refused(tmp_path):
    h = await _start(tmp_path, networks="10.0.0.0/8")  # the test client is 127.0.0.1
    try:
        for path in ("/", "/api/session", "/garvis-ca.crt"):
            assert (await h.client.get(path)).status == 403
    finally:
        await h.service.stop()
        await h.client.close()


async def test_unknown_host_names_are_refused(tmp_path):
    h = await _start(tmp_path, hosts={"garvis-pc.local", "192.168.1.20"})
    try:
        assert (await h.client.get("/")).status == 421
        assert (await h.client.get("/", headers={"Host": "attacker.example"})).status == 421
        assert (await h.client.get("/", headers={"Host": "garvis-pc.local:8443"})).status == 200
        assert (await h.client.get("/", headers={"Host": "192.168.1.20:8443"})).status == 200
    finally:
        await h.service.stop()
        await h.client.close()


@pytest.mark.parametrize(
    "remote, allowed",
    [
        ("192.168.1.7", True),
        ("10.1.2.3", True),
        ("172.20.0.5", True),
        ("127.0.0.1", True),
        ("::ffff:192.168.1.7", True),
        ("fe80::1%12", True),
        ("8.8.8.8", False),
        ("100.64.0.1", False),
        ("2001:4860:4860::8888", False),
        ("", False),
        (None, False),
        ("not-an-ip", False),
    ],
)
def test_default_network_allowlist(remote, allowed):
    assert is_client_allowed(remote, parse_networks("")) is allowed


def test_invalid_network_setting_fails_clearly():
    with pytest.raises(ConfigurationError):
        parse_networks("192.168.1.0/24, nonsense")


# --------------------------------------------------------------- messages
async def test_message_runs_a_turn_and_the_reply_arrives_via_events(harness):
    await harness.pair()
    response = await harness.client.post("/api/messages", json={"text": "hello"}, headers=harness.auth())
    assert response.status == 202
    turn_id = (await response.json())["turn_id"]
    await wait_until(lambda: harness.service.get_turn(turn_id).done)
    kinds = [(e["kind"], e.get("text")) for e in (await _events(harness))["events"]]
    assert ("user", "hello") in kinds and ("assistant", "reply to: hello") in kinds


@pytest.mark.parametrize(
    "kwargs, status",
    [
        ({"json": {"text": ""}}, 400),
        ({"json": {"text": "x" * 5000}}, 400),
        ({"json": {"message": "hi"}}, 400),
        ({"json": ["hi"]}, 400),
        ({"data": "hi"}, 400),
        ({"data": "{broken", "headers": {"Content-Type": "application/json"}}, 400),
    ],
    ids=["empty", "too-long", "wrong-field", "not-object", "not-json", "broken-json"],
)
async def test_invalid_messages_are_rejected(harness, kwargs, status):
    await harness.pair()
    headers = harness.auth(**kwargs.pop("headers", {}))
    response = await harness.client.post("/api/messages", headers=headers, **kwargs)
    assert response.status == status
    assert "error" in await response.json()
    assert harness.llm.calls == []


async def test_concurrent_message_gets_409_and_does_not_touch_the_context(tmp_path):
    llm = GatedLLM()
    h = await _start(tmp_path, llm=llm)
    try:
        await h.pair()
        first, second = await asyncio.gather(
            h.client.post("/api/messages", json={"text": "one"}, headers=h.auth()),
            h.client.post("/api/messages", json={"text": "two"}, headers=h.auth()),
        )
        assert sorted([first.status, second.status]) == [202, 409]
        llm.release()
        await wait_until(lambda: not h.service.is_busy)
        assert len(llm.calls) == 1
    finally:
        await h.service.stop()
        await h.client.close()


async def test_events_long_poll_parameters_are_validated(harness):
    await harness.pair()
    response = await harness.client.get("/api/events?after=abc", headers=harness.auth())
    assert response.status == 400
    response = await harness.client.get("/api/events?after=0&wait=0.05", headers=harness.auth())
    data = await response.json()
    assert set(data) >= {"events", "last_seq", "status", "busy", "pending_confirmations", "reset"}


# ------------------------------------------------------------------- voice
async def test_voice_upload_transcribes_runs_a_turn_and_speech_can_be_fetched(harness):
    await harness.pair()
    response = await harness.client.post(
        "/api/voice", data=make_wav(1.0), headers=harness.auth(**{"Content-Type": "audio/wav"})
    )
    assert response.status == 202
    body = await response.json()
    assert body["transcript"] == "what time is it"
    assert not Path(harness.stt.paths[0]).exists()

    await wait_until(lambda: harness.service.get_turn(body["turn_id"]).done)
    speech = await harness.client.get(f"/api/turns/{body['turn_id']}/speech", headers=harness.auth())
    assert speech.status == 200
    assert speech.headers["Content-Type"] == "audio/wav"
    assert (await speech.read())[:4] == b"RIFF"
    assert harness.tts.texts == ["reply to: what time is it"]


@pytest.mark.parametrize(
    "data, content_type, status",
    [
        (make_wav(1.0), "audio/mp4", 415),
        (b"", "audio/wav", 422),
        (b"RIFF\x00\x00\x00\x00WAVEjunk", "audio/wav", 400),
        (b"OggS" + b"\x00" * 64, "audio/wav", 415),
        (make_wav(0.1), "audio/wav", 422),
        (make_wav(11.0), "audio/wav", 413),
    ],
    ids=["wrong-type", "empty", "corrupt", "not-wav", "too-short", "too-long"],
)
async def test_invalid_voice_uploads_are_rejected_without_stt(harness, data, content_type, status):
    await harness.pair()
    response = await harness.client.post(
        "/api/voice", data=data, headers=harness.auth(**{"Content-Type": content_type})
    )
    assert response.status == status
    assert "error" in await response.json()
    assert harness.stt.paths == []


async def test_oversized_voice_upload_is_rejected(tmp_path):
    h = await _start(tmp_path, max_audio_bytes=10_000)
    try:
        await h.pair()
        response = await h.client.post(
            "/api/voice", data=make_wav(1.0), headers=h.auth(**{"Content-Type": "audio/wav"})
        )
        assert response.status == 413
        assert h.stt.paths == []
    finally:
        await h.service.stop()
        await h.client.close()


async def test_stt_failure_returns_502_and_no_turn(tmp_path):
    h = await _start(tmp_path, stt=FakeSTT(error="Transcription failed: boom"))
    try:
        await h.pair()
        response = await h.client.post(
            "/api/voice", data=make_wav(1.0), headers=h.auth(**{"Content-Type": "audio/wav"})
        )
        assert response.status == 502
        assert "boom" in (await response.json())["error"]
        assert h.llm.calls == []
        assert not Path(h.stt.paths[0]).exists()
    finally:
        await h.service.stop()
        await h.client.close()


async def test_silence_with_no_transcript_returns_422_and_no_turn(tmp_path):
    h = await _start(tmp_path, stt=FakeSTT(text="   "))
    try:
        await h.pair()
        response = await h.client.post(
            "/api/voice", data=make_wav(1.0), headers=h.auth(**{"Content-Type": "audio/wav"})
        )
        assert response.status == 422
        assert h.llm.calls == []
    finally:
        await h.service.stop()
        await h.client.close()


async def test_tts_failure_returns_502(tmp_path):
    h = await _start(tmp_path, tts=FakeTTS(error="Voice model not found"))
    try:
        await h.pair()
        response = await h.client.post("/api/messages", json={"text": "hi"}, headers=h.auth())
        turn_id = (await response.json())["turn_id"]
        await wait_until(lambda: h.service.get_turn(turn_id).done)
        speech = await h.client.get(f"/api/turns/{turn_id}/speech", headers=h.auth())
        assert speech.status == 502
        assert "Voice model" in (await speech.json())["error"]
    finally:
        await h.service.stop()
        await h.client.close()


@pytest.mark.parametrize("turn_id", ["unknown", "bad$id", "x" * 40])
async def test_speech_for_unknown_turns_is_404(harness, turn_id):
    await harness.pair()
    response = await harness.client.get(f"/api/turns/{turn_id}/speech", headers=harness.auth())
    assert response.status == 404


# ----------------------------------------------------------- confirmations
async def _pending(h: _Harness) -> dict:
    h.llm._responses[:] = [tool_call_response("high_risk_echo", {}), final_response("done")]
    response = await h.client.post("/api/messages", json={"text": "do it"}, headers=h.auth())
    assert response.status == 202
    await wait_until(lambda: h.service.snapshot()["pending_confirmations"])
    return (await _events(h))["pending_confirmations"][0]


async def test_confirmation_is_shown_and_approved_from_the_phone(harness):
    await harness.pair()
    request = await _pending(harness)
    assert request["tool_name"] == "high_risk_echo"
    assert harness.tool.executions == 0
    response = await harness.client.post(
        f"/api/confirmations/{request['id']}", json={"approved": True}, headers=harness.auth()
    )
    assert response.status == 200
    await wait_until(lambda: harness.tool.executions == 1)
    await wait_until(lambda: not harness.service.is_busy)
    assert (await _events(harness))["pending_confirmations"] == []


async def test_confirmation_is_rejected_from_the_phone(harness):
    await harness.pair()
    request = await _pending(harness)
    response = await harness.client.post(
        f"/api/confirmations/{request['id']}", json={"approved": False}, headers=harness.auth()
    )
    assert response.status == 200
    await wait_until(lambda: not harness.service.is_busy)
    assert harness.tool.executions == 0
    outcomes = [e["outcome"] for e in (await _events(harness))["events"] if e["kind"] == "confirmation_resolved"]
    assert outcomes == ["denied"]


async def test_confirmation_requests_are_validated(harness):
    await harness.pair()
    request = await _pending(harness)
    url = f"/api/confirmations/{request['id']}"
    assert (await harness.client.post(url, json={"approved": "yes"}, headers=harness.auth())).status == 400
    assert (await harness.client.post(url, json={}, headers=harness.auth())).status == 400
    bad_id = "/api/confirmations/not-a-uuid"
    assert (await harness.client.post(bad_id, json={"approved": True}, headers=harness.auth())).status == 400
    unknown = "/api/confirmations/00000000-0000-0000-0000-000000000000"
    assert (await harness.client.post(unknown, json={"approved": True}, headers=harness.auth())).status == 404
    assert harness.tool.executions == 0
    await harness.client.post(url, json={"approved": False}, headers=harness.auth())
    await wait_until(lambda: not harness.service.is_busy)
