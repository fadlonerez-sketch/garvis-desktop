"""/api/v1 -- the native Android app's API (app/mobile/native_api.py) on a
real TaskManager, MobileService, ReminderStore, ImageService and
PhoneActionBroker: bearer authentication, pairing/rotation/revocation,
retry-safe submissions, the event stream, reminders, images, phone actions
and the Tailscale-only remote guard.

In-process aiohttp test server on 127.0.0.1; a real Agent with a gated/
scripted LLM and fake STT/TTS (no model, desktop, microphone or network)."""
from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.core.events import EventBus
from app.images.service import ImageService
from app.mobile import server as server_module
from app.mobile.companion import Companion
from app.mobile.native_api import AUTH_FAILURE_LIMIT, AuthFailureLimiter
from app.mobile.network import is_tailscale_address, is_tailscale_connection, parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.phone_actions import PhoneActionBroker
from app.mobile.server import COOKIE_NAME, create_app
from app.mobile.service import MobileService
from app.reminders.store import ReminderStore
from app.reminders.tools import ReminderZone
from app.tasks.manager import TaskManager
from app.tasks.protocols import ProtocolStore
from tests.mobile_fakes import FakeSTT, FakeTTS, GatedLLM, make_agent, make_voice, make_wav, wait_until
from tests.test_images_and_phone_actions import FakeProvider


class _App:
    def __init__(self, **parts) -> None:
        self.__dict__.update(parts)
        self.token = ""

    def auth(self, **extra) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", **extra}

    async def pair(self) -> dict:
        code = self.pairing.new_pairing_code()
        response = await self.client.post("/api/v1/pair", json={"code": code.code, "device_name": "Example Phone"})
        assert response.status == 201
        data = await response.json()
        self.token = data["token"]
        return data

    async def get(self, path: str, **kwargs):
        return await self.client.get(path, headers=self.auth(), **kwargs)

    async def post(self, path: str, json=None, **kwargs):
        return await self.client.post(path, json=json if json is not None else {}, headers=self.auth(), **kwargs)


async def _start(tmp_path, *, image_provider=None, networks="", remote_access=False, llm=None, stt=None):
    events = EventBus()
    llm = llm or GatedLLM(gated=False)
    agent = make_agent(tmp_path, llm, events=events)
    protocols = ProtocolStore(tmp_path / "protocols")
    tasks = TaskManager(agent, events, tmp_path / "tasks.json", protocols, available_tools=lambda: set())
    tasks.start()
    stt = stt or FakeSTT("open my project and run its tests")
    tts = FakeTTS()
    service = MobileService(agent, events, make_voice(stt, tts), tasks=tasks, protocols=protocols)
    service.start()
    companion = Companion(
        reminders=ReminderStore(tmp_path / "reminders.json"),
        zone=ReminderZone(""),
        images=ImageService(tmp_path / "images", image_provider, "Image generation isn't set up."),
        phone=PhoneActionBroker(),
    )
    companion.reminders.add_listener(service.poke)
    companion.phone.add_listener(service.poke)
    companion.images.add_listener(lambda job: service.post_entry("image", image_id=job.id)
                                  if job.status == "completed" else service.poke())
    pairing = PairingManager(tmp_path / "devices.json")
    app = create_app(service, pairing, allowed_networks=parse_networks(networks), allowed_hosts=None,
                     ca_cert_der=None, max_audio_bytes=1_000_000, companion=companion, remote_access=remote_access)
    client = TestClient(TestServer(app, host="127.0.0.1"))
    await client.start_server()
    return _App(client=client, service=service, tasks=tasks, companion=companion, pairing=pairing, llm=llm,
                stt=stt, tts=tts)


@pytest.fixture
async def phone(tmp_path):
    h = await _start(tmp_path)
    await h.pair()
    yield h
    await h.service.stop()
    await h.tasks.stop()
    await h.client.close()


# ------------------------------------------------------------ authentication
async def test_hello_is_public_but_says_nothing_private(tmp_path):
    h = await _start(tmp_path)
    try:
        response = await h.client.get("/api/v1/hello")
        assert response.status == 200
        assert await response.json() == {"api": 1, "server": "garvis", "paired": False, "remote": False}
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


ROUTES = [("GET", "/api/v1/status"), ("GET", "/api/v1/events"), ("POST", "/api/v1/messages"),
          ("POST", "/api/v1/voice"), ("GET", "/api/v1/tasks"), ("GET", "/api/v1/tasks/abc"),
          ("POST", "/api/v1/tasks/abc/cancel"), ("GET", "/api/v1/reminders"), ("POST", "/api/v1/reminders"),
          ("POST", "/api/v1/reminders/abc/snooze"), ("GET", "/api/v1/images"), ("POST", "/api/v1/images"),
          ("GET", "/api/v1/images/abc/file"), ("POST", "/api/v1/phone/capabilities"),
          ("POST", "/api/v1/token/refresh"), ("POST", "/api/v1/unpair"), ("GET", "/api/v1/protocols")]


@pytest.mark.parametrize("method,path", ROUTES)
async def test_every_route_needs_a_valid_bearer_token(phone, method, path):
    for headers in ({}, {"Authorization": "Bearer not-a-token"}, {"Authorization": phone.token},
                    {"Cookie": f"{COOKIE_NAME}={phone.token}", "X-Garvis-Client": "1"}):
        response = await phone.client.request(method, path, headers=headers, json={})
        assert response.status == 401, (headers, path)
        assert (await response.json())["code"] == "unpaired"


async def test_wrong_pairing_code_and_single_use(tmp_path):
    h = await _start(tmp_path)
    try:
        code = h.pairing.new_pairing_code()
        wrong = "00000000" if code.code != "00000000" else "11111111"
        response = await h.client.post("/api/v1/pair", json={"code": wrong})
        assert response.status == 403
        assert (await h.client.post("/api/v1/pair", json={"code": code.code})).status == 201
        assert (await h.client.post("/api/v1/pair", json={"code": code.code})).status == 403  # single use
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_token_rotation_and_unpair(phone):
    old = phone.token
    response = await phone.post("/api/v1/token/refresh")
    assert response.status == 200
    phone.token = (await response.json())["token"]
    assert phone.token != old
    assert (await phone.client.get("/api/v1/status", headers={"Authorization": f"Bearer {old}"})).status == 401
    assert (await phone.get("/api/v1/status")).status == 200
    assert (await phone.post("/api/v1/unpair")).status == 200
    assert (await phone.get("/api/v1/status")).status == 401


async def test_desktop_can_revoke_one_device(phone):
    (device,) = phone.pairing.devices()
    assert phone.pairing.revoke_device(device.id)
    assert (await phone.get("/api/v1/status")).status == 401


async def test_status_reports_state_features_and_time_zone(phone):
    response = await phone.client.get("/api/v1/status", headers=phone.auth(**{"X-Garvis-Timezone": "Asia/Jerusalem"}))
    data = await response.json()
    assert data["garvis"]["state"] == "online"
    assert data["features"]["images"] == {"available": False, "provider": "",
                                          "reason": "Image generation isn't set up."}
    assert data["time_zone"] == "Asia/Jerusalem"
    assert data["token_expires_at"] > time.time()


# -------------------------------------------------------------- conversation
async def _events(phone, after=0, **params) -> dict:
    query = "&".join(f"{k}={v}" for k, v in {"after": after, **params}.items())
    response = await phone.get(f"/api/v1/events?{query}")
    assert response.status == 200
    return await response.json()


async def test_message_becomes_a_task_once_and_the_reply_is_linked(phone):
    body = {"text": "Summarize the document on my desktop", "request_id": "msg-0000001"}
    first = await phone.post("/api/v1/messages", body)
    assert first.status == 202
    first = await first.json()
    again = await (await phone.post("/api/v1/messages", body)).json()
    assert again["turn_id"] == first["turn_id"]  # a retry never runs the request twice
    assert first["task"]["status"] in ("queued", "running", "completed")
    await wait_until(lambda: phone.tasks.get(first["turn_id"]).terminal)
    assert len(phone.llm.calls) == 1
    snapshot = await _events(phone)
    user = [e for e in snapshot["events"] if e["kind"] == "user"]
    assistant = [e for e in snapshot["events"] if e["kind"] == "assistant"]
    assert user[-1]["turn_id"] == first["turn_id"] and assistant[-1]["turn_id"] == first["turn_id"]
    assert assistant[-1]["text"].startswith("reply to:")
    assert phone.tasks.get(first["turn_id"]).status == "completed"
    assert snapshot["epoch"] == phone.service.epoch


async def test_invalid_messages_are_refused(phone):
    assert (await phone.post("/api/v1/messages", {"text": "   "})).status == 400
    assert (await phone.post("/api/v1/messages", {"text": "hi", "request_id": "bad id!"})).status == 400
    assert (await phone.post("/api/v1/messages", {"words": "hi"})).status == 400


async def test_cancel_a_running_task(tmp_path):
    h = await _start(tmp_path, llm=GatedLLM(gated=True))
    await h.pair()
    try:
        submitted = await (await h.post("/api/v1/messages", {"text": "long job"})).json()
        await wait_until(lambda: h.tasks.get(submitted["turn_id"]).status == "running")
        response = await h.post(f"/api/v1/tasks/{submitted['turn_id']}/cancel")
        assert response.status == 202 and (await response.json())["status"] == "cancelling"
        h.llm.release()
        await wait_until(lambda: h.tasks.get(submitted["turn_id"]).terminal)
        assert h.tasks.get(submitted["turn_id"]).status == "cancelled"
        assert (await h.post(f"/api/v1/tasks/{submitted['turn_id']}/cancel")).status == 409
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_events_long_poll_waits_and_wakes(phone):
    snapshot = await _events(phone)
    known = {"after": snapshot["last_seq"], "epoch": snapshot["epoch"], "rv": snapshot["reminders_version"],
             "iv": snapshot["images_version"], "tv": snapshot["tasks_version"]}
    started = time.monotonic()
    await _events(phone, **{**known, "after": known.pop("after"), "wait": 0.3})
    assert time.monotonic() - started >= 0.25  # nothing new: it waited

    async def later():
        await asyncio.sleep(0.1)
        phone.companion.reminders.create("wake up", tz="UTC", due_at=time.time() + 600)

    asyncio.ensure_future(later())
    started = time.monotonic()
    snapshot2 = await _events(phone, after=snapshot["last_seq"], epoch=snapshot["epoch"],
                              rv=snapshot["reminders_version"], iv=snapshot["images_version"],
                              tv=snapshot["tasks_version"], wait=5)
    assert time.monotonic() - started < 3
    assert snapshot2["reminders_version"] == snapshot["reminders_version"] + 1


async def test_events_reset_when_garvis_restarted(phone):
    snapshot = await _events(phone, after=5, epoch="0123456789ab")
    assert snapshot["reset"] is True


async def test_voice_upload_is_transcribed_once_per_request_id(phone):
    headers = phone.auth(**{"Content-Type": "audio/wav", "X-Request-Id": "voice-000001"})
    first = await phone.client.post("/api/v1/voice", data=make_wav(1.0), headers=headers)
    assert first.status == 202
    first = await first.json()
    assert first["transcript"] == "open my project and run its tests"
    again = await (await phone.client.post("/api/v1/voice", data=make_wav(1.0), headers=headers)).json()
    assert again["turn_id"] == first["turn_id"]
    assert len(phone.stt.paths) == 1
    await wait_until(lambda: phone.tasks.get(first["turn_id"]).terminal)
    speech = await phone.get(f"/api/v1/tasks/{first['turn_id']}/speech")
    assert speech.status == 200 and (await speech.read())[:4] == b"RIFF"


async def test_voice_rejects_bad_audio(phone):
    response = await phone.client.post("/api/v1/voice", data=b"garbage",
                                       headers=phone.auth(**{"Content-Type": "audio/wav"}))
    assert response.status == 415 and (await response.json())["code"] == "audio_unsupported"
    response = await phone.client.post("/api/v1/voice", data=b"x", headers=phone.auth(**{"Content-Type": "audio/ogg"}))
    assert response.status == 415


# ---------------------------------------------------------------- reminders
async def test_reminder_lifecycle_over_the_api(phone):
    due = time.time() + 3600
    created = await phone.post("/api/v1/reminders", {"text": "Meeting", "kind": "call", "due_at": due,
                                                     "tz": "Asia/Jerusalem", "request_id": "rem-0000001"})
    assert created.status == 201
    reminder = (await created.json())["reminder"]
    duplicate = await (await phone.post("/api/v1/reminders", {"text": "Meeting", "kind": "call", "due_at": due,
                                                              "request_id": "rem-0000001"})).json()
    assert duplicate["reminder"]["id"] == reminder["id"]
    listed = await (await phone.get("/api/v1/reminders")).json()
    assert [r["id"] for r in listed["reminders"]] == [reminder["id"]]

    snoozed = await (await phone.post(f"/api/v1/reminders/{reminder['id']}/snooze", {"minutes": 10})).json()
    assert snoozed["reminder"]["snoozed"] is True
    occurrence = snoozed["reminder"]["due_at"]
    fired = await (await phone.post(f"/api/v1/reminders/{reminder['id']}/fired", {"occurrence": occurrence})).json()
    assert fired["reminder"]["status"] == "completed"
    again = await (await phone.post(f"/api/v1/reminders/{reminder['id']}/fired", {"occurrence": occurrence})).json()
    assert again["reminder"]["fire_count"] == 1

    updated = await phone.post(f"/api/v1/reminders/{reminder['id']}",
                               {"repeat": "weekly", "local_time": "09:00", "days": [6]})
    assert (await updated.json())["reminder"]["repeat"] == "weekly"
    cancelled = await (await phone.post(f"/api/v1/reminders/{reminder['id']}/cancel")).json()
    assert cancelled["reminder"]["status"] == "cancelled"
    speech = await phone.get(f"/api/v1/reminders/{reminder['id']}/speech")
    assert speech.status == 200
    assert phone.tts.texts[-1] == phone.companion.spoken(phone.companion.reminders.get(reminder["id"]))
    assert "meeting" in phone.tts.texts[-1].lower() and "You asked me" not in phone.tts.texts[-1]


def _phone_call_body(due: float, request_id: str = "Q2FsbE1lQXQ4UE0tMTIz") -> dict:
    # Exactly what ReminderRepository.create queues on the phone for "Garvis calls me", once.
    return {"text": "Call me about the meeting", "kind": "call", "tz": "Asia/Jerusalem", "repeat": "none",
            "local_id": "local-" + request_id, "due_at": due, "request_id": request_id}


async def test_phone_call_reminder_syncs_when_the_pc_lacks_tzdata(tmp_path, monkeypatch, caplog):
    """The 'Waiting to sync with your PC' regression, end to end on the PC side: the live Garvis ran on
    a Python without tzdata, so this exact request was refused with 400 'Unknown time zone'."""
    from zoneinfo import ZoneInfoNotFoundError

    import app.reminders.schedule as schedule

    def missing(name):
        raise ZoneInfoNotFoundError(name)

    monkeypatch.setattr(schedule, "ZoneInfo", missing)
    h = await _start(tmp_path)
    await h.pair()
    try:
        due = time.time() + 3600
        response = await h.post("/api/v1/reminders", _phone_call_body(due))
        assert response.status == 201, await response.text()
        reminder = (await response.json())["reminder"]
        assert reminder["kind"] == "call" and reminder["status"] == "scheduled"
        assert abs(reminder["due_at"] - due) < 1e-3
        # A retry after a dropped connection (same request id) never makes a second call.
        again = await h.post("/api/v1/reminders", _phone_call_body(due))
        assert (await again.json())["reminder"]["id"] == reminder["id"]
        assert len(h.companion.reminders.list()) == 1
        # Persisted: a restarted PC still has it and still de-duplicates the retry.
        restarted = ReminderStore(tmp_path / "reminders.json")
        assert restarted.get(reminder["id"]).kind == "call"
        assert restarted.create("Call me about the meeting", tz="Asia/Jerusalem", due_at=due, kind="call",
                                request_id="Q2FsbE1lQXQ4UE0tMTIz").id == reminder["id"]
        # A repeating one can't be scheduled without zones: refused with the fix, and logged on the PC.
        body = {**_phone_call_body(due, "cmVwZWF0aW5nLWNhbGwtMQ"), "repeat": "daily", "local_time": "09:00",
                "days": []}
        body.pop("due_at")
        refused = await h.post("/api/v1/reminders", body)
        assert refused.status == 400
        assert "tzdata" in (await refused.json())["error"]
        assert any(r.getMessage() == "mobile.native.reminder_refused" for r in caplog.records)
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_phone_call_reminder_round_trip_with_time_zones(phone):
    """Created with the phone's zone, listed for the phone, its time kept to the second, the call's
    greeting synthesized (what the phone plays when Garvis 'calls'), and the ring reported once."""
    due = time.time() + 2 * 3600
    response = await phone.post("/api/v1/reminders", _phone_call_body(due))
    assert response.status == 201
    reminder = (await response.json())["reminder"]
    assert reminder["tz"] == "Asia/Jerusalem" and reminder["due_local"].endswith("(Asia/Jerusalem)")
    listed = await (await phone.get("/api/v1/reminders")).json()
    assert [(r["id"], r["kind"], r["due_at"]) for r in listed["reminders"]] == [(reminder["id"], "call", reminder["due_at"])]
    speech = await phone.get(f"/api/v1/reminders/{reminder['id']}/speech")
    assert speech.status == 200 and (await speech.read())[:4] == b"RIFF"
    # What the PC's voice says is the reminder's own sentence ("Call me about" is the request, not the content).
    assert phone.tts.texts[-1] == phone.companion.spoken(phone.companion.reminders.get(reminder["id"]))
    assert "the meeting" in phone.tts.texts[-1] and "call you" not in phone.tts.texts[-1]
    for _ in range(2):
        fired = await (await phone.post(f"/api/v1/reminders/{reminder['id']}/fired",
                                        {"occurrence": reminder["due_at"]})).json()
    assert fired["reminder"]["status"] == "completed" and fired["reminder"]["fire_count"] == 1


async def test_reminder_create_auth_and_validation_failures(phone):
    due = time.time() + 600
    unauthenticated = await phone.client.post("/api/v1/reminders", json=_phone_call_body(due),
                                              headers={"Authorization": "Bearer wrong"})
    assert unauthenticated.status == 401 and (await unauthenticated.json())["code"] == "unpaired"
    late = await phone.post("/api/v1/reminders", _phone_call_body(time.time() - 7200, "bGF0ZS1jYWxsLTAwMDE"))
    assert late.status == 400 and (await late.json())["code"] == "invalid_reminder"
    bad_kind = await phone.post("/api/v1/reminders", {**_phone_call_body(due, "YmFkLWtpbmQtMDAwMQ"), "kind": "sms"})
    assert bad_kind.status == 400
    assert phone.companion.reminders.list() == []


async def test_reminder_validation(phone):
    assert (await phone.post("/api/v1/reminders", {"text": "x", "due_at": time.time() - 7200})).status == 400
    assert (await phone.post("/api/v1/reminders", {"text": "x", "due_at": "soon"})).status == 400
    assert (await phone.post("/api/v1/reminders", {"text": "x", "repeat": "hourly",
                                                   "local_time": "09:00"})).status == 400
    assert (await phone.post("/api/v1/reminders/nope/complete")).status == 404


# ------------------------------------------------------------------- images
async def test_images_unavailable_without_a_provider(phone):
    response = await phone.post("/api/v1/images", {"prompt": "a futuristic city"})
    assert response.status == 503 and (await response.json())["code"] == "unavailable"
    listed = await (await phone.get("/api/v1/images")).json()
    assert listed["available"] is False and listed["images"] == []


async def test_image_generation_round_trip(tmp_path):
    h = await _start(tmp_path, image_provider=FakeProvider())
    await h.pair()
    try:
        response = await h.post("/api/v1/images", {"prompt": "orange glowing sphere", "request_id": "image-000001"})
        assert response.status == 202
        image = (await response.json())["image"]
        await wait_until(lambda: h.companion.images.get(image["id"]).status == "completed")
        detail = await (await h.get(f"/api/v1/images/{image['id']}")).json()
        assert detail["image"]["has_image"] is True
        file = await h.get(f"/api/v1/images/{image['id']}/file")
        assert file.status == 200 and file.headers["Content-Type"] == "image/jpeg"
        snapshot = await _events(h)
        assert any(e["kind"] == "image" and e["image_id"] == image["id"] for e in snapshot["events"])
        assert (await h.post(f"/api/v1/images/{image['id']}/delete")).status == 200
        assert (await h.get(f"/api/v1/images/{image['id']}/file")).status == 404
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


# ------------------------------------------------------------ phone actions
async def test_phone_action_round_trip_through_the_event_stream(phone):
    response = await phone.post("/api/v1/phone/capabilities",
                                {"capabilities": ["apps", "root_access"], "paused": False, "accessibility": False})
    assert (await response.json())["capabilities"] == ["apps", "images"]  # unknown names are dropped
    request = asyncio.ensure_future(phone.companion.phone.request("open_app", {"app": "Spotify"}, timeout=5))
    await wait_until(lambda: phone.companion.phone.pending_for(phone.pairing.devices()[0].id))
    snapshot = await _events(phone)
    (action,) = snapshot["phone_actions"]
    assert action["action"] == "open_app" and action["confirm"] is False
    response = await phone.post(f"/api/v1/phone/actions/{action['id']}/result",
                                {"status": "done", "message": "Opened Spotify"})
    assert response.status == 200
    assert (await request)["status"] == "done"
    assert (await phone.post(f"/api/v1/phone/actions/{action['id']}/result", {"status": "done"})).status == 404


# ------------------------------------------------------ failed sign-in limit
def test_auth_failure_limiter_counts_per_address_and_ages_out():
    now = [1000.0]
    limiter = AuthFailureLimiter(limit=3, window=60, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.retry_after("100.64.0.9") == 0
        limiter.failed("100.64.0.9")
    assert 59 <= limiter.retry_after("100.64.0.9") <= 60
    assert limiter.retry_after("192.168.10.200") == 0  # other addresses are unaffected
    now[0] += 61
    assert limiter.retry_after("100.64.0.9") == 0  # the failures aged out
    limiter.failed("100.64.0.9")
    limiter.succeeded("100.64.0.9")
    assert limiter._failures == {}
    for index in range(AuthFailureLimiter.MAX_ADDRESSES + 5):
        limiter.failed(f"10.0.{index // 250}.{index % 250}")
    assert len(limiter._failures) == AuthFailureLimiter.MAX_ADDRESSES  # bounded memory


async def test_repeated_bad_tokens_are_refused_with_429_and_logged_without_the_token(phone, caplog):
    guess = "guessed-token-" + "x" * 30
    for _ in range(AUTH_FAILURE_LIMIT):
        response = await phone.client.get("/api/v1/status", headers={"Authorization": f"Bearer {guess}"})
        assert response.status == 401
    limited = await phone.client.get("/api/v1/status", headers={"Authorization": f"Bearer {guess}"})
    assert limited.status == 429 and (await limited.json())["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) > 0
    # While the address is cooling down even a valid token from it waits (it can't be told apart).
    assert (await phone.get("/api/v1/status")).status == 429
    assert (await phone.client.get("/api/v1/hello")).status == 200  # the public probe is never limited
    failures = [r for r in caplog.records if r.getMessage() == "mobile.native.auth_failed"]
    assert len(failures) == AUTH_FAILURE_LIMIT
    assert failures[0].extra_fields["reason"] == "unknown_token" and failures[0].extra_fields["via"] == "lan"
    assert guess not in caplog.text and phone.token not in caplog.text


async def test_wrong_pairing_codes_count_towards_the_limit(tmp_path):
    h = await _start(tmp_path)
    try:
        for _ in range(AUTH_FAILURE_LIMIT // 5):  # every 5 wrong codes burn the code: the PC shows a new one
            code = h.pairing.new_pairing_code()
            wrong = "0" * 8 if code.code != "0" * 8 else "1" * 8
            for _ in range(5):
                assert (await h.client.post("/api/v1/pair", json={"code": wrong})).status == 403
        code = h.pairing.new_pairing_code()
        assert (await h.client.post("/api/v1/pair", json={"code": code.code})).status == 429
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_a_successful_sign_in_clears_earlier_failures(phone):
    for _ in range(AUTH_FAILURE_LIMIT - 1):
        await phone.client.get("/api/v1/status", headers={"Authorization": "Bearer stale"})
    assert (await phone.get("/api/v1/status")).status == 200
    assert (await phone.client.get("/api/v1/status", headers={"Authorization": "Bearer stale"})).status == 401


# ---------------------------------------------------------------- remote guard
def test_tailscale_address_checks():
    assert is_tailscale_address("100.101.102.103") and is_tailscale_address("fd7a:115c:a1e0::1")
    assert not is_tailscale_address("192.168.10.156") and not is_tailscale_address("8.8.8.8")
    # A CGNAT source alone isn't enough: it must arrive on this PC's own tailnet address.
    assert not is_tailscale_connection("100.101.102.103", "192.168.10.156")
    assert is_tailscale_connection("100.101.102.103", "100.64.0.7")


async def test_remote_access_off_refuses_non_lan_clients(tmp_path):
    h = await _start(tmp_path, networks="10.0.0.0/8")  # 127.0.0.1 is outside the allowed networks
    try:
        assert (await h.client.get("/api/v1/hello")).status == 403
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_remote_tailnet_client_reaches_only_the_native_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "is_tailscale_connection", lambda remote, local: True)
    h = await _start(tmp_path, networks="10.0.0.0/8", remote_access=True)
    try:
        hello = await h.client.get("/api/v1/hello")
        assert hello.status == 200 and (await hello.json())["remote"] is True
        assert (await h.client.get("/")).status == 403  # the web app isn't offered remotely
        assert (await h.client.get("/api/events")).status == 403
        assert (await h.client.get("/api/v1/hello", headers={"Host": "evil.example"})).status == 421
        await h.pair()
        assert (await h.get("/api/v1/status")).status == 200  # still needs (and accepts) the device token
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_remote_access_requires_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "is_tailscale_connection", lambda remote, local: True)
    h = await _start(tmp_path, networks="10.0.0.0/8", remote_access=False)
    try:
        assert (await h.client.get("/api/v1/hello")).status == 403
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


# ------------------------------------------------------------------ bridge
def test_bridge_registers_the_companion_tools_and_serves_api_v1_over_tls(tmp_path):
    import json
    import ssl
    import urllib.request

    from app.ui.agent_bridge import AgentBridge
    from tests.test_mobile_integration import _bridge_settings, _wait

    bridge = AgentBridge(settings=_bridge_settings(
        tmp_path, mobile_enabled=True, mobile_host="127.0.0.1", mobile_port=0, mobile_data_dir=tmp_path / "mobile"))
    try:
        _wait(lambda: bridge.mobile.is_running)
        names = {entry["name"] for entry in bridge.tool_catalog()}
        assert {"create_reminder", "list_reminders", "snooze_reminder", "cancel_reminder", "generate_image",
                "phone_action"} <= names
        # The TaskManager may run them: they are in the tool set it validates against.
        assert "phone_action" in bridge._tool_names
        context = ssl.create_default_context(cafile=str(tmp_path / "mobile" / "ca.crt"))
        url = f"https://127.0.0.1:{bridge.mobile.bound_port}/api/v1/hello"
        with urllib.request.urlopen(url, context=context, timeout=5) as response:
            assert json.loads(response.read())["api"] == 1
        assert bridge.mobile.companion.images.available is False  # nothing configured -> nothing pretends
    finally:
        bridge.shutdown()


def test_bridge_without_mobile_adds_no_phone_tools(tmp_path):
    from app.ui.agent_bridge import AgentBridge
    from tests.test_mobile_integration import _bridge_settings

    bridge = AgentBridge(settings=_bridge_settings(tmp_path))
    try:
        names = {entry["name"] for entry in bridge.tool_catalog()}
        assert "phone_action" not in names and "create_reminder" not in names
    finally:
        bridge.shutdown()


def test_invalid_remote_access_setting_is_refused():
    from app.core.config import Settings
    from app.core.exceptions import ConfigurationError
    from app.mobile.build import build_companion

    with pytest.raises(ConfigurationError):
        build_companion(Settings(_env_file=None, mobile_remote_access="public"))


# ------------------------------------------------- away from home (Tailscale)
async def test_status_tells_the_phone_where_to_reach_the_pc_away_from_home(tmp_path, monkeypatch):
    calls = []

    def tailnet():
        calls.append(1)
        return ["100.101.102.103", "fd7a:115c:a1e0::1"]

    monkeypatch.setattr(server_module, "detect_tailscale_addresses", tailnet)
    h = await _start(tmp_path, remote_access=True)
    try:
        await h.pair()
        status = await (await h.get("/api/v1/status")).json()
        # IPv4 only (what the app's address field takes), on the port the phone reached.
        assert status["remote_urls"] == [f"https://100.101.102.103:{h.client.port}"]
        await h.get("/api/v1/status")
        assert len(calls) == 1  # re-read at most once a minute, not per request
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


async def test_no_away_from_home_address_when_remote_access_is_off(phone, monkeypatch):
    monkeypatch.setattr(server_module, "detect_tailscale_addresses", lambda: ["100.101.102.103"])
    assert (await (await phone.get("/api/v1/status")).json())["remote_urls"] == []


def test_tailscale_addresses_fall_back_to_the_cli(monkeypatch, tmp_path):
    from app.mobile import network

    real_cli = network._tailscale_cli_addresses
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("192.168.10.156", 0))])
    monkeypatch.setattr(network, "_tailscale_cli_addresses", lambda: ["100.101.102.103"])
    assert network.detect_tailscale_addresses() == ["100.101.102.103"]
    monkeypatch.setattr(network, "_tailscale_cli_addresses", lambda: [])
    assert network.detect_tailscale_addresses() == []
    assert real_cli(tmp_path / "no-tailscale.exe") == []  # not installed: nothing, no error


async def test_a_hebrew_call_reminder_is_handed_to_the_phones_voice(phone):
    """The PC's Piper voice is English: a Hebrew reminder's sentence comes back as text (415, code
    voice_language) for the phone's own Hebrew voice; an English one is spoken by the PC as before."""
    due = time.time() + 3600
    hebrew = await (await phone.post("/api/v1/reminders", {**_phone_call_body(due), "text": "להתקשר לאמא"})).json()
    assert hebrew["reminder"]["spoken"] == "" or "להתקשר לאמא" in hebrew["reminder"]["spoken"]
    response = await phone.get(f"/api/v1/reminders/{hebrew['reminder']['id']}/speech")
    body = await response.json()
    assert response.status == 415 and body["code"] == "voice_language" and body["lang"] == "he"
    assert body["spoken"].startswith("היי") and "להתקשר לאמא שלך" in body["spoken"]
    english = await (await phone.post("/api/v1/reminders", {**_phone_call_body(due, "RW5nbGlzaENhbGwtMDAx"),
                                                            "text": "call mom"})).json()
    speech = await phone.get(f"/api/v1/reminders/{english['reminder']['id']}/speech")
    assert speech.status == 200 and (await speech.read())[:4] == b"RIFF"
    assert "call your mom" in phone.tts.texts[-1]
