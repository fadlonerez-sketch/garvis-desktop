"""/api/v1/voice/* -- voice ownership over the real HTTPS app (same in-process harness as
test_mobile_native_api.py): a paired Eye Station and a paired phone check in, claim and release
the voice, and always agree on the one endpoint allowed to listen."""
from __future__ import annotations

import asyncio

import pytest

from app.core.voice_ownership import DESKTOP_ENDPOINT, VoiceState
from tests.test_eye_station_api import _pair_station
from tests.test_mobile_native_api import _App, _start


class _Client:
    """One paired device, speaking the voice protocol the way the apps do."""

    def __init__(self, h: _App, token: str) -> None:
        self.h = h
        self.token = token
        self.epoch = ""
        self.version = 0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    async def call(self, path: str, body: dict | None = None):
        return await self.h.client.post(path, json=body if body is not None else {}, headers=self._headers())

    async def beat(self, *, want: str | None = None, mic: bool | None = None, wait: float = 0) -> dict:
        body = {"epoch": self.epoch, "version": self.version, "wait": wait}
        if want is not None:
            body["want"] = want
        if mic is not None:
            body["mic"] = mic
        response = await self.call("/api/v1/voice/heartbeat", body)
        assert response.status == 200, await response.text()
        return self.seen(await response.json())

    def seen(self, view: dict) -> dict:
        if view["epoch"] != self.epoch or view["version"] >= self.version:
            self.epoch, self.version = view["epoch"], view["version"]
        return view

    async def ack(self) -> dict:
        """Microphone closed, having seen the latest version."""
        await self.beat()
        return await self.beat(mic=False)


@pytest.fixture
async def devices(tmp_path):
    h = await _start(tmp_path)
    await _pair_station(h)
    station = _Client(h, h.token)
    await h.pair()
    phone = _Client(h, h.token)
    ownership = h.companion.voice_ownership
    ownership.register_local(DESKTOP_ENDPOINT, kind="desktop", label="Garvis Desktop")
    yield h, station, phone, ownership
    await h.service.stop()
    await h.tasks.stop()
    await h.client.close()


def _state(ownership, device_id: str) -> VoiceState:
    return ownership.state_of(device_id)


async def test_first_station_gets_standby_and_the_desktop_sleeps(devices):
    h, station, phone, ownership = devices
    view = await station.beat(want="standby", mic=False)
    assert view["state"] == "standby" and view["listen"] is True
    assert 0 < view["lease_seconds"] <= 20
    ownership.set_wants_standby(DESKTOP_ENDPOINT, True)  # ticked later: queued, not listening
    assert ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    view = await phone.beat()
    assert view["state"] == "sleep" and view["listen"] is False
    assert view["owner"] == {"kind": "station", "label": "Eye Station", "state": "standby"}


async def test_phone_reminder_priority_round_trip_restores_the_station(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    response = await phone.call("/api/v1/voice/claim", {"mode": "priority", "reason": "reminder", "ref": "abc123",
                                                         "max_seconds": 60})
    assert response.status == 200
    data = await response.json()
    claim = data["claim"]
    assert claim["mode"] == "priority" and claim["ref"] == "abc123"
    assert data["voice"]["state"] == "sleep" and data["voice"]["reason"] == "waiting_for_release"
    phone.seen(data["voice"])

    told = await station.beat(mic=True)
    assert told["state"] == "sleep" and told["listen"] is False
    await station.beat(mic=False)  # Ears stopped: the microphone is released
    granted = await phone.beat()
    assert granted["state"] == "priority" and granted["listen"] is True

    response = await phone.call("/api/v1/voice/release", {"claim_id": claim["id"]})
    assert (await response.json())["released"] is True
    await phone.ack()
    view = await station.beat()
    assert view["state"] == "standby"


async def test_long_poll_answers_as_soon_as_ownership_changes(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    poll = asyncio.create_task(station.beat(mic=True, wait=5))
    await asyncio.sleep(0.2)
    assert not poll.done()
    started = asyncio.get_running_loop().time()
    await phone.call("/api/v1/voice/claim", {"mode": "priority", "reason": "call"})
    view = await asyncio.wait_for(poll, 2)
    assert view["state"] == "sleep"
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_long_poll_reports_the_lease_as_renewed_on_arrival(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    view = await station.beat(mic=True, wait=1.2)  # nothing changes: answers at the wait
    assert view["lease_seconds"] == pytest.approx(20.0, abs=0.5)  # not 20 - 1.2


async def test_station_cannot_take_priority(devices):
    h, station, phone, ownership = devices
    response = await station.call("/api/v1/voice/claim", {"mode": "priority", "reason": "reminder"})
    assert response.status == 403
    assert (await response.json())["code"] == "station_scope"


async def test_wake_conversation_from_a_sleeping_station_is_refused(devices):
    h, station, phone, ownership = devices
    ownership.set_wants_standby(DESKTOP_ENDPOINT, True)
    await station.beat(want="standby", mic=False)
    response = await station.call("/api/v1/voice/claim", {"mode": "active", "reason": "wake"})
    assert response.status == 409
    data = await response.json()
    assert data["code"] == "not_owner" and data["voice"]["state"] == "sleep"


async def test_station_conversation_keeps_everything_else_asleep_and_returns_to_standby(devices):
    h, station, phone, ownership = devices
    ownership.set_wants_standby(DESKTOP_ENDPOINT, True)
    ownership.report_mic(DESKTOP_ENDPOINT, False, ownership.version)
    ownership.set_wants_standby(DESKTOP_ENDPOINT, False)
    ownership.report_mic(DESKTOP_ENDPOINT, False, ownership.version)
    await station.beat(want="standby", mic=False)
    response = await station.call("/api/v1/voice/claim", {"mode": "active", "reason": "wake"})
    data = await response.json()
    assert data["voice"]["state"] == "active"
    assert (await phone.beat())["state"] == "sleep"
    assert ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    await station.call("/api/v1/voice/release", {"claim_id": data["claim"]["id"]})
    assert (await station.beat())["state"] == "standby"


async def test_explicit_phone_conversation_takes_over_and_gives_back(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    response = await phone.call("/api/v1/voice/claim", {"mode": "active", "reason": "conversation",
                                                         "explicit": True})
    claim = (await response.json())["claim"]
    await station.ack()
    assert (await phone.beat())["state"] == "active"
    await phone.call("/api/v1/voice/release", {"claim_id": claim["id"]})
    await phone.ack()
    assert (await station.beat())["state"] == "standby"


async def test_leaving_hands_standby_to_the_waiting_desktop(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    ownership.set_wants_standby(DESKTOP_ENDPOINT, True)
    response = await station.call("/api/v1/voice/leave")
    assert (await response.json())["voice"]["state"] == "disconnected"
    assert ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY


async def test_unpairing_gives_up_ownership(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    ownership.set_wants_standby(DESKTOP_ENDPOINT, True)
    await station.call("/api/v1/unpair")
    assert ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY


async def test_every_client_sees_the_same_owner_and_version(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    await phone.call("/api/v1/voice/claim", {"mode": "priority", "reason": "call"})
    await station.ack()
    views = [await station.beat(), await phone.beat(), ownership.view(DESKTOP_ENDPOINT)]
    assert len({v["version"] for v in views}) == 1
    assert all(v["owner"] == {"kind": "phone", "label": "Android", "state": "priority"} for v in views)
    assert [v["state"] for v in views] == ["sleep", "priority", "sleep"]


async def test_simultaneous_claims_over_http_produce_one_owner(devices):
    h, station, phone, ownership = devices
    await station.beat(want="standby", mic=False)
    results = await asyncio.gather(
        phone.call("/api/v1/voice/claim", {"mode": "active", "reason": "conversation", "explicit": True}),
        station.call("/api/v1/voice/claim", {"mode": "active", "reason": "wake"}),
        phone.call("/api/v1/voice/claim", {"mode": "priority", "reason": "call"}),
    )
    assert all(r.status in (200, 409) for r in results)
    listening = [e for e in ownership.listening_endpoints()]
    granted = [s for s in (await station.beat(), await phone.beat()) if s["listen"]]
    assert len(granted) <= 1
    assert len(listening) <= 1


@pytest.mark.parametrize("body", [{"want": "always"}, {"mic": "yes"}, {"version": "x"}, {"wait": "soon"}])
async def test_bad_heartbeats_are_rejected(devices, body):
    h, station, phone, ownership = devices
    response = await phone.call("/api/v1/voice/heartbeat", body)
    assert response.status == 400


@pytest.mark.parametrize("body", [{"mode": "forever"}, {"mode": "active", "reason": "DROP TABLE"},
                                  {"mode": "priority", "max_seconds": -1}, {"mode": "active", "explicit": "yes"}])
async def test_bad_claims_are_rejected(devices, body):
    h, station, phone, ownership = devices
    response = await phone.call("/api/v1/voice/claim", body)
    assert response.status == 400


async def test_voice_routes_need_a_paired_device(devices):
    h, station, phone, ownership = devices
    response = await h.client.post("/api/v1/voice/heartbeat", json={})
    assert response.status == 401
