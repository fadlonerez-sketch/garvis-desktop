"""Garvis Eye Station (the tablet face) on /api/v1: it pairs as a *station*,
its token opens only the station routes, it is never mistaken for a phone,
and it can read only the voice turns it started itself.

Same in-process harness as test_mobile_native_api.py: real TaskManager,
MobileService, PairingManager and PhoneActionBroker; scripted LLM, fake STT/TTS."""
from __future__ import annotations

import json

import pytest

from app.mobile.pairing import PairingManager
from tests.mobile_fakes import make_wav, wait_until
from tests.test_mobile_native_api import _App, _start


async def _pair_station(h: _App) -> dict:
    code = h.pairing.new_pairing_code()
    response = await h.client.post("/api/v1/pair", json={"code": code.code, "client": "eye_station",
                                                         "device_name": "Example Tablet"})
    assert response.status == 201
    data = await response.json()
    h.token = data["token"]
    return data


@pytest.fixture
async def station(tmp_path):
    h = await _start(tmp_path)
    await _pair_station(h)
    yield h
    await h.service.stop()
    await h.tasks.stop()
    await h.client.close()


async def test_pairing_as_eye_station_creates_a_station_device(station, tmp_path):
    [record] = station.pairing.devices()
    assert record.kind == "station" and record.label == "Eye Station"
    stored = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    assert stored["devices"][0]["kind"] == "station"
    assert PairingManager(tmp_path / "devices.json").devices()[0].kind == "station"  # survives a restart
    status = await (await station.get("/api/v1/status")).json()
    assert status["kind"] == "station"
    hello = await (await station.get("/api/v1/hello")).json()
    assert hello["paired"] is True


async def test_a_plain_pairing_is_still_a_phone(tmp_path):
    h = await _start(tmp_path)
    try:
        data = await h.pair()
        assert data["kind"] == "phone"
        assert h.pairing.devices()[0].kind == "phone"
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()


def test_devices_saved_before_kinds_existed_load_as_phones(tmp_path):
    (tmp_path / "devices.json").write_text(json.dumps({"devices": [
        {"id": "ab12cd34", "label": "Android", "created_at": 1.0, "expires_at": 4e9, "token_sha256": "0" * 64}]}),
        encoding="utf-8")
    manager = PairingManager(tmp_path / "devices.json")
    assert manager.devices()[0].kind == "phone"
    assert manager.device_count(kind="phone") == 1 and manager.device_count(kind="station") == 0


def test_unknown_device_kind_is_refused(tmp_path):
    manager = PairingManager(tmp_path / "devices.json")
    code = manager.new_pairing_code()
    with pytest.raises(ValueError):
        manager.redeem(code.code, kind="robot-overlord")


STATION_DENIED = [("GET", "/api/v1/events"), ("POST", "/api/v1/messages"), ("GET", "/api/v1/tasks"),
                  ("GET", "/api/v1/reminders"), ("POST", "/api/v1/reminders"), ("GET", "/api/v1/images"),
                  ("POST", "/api/v1/images"), ("POST", "/api/v1/phone/capabilities"),
                  ("POST", "/api/v1/phone/actions/0123456789abcdef/result"), ("GET", "/api/v1/protocols"),
                  ("POST", "/api/v1/confirmations/00000000-0000-0000-0000-000000000000")]


@pytest.mark.parametrize("method,path", STATION_DENIED)
async def test_station_token_is_refused_outside_station_routes(station, method, path):
    if method == "GET":
        response = await station.get(path)
    else:
        response = await station.post(path, {"text": "x", "capabilities": [], "status": "done", "approved": True})
    assert response.status == 403
    assert (await response.json())["code"] == "station_scope"


async def test_station_is_never_registered_as_a_phone(station):
    for path in ("/api/v1/status", "/api/v1/presence", "/api/v1/hello"):
        assert (await station.get(path)).status == 200
    assert station.companion.phone.connected_devices() == []
    assert station.companion.phone.describe() == {"connected": False}
    # agent_bridge's phone_ready (reminders ring on the phone) counts phones only
    assert station.pairing.device_count(kind="phone") == 0


async def test_presence_long_poll_works_for_a_station(station):
    current = await (await station.get("/api/v1/presence")).json()
    assert current["state"] == "idle" and "speaking" in current["states"]
    station.companion.presence.set_mode("thinking")
    changed = await (await station.get(f"/api/v1/presence?after={current['version']}&wait=2")).json()
    assert changed["state"] == "thinking" and changed["version"] > current["version"]


async def test_token_refresh_keeps_the_station_kind(station):
    response = await station.post("/api/v1/token/refresh")
    assert response.status == 200
    station.token = (await response.json())["token"]
    assert station.pairing.devices()[0].kind == "station"
    assert (await station.get("/api/v1/reminders")).status == 403


async def test_station_voice_turn_round_trip_and_only_its_own_turns(station):
    headers = station.auth(**{"Content-Type": "audio/wav", "X-Request-Id": "station-voice-01"})
    response = await station.client.post("/api/v1/voice", data=make_wav(1.0), headers=headers)
    assert response.status == 202
    turn = await response.json()
    assert turn["transcript"] == "open my project and run its tests"
    task = station.tasks.get(turn["turn_id"])
    assert task.device == "Eye Station"
    await wait_until(lambda: station.tasks.get(turn["turn_id"]).terminal)
    detail = await station.get(f"/api/v1/tasks/{turn['turn_id']}")
    assert detail.status == 200
    speech = await station.get(f"/api/v1/tasks/{turn['turn_id']}/speech")
    assert speech.status == 200 and (await speech.read())[:4] == b"RIFF"

    # A turn from someone else (the desktop, the phone) is invisible to the station.
    other = station.tasks.submit_instruction("what's on my calendar", origin="desktop")
    await wait_until(lambda: station.tasks.get(other.id).terminal)
    for path in (f"/api/v1/tasks/{other.id}", f"/api/v1/tasks/{other.id}/speech"):
        assert (await station.get(path)).status == 404
    assert (await station.post(f"/api/v1/tasks/{other.id}/cancel")).status == 404


async def test_unpairing_a_station_works_and_ends_its_access(station):
    assert (await station.post("/api/v1/unpair")).status == 200
    assert station.pairing.device_count() == 0
    assert (await station.get("/api/v1/presence")).status == 401
