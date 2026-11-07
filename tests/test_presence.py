"""Garvis's display state (app/core/presence.py) -- the feed a future face
display (Garvis Face) follows -- and its API. No face UI exists yet; these
pin the contract: six states, versioned changes, long-poll, attention
notifications, and that it follows the existing view-model."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from app.core.presence import PresenceHub, PresenceState
from app.ui.state import GarvisState
from tests.test_mobile_native_api import _phone_call_body, _start
from tests.test_ui_dashboard import _FakeBridge


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_six_states():
    assert [s.value for s in PresenceState] == ["idle", "listening", "thinking", "speaking", "notification", "error"]


def test_modes_map_to_states_and_each_change_is_versioned():
    hub = PresenceHub()
    assert hub.snapshot().state == PresenceState.IDLE and hub.snapshot().version == 0
    for mode in ("listening", "thinking", "speaking", "error", "idle"):
        hub.set_mode(mode)
        assert hub.snapshot().state.value == mode
    assert hub.snapshot().version == 5
    hub.set_mode("idle")  # no change: no new version
    hub.set_mode("standby-ish nonsense")
    assert hub.snapshot().version == 5


def test_a_notification_shows_while_idle_and_expires():
    clock = Clock()
    hub = PresenceHub(clock=clock, notification_seconds=8)
    hub.notify("reminder")
    now = hub.snapshot()
    assert (now.state, now.reason) == (PresenceState.NOTIFICATION, "reminder")
    hub.set_mode("speaking")  # activity wins over the notification
    assert hub.snapshot().state == PresenceState.SPEAKING
    hub.set_mode("idle")
    assert hub.snapshot().state == PresenceState.NOTIFICATION  # still within its time
    clock.now += 9
    assert hub.snapshot().state == PresenceState.IDLE and hub.snapshot().reason == ""


async def test_long_poll_wakes_on_a_change_from_another_thread():
    hub = PresenceHub()
    hub.bind(asyncio.get_running_loop())
    version = hub.snapshot().version
    threading.Timer(0.1, hub.set_mode, args=("thinking",)).start()  # the UI thread, in Garvis
    started = time.monotonic()
    current = await hub.wait(version, timeout=5)
    assert current.state == PresenceState.THINKING and current.version == version + 1
    assert time.monotonic() - started < 2
    assert (await hub.wait(current.version, timeout=0.05)).version == current.version  # nothing new: times out


async def test_an_expiring_notification_is_announced():
    hub = PresenceHub(notification_seconds=0.1)
    hub.bind(asyncio.get_running_loop())
    hub.notify("confirmation")
    shown = hub.snapshot()
    assert shown.state == PresenceState.NOTIFICATION
    back = await hub.wait(shown.version, timeout=3)
    assert back.state == PresenceState.IDLE


def test_a_broken_display_listener_never_breaks_garvis():
    hub = PresenceHub()
    hub.add_listener(lambda p: 1 / 0)
    seen = []
    hub.add_listener(seen.append)
    hub.set_mode("listening")
    assert [p.state for p in seen] == [PresenceState.LISTENING]


def test_it_follows_the_dashboard_view_model():
    QApplication.instance() or QApplication([])
    bridge = _FakeBridge()
    bridge.presence = PresenceHub()
    state = GarvisState()
    state.connect_bridge(bridge)
    state.on_session_state("listening")
    assert bridge.presence.snapshot().state == PresenceState.LISTENING
    state.on_playing(True)
    assert bridge.presence.snapshot().state == PresenceState.SPEAKING
    state.on_playing(False)
    state.on_session_state("idle")
    assert bridge.presence.snapshot().state == PresenceState.IDLE


# ----------------------------------------------------------------------- API
@pytest.fixture
async def phone(tmp_path):
    h = await _start(tmp_path)
    h.companion.presence.bind(asyncio.get_running_loop())
    yield h
    await h.service.stop()
    await h.tasks.stop()
    await h.client.close()


async def test_the_presence_api_needs_a_paired_device(phone):
    assert (await phone.client.get("/api/v1/presence")).status == 401


async def test_the_presence_api_long_polls(phone):
    await phone.pair()
    first = await (await phone.get("/api/v1/presence")).json()
    assert first["state"] == "idle" and "notification" in first["states"]
    asyncio.get_running_loop().call_later(0.1, phone.companion.presence.set_mode, "thinking")
    changed = await (await phone.get(f"/api/v1/presence?after={first['version']}&wait=5")).json()
    assert changed["state"] == "thinking" and changed["version"] == first["version"] + 1
    assert (await phone.get("/api/v1/presence?after=x")).status == 400


async def test_a_reminder_that_rang_is_a_notification(phone):
    await phone.pair()
    due = time.time() + 3600
    reminder = (await (await phone.post("/api/v1/reminders", _phone_call_body(due))).json())["reminder"]
    for _ in range(2):  # a repeated report is not a second notification
        await phone.post(f"/api/v1/reminders/{reminder['id']}/fired", {"occurrence": reminder["due_at"]})
    now = await (await phone.get("/api/v1/presence")).json()
    assert (now["state"], now["reason"]) == ("notification", "reminder")
    assert now["version"] == 1
