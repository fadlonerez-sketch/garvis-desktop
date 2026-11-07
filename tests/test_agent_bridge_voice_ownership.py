"""The Desktop's microphone obeys voice ownership (app/core/voice_ownership.py): a real
AgentBridge (real Agent, real loop, real StandbyListener/ConversationSession) on the fake
recorder of test_agent_bridge_standby.py -- the recorder's open/close log is the proof of what
the microphone actually did. Other devices are driven through the same manager calls the
/api/v1/voice routes make."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt

from app.core.voice_ownership import DESKTOP_ENDPOINT, VoiceState
from tests.test_agent_bridge_standby import _bridge, _collect, _settle_session_end, _wait_until

TABLET = "tab12345"
PHONE = "pho12345"


def _tablet(bridge, **kwargs) -> dict:
    return bridge.voice_ownership.heartbeat(TABLET, kind="station", label="Eye Station", **kwargs)


def _tablet_releases(bridge) -> None:
    view = bridge.voice_ownership.view(TABLET)
    _tablet(bridge, mic_open=False, seen_version=view["version"], seen_epoch=view["epoch"])


def _phone_claim(bridge, mode="priority", reason="reminder", **kwargs):
    return bridge.voice_ownership.claim(PHONE, mode, reason, kind="phone", label="Example Phone", **kwargs)[0]


def _phone_releases(bridge, claim) -> None:
    manager = bridge.voice_ownership
    manager.release(PHONE, claim.id)
    manager.heartbeat(
        PHONE,
        kind="phone",
        label="Example Phone",
        mic_open=False,
        seen_version=manager.version,
        seen_epoch=manager.epoch,
    )


def test_desktop_does_not_listen_while_the_tablet_owns_standby(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    owners = _collect(bridge.voice_owner_changed)
    try:
        _tablet(bridge, want_standby=True, mic_open=False)  # Eye Station opened first
        bridge.set_standby_enabled(True)  # then the user ticks the Desktop's checkbox
        time.sleep(0.2)
        assert recorder.ops == []  # the Desktop microphone never opened
        assert not bridge.is_standby_active
        assert bridge.voice_ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
        assert _wait_until(lambda: owners and "Eye Station is listening" in owners[-1])
        assert "next in line" in owners[-1]
        bridge.voice_ownership.leave(TABLET)  # the tablet's app closes: the Desktop is next
        assert _wait_until(lambda: bridge.is_standby_active)
        assert recorder.ops == ["open"]
    finally:
        bridge.shutdown()


def test_phone_reminder_puts_the_desktop_to_sleep_and_standby_resumes_afterwards(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        claim = _phone_claim(bridge)
        # The Desktop closes its microphone and confirms; only then does the phone get the voice.
        assert _wait_until(lambda: bridge.voice_ownership.state_of(PHONE) == VoiceState.PRIORITY)
        assert not recorder.open and not bridge.is_standby_active
        assert recorder.ops == ["open", "close"]
        _phone_releases(bridge, claim)
        assert _wait_until(lambda: bridge.is_standby_active)
        assert recorder.ops == ["open", "close", "open"]
        assert bridge.voice_ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY
    finally:
        bridge.shutdown()


def test_explicit_desktop_session_waits_for_the_tablet_and_hands_back_afterwards(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        _tablet(bridge, want_standby=True, mic_open=True)
        bridge.start_session()  # the user clicks Start Session
        session = bridge._session
        time.sleep(0.2)
        assert recorder.ops == []  # the tablet hasn't released its microphone yet
        assert bridge.voice_ownership.state_of(TABLET) == VoiceState.SLEEP
        _tablet_releases(bridge)
        assert _wait_until(lambda: session.is_active)
        assert recorder.open
        assert bridge.voice_ownership.state_of(DESKTOP_ENDPOINT) == VoiceState.ACTIVE
        bridge.end_session()
        _settle_session_end(bridge)
        assert not recorder.open
        assert _wait_until(lambda: bridge.voice_ownership.state_of(TABLET) == VoiceState.STANDBY)
    finally:
        bridge.shutdown()


def test_priority_call_ends_a_desktop_session_and_closes_the_microphone(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    states = _collect(bridge.session_state_changed)
    try:
        bridge.start_session()
        session = bridge._session
        assert _wait_until(lambda: session.is_active)
        claim = _phone_claim(bridge, reason="call")
        assert _wait_until(lambda: bridge._session is None and not recorder.open)
        assert _wait_until(lambda: states and states[-1] == "idle")
        assert _wait_until(lambda: bridge.voice_ownership.state_of(PHONE) == VoiceState.PRIORITY)
        _phone_releases(bridge, claim)
        time.sleep(0.1)
        assert not recorder.open  # the session doesn't come back by itself
    finally:
        bridge.shutdown()


def test_session_and_push_to_talk_are_refused_while_a_call_holds_the_voice(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    errors = _collect(bridge.error)
    states = _collect(bridge.session_state_changed)
    try:
        _phone_claim(bridge, reason="call")
        bridge.start_session()
        assert bridge._session is None
        assert _wait_until(lambda: any("busy with a call" in e for e in errors))
        assert _wait_until(lambda: states[-1:] == ["idle"])
        bridge.start_listening()  # same refusal (shown once: repeated notices are deduplicated)
        assert _wait_until(lambda: bridge._ptt_cancelled)
        bridge.stop_listening()
        time.sleep(0.1)
        assert recorder.ops == []
        assert not bridge._push_to_talk_active
    finally:
        bridge.shutdown()


def test_stale_wake_from_a_desktop_that_no_longer_owns_standby_is_ignored(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        _tablet(bridge, want_standby=True, mic_open=False)
        bridge.start_session(greet=True)  # a wake that raced the hand-over
        assert bridge._session is None
        time.sleep(0.1)
        assert recorder.ops == []
        assert bridge.voice_ownership.state_of(TABLET) == VoiceState.STANDBY
    finally:
        bridge.shutdown()


def test_push_to_talk_cut_off_by_priority_discards_the_recording(tmp_path):
    bridge, recorder, stt, _, agent_calls = _bridge(tmp_path)
    try:
        bridge.start_listening()
        assert _wait_until(lambda: recorder.open)
        _phone_claim(bridge, reason="call")
        assert _wait_until(lambda: not recorder.open)
        bridge.stop_listening()
        time.sleep(0.1)
        assert stt.calls == 0 and agent_calls == []
        assert bridge.voice_ownership.state_of(PHONE) == VoiceState.PRIORITY
    finally:
        bridge.shutdown()


def test_push_to_talk_returns_the_voice_to_the_tablet(tmp_path):
    bridge, recorder, stt, _, _ = _bridge(tmp_path)
    try:
        _tablet(bridge, want_standby=True, mic_open=False)
        bridge.start_listening()
        assert _wait_until(lambda: bridge.voice_ownership.state_of(TABLET) == VoiceState.SLEEP)
        assert not recorder.open
        _tablet_releases(bridge)  # having seen the order to sleep
        assert _wait_until(lambda: recorder.open)
        assert bridge.voice_ownership.state_of(TABLET) == VoiceState.SLEEP
        bridge.stop_listening()
        assert _wait_until(lambda: bridge.voice_ownership.state_of(TABLET) == VoiceState.STANDBY)
        assert not recorder.open
    finally:
        bridge.shutdown()


def test_voice_replies_stay_silent_while_another_device_holds_the_voice(tmp_path):
    from app.core.events import AgentEvent, EventType
    from tests.test_agent_bridge_standby import _on_loop

    bridge, _, _, tts, _ = _bridge(tmp_path)
    bridge.set_voice_replies_enabled(True)
    try:
        _tablet(bridge, want_standby=True, mic_open=False)
        _on_loop(bridge, bridge._events.publish(AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "Hi."})))
        time.sleep(0.1)
        assert tts.synthesize_calls == []
        bridge.voice_ownership.leave(TABLET)
        _on_loop(bridge, bridge._events.publish(AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "Hi."})))
        assert _wait_until(lambda: tts.synthesize_calls == ["Hi."])
    finally:
        bridge.shutdown()


def test_ownership_changes_reach_the_window_as_one_line(tmp_path):
    bridge, _, _, _, _ = _bridge(tmp_path)
    lines: list[str] = []
    bridge.voice_owner_changed.connect(lines.append, Qt.ConnectionType.DirectConnection)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: lines and lines[-1].startswith("This PC is the listener"))
        _phone_claim(bridge, reason="call")
        assert _wait_until(lambda: "delivering a reminder or call" in lines[-1])
    finally:
        bridge.shutdown()
