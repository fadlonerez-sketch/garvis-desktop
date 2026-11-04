"""Lifecycle tests for wake-word standby in the real AgentBridge
(app/ui/agent_bridge.py): standby <-> session <-> push-to-talk hand-offs of
the single shared microphone, greeting/no-greeting session starts, End
Session returning to standby, shutdown, and that nothing in the wake path
reaches the Agent/LLM.

A real AgentBridge (real Agent, real background loop) is used, but its
VoicePipeline is swapped for one built on a thread-safe fake recorder, fake
STT/TTS and fake player -- no real microphone, model or speaker.
"""
from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
import time
import wave

import pytest
from PySide6.QtCore import Qt

from app.core.config import Settings
from app.llm.base import Role
from app.ui.agent_bridge import AgentBridge
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import TranscriptionResult
from app.voice.wake import GREETINGS, WakeMatch
from tests.test_conversation_session import _FakePlayer, _FakeTTS, _silence_chunk, _split, _utterance_chunks

_SAMPLE_RATE = 16000


class _ThreadSafeRecorder:
    """Recorder fake fed from the test thread and drained on the bridge
    loop. Logs every open/close so mic ownership order can be asserted."""

    sample_rate = _SAMPLE_RATE

    def __init__(self) -> None:
        self._chunks: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self.ops: list[str] = []
        self.open = False

    @property
    def is_recording(self) -> bool:
        return self.open

    def start(self) -> None:
        with self._lock:
            if self.open:
                return
            self.open = True
            self.ops.append("open")

    def read_available(self) -> bytes:
        # One fed chunk per poll, like a real mic delivering audio
        # incrementally (see tests/test_conversation_session.py:_split).
        try:
            return self._chunks.get_nowait()
        except queue.Empty:
            return b""

    def close(self) -> None:
        with self._lock:
            if self.open:
                self.ops.append("close")
            self.open = False

    def stop(self) -> str:  # push-to-talk
        self.close()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(_SAMPLE_RATE)
            wav_file.writeframes(_silence_chunk(0.5))
        return path

    def feed(self, chunks: list[bytes]) -> None:
        for chunk in chunks:
            self._chunks.put(chunk)


class _FakeSTT:
    def __init__(self) -> None:
        self.text = ""
        self.calls = 0

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(text=self.text)


def _settings(tmp_path, **overrides) -> Settings:
    params = dict(
        _env_file=None,
        memory_db_path=tmp_path / "standby.db",
        llm_provider="tiered",
        omniroute_api_key="fake-test-key",
        session_poll_interval_seconds=0.002,
        session_min_speech_seconds=0.1,
        wake_min_silence_seconds=0.3,
        wake_playback_tail_seconds=0.0,
    )
    params.update(overrides)
    return Settings(**params)


def _bridge(tmp_path, **overrides):
    bridge = AgentBridge(settings=_settings(tmp_path, **overrides))
    recorder = _ThreadSafeRecorder()
    stt = _FakeSTT()
    tts = _FakeTTS()
    bridge._voice = VoicePipeline(stt=stt, tts=tts, recorder=recorder, player=_FakePlayer(finish_after_polls=3))
    agent_calls: list[str] = []
    original = bridge._agent.handle_user_message

    async def _spy(text: str):
        agent_calls.append(text)
        return await original(text)

    bridge._agent.handle_user_message = _spy
    return bridge, recorder, stt, tts, agent_calls


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _on_loop(bridge: AgentBridge, coro):
    return asyncio.run_coroutine_threadsafe(coro, bridge._loop).result(timeout=5)


def _settle_session_end(bridge: AgentBridge) -> None:
    if bridge._pending_session_stop is not None:
        bridge._pending_session_stop.result(timeout=5)


# ----------------------------------------------------------------- tests
def test_wake_mode_is_off_by_default(tmp_path):
    assert Settings(_env_file=None).wake_word_enabled is False
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        time.sleep(0.05)
        assert bridge.standby_enabled is False
        assert not bridge.is_standby_active
        assert recorder.ops == []
    finally:
        bridge.shutdown()


def test_enabling_standby_opens_the_mic_and_disabling_releases_it(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        assert recorder.open
        bridge.set_standby_enabled(False)
        # The listener is detached first, then closes the recorder: wait for the release itself.
        assert _wait_until(lambda: not bridge.is_standby_active and not recorder.open)
        assert recorder.ops == ["open", "close"]
    finally:
        bridge.shutdown()


def test_wake_word_enabled_setting_no_longer_claims_the_microphone_at_launch(tmp_path, monkeypatch):
    # Voice ownership (app/core/voice_ownership.py): the Desktop never claims standby just
    # because it started -- only the user ticking "Wake word standby" does.
    import app.ui.agent_bridge as agent_bridge_module

    recorder = _ThreadSafeRecorder()
    fake_voice = VoicePipeline(stt=_FakeSTT(), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    # Patched before construction so the real microphone is never opened.
    monkeypatch.setattr(agent_bridge_module, "build_voice_pipeline", lambda settings: fake_voice)
    bridge = AgentBridge(settings=_settings(tmp_path, wake_word_enabled=True))
    try:
        time.sleep(0.1)
        assert bridge.standby_enabled is False
        assert not bridge.is_standby_active
        assert recorder.ops == []
        assert bridge.voice_ownership.holder() is None
        bridge.set_standby_enabled(True)  # the explicit request
        assert _wait_until(lambda: bridge.is_standby_active)
        assert recorder.open
    finally:
        bridge.shutdown()
    assert not bridge.is_standby_active
    assert not recorder.open


def test_wake_phrase_emits_wake_detected_and_never_reaches_the_agent(tmp_path):
    bridge, recorder, stt, _, agent_calls = _bridge(tmp_path)
    phrases: list[str] = []
    # Direct: emitted on the bridge loop, and no Qt event loop runs here.
    bridge.wake_detected.connect(phrases.append, Qt.ConnectionType.DirectConnection)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        stt.text = "Hey Garvis, what's the weather"
        recorder.feed(_utterance_chunks(speech_seconds=0.6, silence_seconds=0.5))
        assert _wait_until(lambda: phrases == ["Hey Garvis"])
        time.sleep(0.05)
        assert agent_calls == []
        assert all(m.role == Role.SYSTEM for m in bridge._agent._context.messages)
    finally:
        bridge.shutdown()


def test_wake_start_stops_standby_before_the_session_opens_the_mic(tmp_path):
    bridge, recorder, _, tts, agent_calls = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        recorder.feed(_split(0.5, 0.02, _silence_chunk))  # let standby learn the room
        assert _wait_until(lambda: bridge._standby is not None and bridge._standby.noise_floor is not None)
        learned = bridge._standby.noise_floor

        bridge.start_session(greet=True)
        session = bridge._session
        assert _wait_until(lambda: session.is_active)
        assert not bridge.is_standby_active
        assert recorder.ops == ["open", "close", "open"]  # standby released, then session opened
        assert session._greeting in GREETINGS
        assert _wait_until(lambda: tts.synthesize_calls == [session._greeting])
        assert session._noise_floor == pytest.approx(learned, rel=0.5)
        assert agent_calls == []
        assert all(m.role == Role.SYSTEM for m in bridge._agent._context.messages)
    finally:
        bridge.shutdown()


def test_manual_start_session_has_no_greeting_and_no_seeded_floor(tmp_path):
    bridge, recorder, _, tts, _ = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        recorder.feed(_split(0.5, 0.02, _silence_chunk))
        time.sleep(0.05)

        bridge.start_session()
        session = bridge._session
        assert session._greeting is None
        assert session._noise_floor is None
        assert _wait_until(lambda: session.is_active)
        assert not bridge.is_standby_active
        time.sleep(0.05)
        assert tts.synthesize_calls == []
    finally:
        bridge.shutdown()


def test_manual_start_session_without_wake_mode_is_unchanged(tmp_path):
    bridge, recorder, _, tts, _ = _bridge(tmp_path)
    try:
        bridge.start_session()
        session = bridge._session
        assert _wait_until(lambda: session.is_active)
        assert recorder.ops == ["open"]
        assert session._greeting is None
        bridge.end_session()
        _settle_session_end(bridge)
        assert recorder.ops == ["open", "close"]
        assert not bridge.is_standby_active
        assert tts.synthesize_calls == []
    finally:
        bridge.shutdown()


def test_wake_session_resets_the_conversation_first(tmp_path):
    from app.llm.base import Message

    bridge, _, _, _, _ = _bridge(tmp_path)
    try:
        context = bridge._agent._context
        context.add(Message(role=Role.USER, content="old typed turn"))
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        assert _wait_until(lambda: bridge._session is not None and bridge._session.is_active)
        assert all(m.role == Role.SYSTEM for m in context.messages)
    finally:
        bridge.shutdown()


def test_end_session_returns_to_standby_when_enabled(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        assert _wait_until(lambda: bridge._session.is_active)
        bridge.end_session()
        _settle_session_end(bridge)
        assert bridge.is_standby_active
        assert recorder.open
        assert recorder.ops == ["open", "close", "open", "close", "open"]
    finally:
        bridge.shutdown()


def test_end_session_does_not_start_standby_when_disabled(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.start_session()
        assert _wait_until(lambda: bridge._session.is_active)
        bridge.end_session()
        _settle_session_end(bridge)
        assert not bridge.is_standby_active
        assert not recorder.open
    finally:
        bridge.shutdown()


def test_end_then_immediate_start_does_not_race_standby_against_the_session(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        assert _wait_until(lambda: bridge._session.is_active)

        bridge.end_session()
        bridge.start_session()  # before the old session finished stopping
        new_session = bridge._session
        assert _wait_until(lambda: new_session.is_active)
        time.sleep(0.05)
        assert not bridge.is_standby_active
        assert recorder.open
        # The mic was never opened twice without a close in between.
        assert all(a != b for a, b in zip(recorder.ops, recorder.ops[1:]))
    finally:
        bridge.shutdown()


def test_push_to_talk_suspends_standby_and_resumes_it_afterwards(tmp_path):
    bridge, recorder, stt, _, agent_calls = _bridge(tmp_path)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)

        bridge.start_listening()
        assert _wait_until(lambda: not bridge.is_standby_active and recorder.open)
        assert recorder.ops == ["open", "close", "open"]

        stt.text = ""  # empty transcript -> "Idle.", no agent turn
        bridge.stop_listening()
        assert _wait_until(lambda: bridge.is_standby_active)
        assert recorder.ops == ["open", "close", "open", "close", "open"]
        assert agent_calls == []
    finally:
        bridge.shutdown()


def test_standby_does_not_start_while_a_session_holds_the_mic(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    try:
        bridge.start_session()
        assert _wait_until(lambda: bridge._session.is_active)
        bridge.set_standby_enabled(True)
        time.sleep(0.05)
        assert not bridge.is_standby_active
        assert recorder.ops == ["open"]
    finally:
        bridge.shutdown()


def test_wake_remainder_is_dropped_in_v1(tmp_path):
    bridge, _, _, _, agent_calls = _bridge(tmp_path)
    phrases: list[str] = []
    # Direct: emitted on the bridge loop, and no Qt event loop runs here.
    bridge.wake_detected.connect(phrases.append, Qt.ConnectionType.DirectConnection)
    try:
        bridge._on_wake(WakeMatch(phrase="Hey Garvis", score=1.0, remainder="open chrome"))
        assert phrases == ["Hey Garvis"]
        time.sleep(0.05)
        assert agent_calls == []
    finally:
        bridge.shutdown()


def test_shutdown_stops_standby_and_releases_the_mic(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    bridge.set_standby_enabled(True)
    assert _wait_until(lambda: bridge.is_standby_active)
    bridge.shutdown()
    assert not bridge.is_standby_active
    assert not recorder.open
    assert bridge.standby_enabled is False


def test_shutdown_during_a_wake_session_does_not_restart_standby(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    bridge.set_standby_enabled(True)
    assert _wait_until(lambda: bridge.is_standby_active)
    bridge.start_session(greet=True)
    assert _wait_until(lambda: bridge._session is not None and bridge._session.is_active)
    bridge.shutdown()
    time.sleep(0.05)
    assert not bridge.is_standby_active
    assert not recorder.open


# ------------------------------------------------------- sleep / standby
def _speak_to_session(bridge, recorder, stt, text: str) -> None:
    session = bridge._session
    assert _wait_until(lambda: session.state.value == "listening")
    stt.text = text
    # Longer than Settings.session_min_silence_seconds (0.7s) so the utterance ends.
    recorder.feed(_utterance_chunks(speech_seconds=0.4, silence_seconds=1.0))


def test_standby_command_ends_the_session_and_returns_to_standby(tmp_path):
    bridge, recorder, stt, tts, agent_calls = _bridge(tmp_path)
    sleeps: list[bool] = []
    bridge.sleep_requested.connect(lambda: sleeps.append(True), Qt.ConnectionType.DirectConnection)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        assert _wait_until(lambda: bridge._session is not None and bridge._session.is_active)

        _speak_to_session(bridge, recorder, stt, "Go to standby.")
        assert _wait_until(lambda: sleeps == [True])
        assert agent_calls == []  # never sent to the Agent / LLM
        assert all(m.role == Role.SYSTEM for m in bridge._agent._context.messages)
        assert tts.synthesize_calls[-1] == "Entering standby, Sir."

        bridge.end_session()  # what the UI does on sleep_requested
        _settle_session_end(bridge)
        assert bridge.is_standby_active
        assert recorder.open
        assert bridge._thread.is_alive()  # app still running
        assert bridge._standby._initial_quiet_seconds == bridge._settings.wake_playback_tail_seconds
    finally:
        bridge.shutdown()


def test_next_wake_after_sleep_starts_a_fresh_session(tmp_path):
    bridge, recorder, stt, tts, agent_calls = _bridge(tmp_path)
    bridge.sleep_requested.connect(bridge.end_session, Qt.ConnectionType.DirectConnection)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        first = bridge._session
        assert _wait_until(lambda: first.is_active)
        _speak_to_session(bridge, recorder, stt, "Goodnight, Garvis.")
        assert _wait_until(lambda: bridge._session is None and bridge.is_standby_active)

        stt.text = ""
        bridge.start_session(greet=True)
        second = bridge._session
        assert second is not first
        assert _wait_until(lambda: second.is_active)
        assert second._greeting in GREETINGS
        assert agent_calls == []
        assert all(m.role == Role.SYSTEM for m in bridge._agent._context.messages)
    finally:
        bridge.shutdown()


def test_sleep_commands_are_disabled_without_wake_mode(tmp_path):
    bridge, recorder, stt, _, _ = _bridge(tmp_path)
    calls: list[str] = []

    async def _agent(text: str) -> str:  # no LLM call in this test
        calls.append(text)
        return ""

    bridge._agent.handle_user_message = _agent
    sleeps: list[bool] = []
    bridge.sleep_requested.connect(lambda: sleeps.append(True), Qt.ConnectionType.DirectConnection)
    try:
        bridge.start_session()
        assert bridge._session._sleep_matcher is None
        _speak_to_session(bridge, recorder, stt, "Go to standby.")
        assert _wait_until(lambda: calls == ["Go to standby."])
        assert sleeps == []
        assert bridge._session is not None and bridge._session.is_active
    finally:
        bridge.shutdown()


def test_stale_sleep_request_from_an_ended_session_is_ignored(tmp_path):
    bridge, _, _, _, _ = _bridge(tmp_path)
    sleeps: list[bool] = []
    bridge.sleep_requested.connect(lambda: sleeps.append(True), Qt.ConnectionType.DirectConnection)
    try:
        bridge.start_session()
        old = bridge._session
        bridge.end_session()
        bridge.start_session()
        bridge._on_session_sleep(old)
        assert sleeps == []
        bridge._on_session_sleep(bridge._session)
        assert sleeps == [True]
    finally:
        bridge.shutdown()


# ------------------------------------------------- typed standby commands
def _collect(signal) -> list:
    received: list = []
    signal.connect(lambda *args: received.append(args[0] if len(args) == 1 else args),
                   Qt.ConnectionType.DirectConnection)
    return received


def test_typed_standby_is_handled_locally_and_keeps_garvis_running(tmp_path):
    bridge, recorder, _, tts, agent_calls = _bridge(tmp_path)
    replies = _collect(bridge.assistant_message)
    statuses = _collect(bridge.status_changed)
    users = _collect(bridge.user_message)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        standby = bridge._standby

        bridge.send_message("Good night, Jarvis.")
        assert _wait_until(lambda: replies == ["Goodnight, Sir."])
        assert _wait_until(lambda: statuses[-1:] == ["Done."])
        assert users == ["Good night, Jarvis."]
        assert agent_calls == []  # no Agent turn, so no model request
        assert all(m.role == Role.SYSTEM for m in bridge._agent._context.messages)
        assert tts.synthesize_calls == []  # voice replies are off: shown, not spoken
        assert bridge._standby is standby and standby.is_active  # same listener, still armed
        assert recorder.open
        assert bridge._thread.is_alive() and bridge._loop.is_running()
    finally:
        bridge.shutdown()


def test_repeated_typed_standby_is_idempotent(tmp_path):
    bridge, recorder, _, _, agent_calls = _bridge(tmp_path)
    replies = _collect(bridge.assistant_message)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        standby = bridge._standby
        for text in ("Jarvis, enter standby mode.", "Go to sleep.", "Go to sleep."):
            bridge.send_message(text)
        assert _wait_until(lambda: len(replies) == 3)
        assert set(replies) == {"Entering standby, Sir."}
        assert bridge._standby is standby
        assert recorder.ops.count("open") == 1  # the mic was never reopened
        assert agent_calls == []
    finally:
        bridge.shutdown()


def test_typed_standby_cuts_off_a_reply_that_is_still_playing(tmp_path):
    bridge, recorder, stt, tts, agent_calls = _bridge(tmp_path)
    player = _FakePlayer(finish_after_polls=None)
    bridge._voice = VoicePipeline(stt=stt, tts=tts, recorder=recorder, player=player)
    bridge.set_voice_replies_enabled(True)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        _on_loop(bridge, bridge._voice.speak("An earlier, very long answer."))
        assert player.is_playing

        bridge.send_message("Jarvis, go to sleep.")
        assert _wait_until(lambda: tts.synthesize_calls[-1:] == ["Entering standby, Sir."])
        assert player.stop_calls >= 1
        assert agent_calls == []
        assert bridge.is_standby_active
    finally:
        bridge.shutdown()


def test_typed_messages_that_only_mention_standby_reach_the_agent(tmp_path):
    bridge, _, _, _, _ = _bridge(tmp_path)
    calls: list[str] = []

    async def _agent(text: str) -> str:  # no LLM call in this test
        calls.append(text)
        return ""

    bridge._agent.handle_user_message = _agent
    replies = _collect(bridge.assistant_message)
    texts = ["What does standby mode mean?", "Don't go to sleep.", "Shut down Garvis.", "Quit."]
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        for text in texts:
            bridge.send_message(text)
        assert _wait_until(lambda: calls == texts)
        assert replies == []
    finally:
        bridge.shutdown()


def test_typed_standby_without_wake_mode_goes_to_the_agent_as_before(tmp_path):
    bridge, recorder, _, _, _ = _bridge(tmp_path)
    calls: list[str] = []

    async def _agent(text: str) -> str:  # no LLM call in this test
        calls.append(text)
        return ""

    bridge._agent.handle_user_message = _agent
    try:
        bridge.send_message("Go to sleep.")
        assert _wait_until(lambda: calls == ["Go to sleep."])
        assert not bridge.is_standby_active
        assert not recorder.open  # wake mode off: nothing starts listening
    finally:
        bridge.shutdown()


def test_wake_after_typed_standby_starts_a_greeted_session(tmp_path):
    bridge, _, _, _, agent_calls = _bridge(tmp_path)
    replies = _collect(bridge.assistant_message)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.send_message("Time to sleep.")
        assert _wait_until(lambda: replies == ["Entering standby, Sir."])

        bridge.start_session(greet=True)  # what the UI does on wake_detected
        session = bridge._session
        assert _wait_until(lambda: session.is_active)
        assert session._greeting in GREETINGS
        assert not bridge.is_standby_active  # handed the mic to the session
        assert agent_calls == []
    finally:
        bridge.shutdown()


def test_voice_standby_never_calls_shutdown(tmp_path):
    bridge, recorder, stt, _, agent_calls = _bridge(tmp_path)
    bridge.sleep_requested.connect(bridge.end_session, Qt.ConnectionType.DirectConnection)
    shutdowns: list[bool] = []
    real_shutdown = bridge.shutdown
    bridge.shutdown = lambda: shutdowns.append(True)
    try:
        bridge.set_standby_enabled(True)
        assert _wait_until(lambda: bridge.is_standby_active)
        bridge.start_session(greet=True)
        assert _wait_until(lambda: bridge._session is not None and bridge._session.is_active)
        _speak_to_session(bridge, recorder, stt, "Jarvis, enter standby mode.")
        assert _wait_until(lambda: bridge._session is None and bridge.is_standby_active)
        assert shutdowns == []
        assert bridge._wake_enabled  # standby stays available for the next wake
        assert bridge._thread.is_alive()
        assert agent_calls == []
    finally:
        real_shutdown()
