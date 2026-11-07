"""Regression tests for AgentBridge's session start/end idempotency
guards (app/ui/agent_bridge.py) -- written for the investigation into a
reported runtime bug where a session appeared to produce multiple rapid,
unexpected Agent turns.

These only exercise the SYNCHRONOUS bookkeeping in start_session()/
end_session() (setting/clearing self._session, self._session_active,
self._pending_session_stop) -- that bookkeeping happens on the calling
(GUI) thread before anything is scheduled onto AgentBridge's background
asyncio loop, so it's fully deterministic without needing to fake a
microphone or wait on a real background thread. A real AgentBridge is
still constructed (not a duck-typed stand-in) so these tests exercise the
actual guard logic, not a reimplementation of it.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.events import AgentEvent, EventType
from app.ui.agent_bridge import AgentBridge


def _make_bridge(tmp_path) -> AgentBridge:
    settings = Settings(
        memory_db_path=tmp_path / "test.db",
        llm_provider="tiered",  # providers constructed lazily, no network call at build time
        omniroute_api_key="fake-test-key",
    )
    return AgentBridge(settings=settings)


def test_start_session_creates_exactly_one_session(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        assert bridge._session is None
        bridge.start_session()
        session = bridge._session
        assert session is not None
        assert bridge._session_active is True
    finally:
        bridge.shutdown()


def test_start_session_twice_does_not_create_a_second_session(tmp_path):
    """Regression guard: a double-click (or any duplicate call) on Start
    Session must not create two ConversationSession instances."""
    bridge = _make_bridge(tmp_path)
    try:
        bridge.start_session()
        first_session = bridge._session
        bridge.start_session()  # second call while already "active"
        assert bridge._session is first_session
    finally:
        bridge.shutdown()


def test_end_session_without_an_active_session_is_a_noop(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        assert bridge._session is None
        bridge.end_session()  # must not raise, must not schedule anything
        assert bridge._session is None
        assert bridge._pending_session_stop is None
    finally:
        bridge.shutdown()


def test_end_session_clears_session_state_immediately(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        bridge.start_session()
        assert bridge._session is not None
        bridge.end_session()
        # Cleared synchronously -- a subsequent start_session() (or Agent
        # call) must never see the old session as still "current", even
        # though its async stop() coroutine may still be running.
        assert bridge._session is None
        assert bridge._session_active is False
    finally:
        bridge.shutdown()


def test_end_session_twice_is_idempotent(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        bridge.start_session()
        bridge.end_session()
        pending_after_first_end = bridge._pending_session_stop
        bridge.end_session()  # second call: self._session is already None
        assert bridge._pending_session_stop is pending_after_first_end
    finally:
        bridge.shutdown()


def test_restarting_after_end_creates_a_genuinely_new_session_object(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        bridge.start_session()
        first_session = bridge._session
        bridge.end_session()
        assert bridge._session is None

        bridge.start_session()
        second_session = bridge._session
        assert second_session is not None
        assert second_session is not first_session
    finally:
        bridge.shutdown()


def test_session_active_flag_suppresses_the_normal_voice_replies_auto_speak_path():
    """Regression guard for the double-speak risk this flag exists to
    prevent: while a session is active, the ASSISTANT_MESSAGE auto-speak
    branch in _on_event must not also fire (the session already speaks
    each reply itself, interruptibly)."""
    from unittest.mock import Mock

    class _FakeBridge:
        _assistant_message_seq = 0
        _voice_replies_enabled = True
        _session_active = True  # session is active
        assistant_message = Mock()
        _loop = Mock()

    fake = _FakeBridge()
    AgentBridge._on_event(fake, AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "hello"}))
    fake._loop.create_task.assert_not_called()


def test_session_active_false_allows_the_normal_voice_replies_auto_speak_path():
    """Sanity check for the test above: the suppression is specific to an
    active session, not a general regression -- outside a session, the
    existing "Speak replies" checkbox behavior must be unaffected."""
    from unittest.mock import Mock

    class _FakeBridge:
        _assistant_message_seq = 0
        _voice_replies_enabled = True
        _session_active = False  # no session active
        assistant_message = Mock()
        _loop = Mock()
        _may_speak_aloud = Mock(return_value=True)  # no other Garvis device holds the voice
        _safe_speak = Mock()  # a plain Mock, not a real coroutine function --
        # avoids an unawaited-coroutine warning since _loop.create_task is
        # also a Mock here and never actually runs what's passed to it.

    fake = _FakeBridge()
    AgentBridge._on_event(fake, AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "hello"}))
    fake._loop.create_task.assert_called_once()


def test_session_state_event_reaches_the_signal_exactly_once(tmp_path):
    """Regression guard against a double EventBus subscription: publishing
    one SESSION_STATE event must fire session_state_changed exactly once."""
    bridge = _make_bridge(tmp_path)
    try:
        received: list[str] = []
        bridge.session_state_changed.connect(received.append)

        import asyncio

        asyncio.run(bridge._events.publish(AgentEvent(type=EventType.SESSION_STATE, payload={"state": "listening"})))

        assert received == ["listening"]
    finally:
        bridge.shutdown()
