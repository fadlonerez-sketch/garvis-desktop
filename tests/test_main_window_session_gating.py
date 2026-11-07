"""Regression tests: Start Session is disabled while a typed or push-to-talk
turn is running (app/ui/main_window.py).

Starting a session resets the Agent's conversation context, so it must not
happen while a manual turn still owns that context. The real MainWindow is
built against a fake AgentBridge (same Qt signals, records calls), so no
Agent, LLM, microphone or background loop is created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import app.ui.main_window as main_window


class _FakeBridge(QObject):
    status_changed = Signal(str)
    user_message = Signal(str)
    assistant_message = Signal(str)
    tool_started = Signal(str, dict)
    tool_finished = Signal(str, bool, str)
    confirmation_requested = Signal(str, str, dict, str)
    confirmation_resolved = Signal(str, str)
    remote_user_message = Signal(str, str)
    error = Signal(str)
    listening_changed = Signal(bool)
    session_state_changed = Signal(str)
    standby_changed = Signal(bool)
    wake_detected = Signal(str)
    sleep_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def send_message(self, text: str) -> None:
        self.calls.append("send_message")

    def start_listening(self) -> None:
        self.calls.append("start_listening")

    def stop_listening(self) -> None:
        self.calls.append("stop_listening")

    def start_session(self) -> None:
        self.calls.append("start_session")

    def end_session(self) -> None:
        self.calls.append("end_session")

    def set_voice_replies_enabled(self, enabled: bool) -> None:
        pass

    def stop_speaking(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.fixture
def window(monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(main_window, "AgentBridge", _FakeBridge)
    win = main_window.MainWindow()
    yield win
    win.close()
    win.deleteLater()


def _send_typed(win, text: str = "hello") -> None:
    win._input_line.setText(text)
    win._send_button.click()


def _start_push_to_talk(win) -> None:
    win._mic_button.click()
    win._bridge.listening_changed.emit(True)


def test_start_session_is_enabled_when_idle(window):
    assert window._session_button.isEnabled()


def test_start_session_is_disabled_during_a_typed_turn(window):
    _send_typed(window)
    assert window._bridge.calls == ["send_message"]
    assert not window._session_button.isEnabled()
    window._bridge.status_changed.emit("Thinking...")
    assert not window._session_button.isEnabled()


def test_start_session_is_disabled_during_push_to_talk(window):
    _start_push_to_talk(window)
    assert window._bridge.calls == ["start_listening"]
    assert not window._session_button.isEnabled()

    window._mic_button.click()  # stop recording -> transcribe -> agent turn
    window._bridge.listening_changed.emit(False)
    for status in ("Transcribing...", "Thinking..."):
        window._bridge.status_changed.emit(status)
        assert not window._session_button.isEnabled()


@pytest.mark.parametrize("final_status", ["Done.", "Idle."])
def test_start_session_is_re_enabled_when_a_typed_turn_ends(window, final_status):
    _send_typed(window)
    window._bridge.status_changed.emit(final_status)
    assert window._session_button.isEnabled()
    # Existing behavior unchanged.
    assert window._send_button.isEnabled()
    assert not window._stop_button.isEnabled()


@pytest.mark.parametrize("final_status", ["Done.", "Idle."])
def test_start_session_is_re_enabled_when_push_to_talk_ends(window, final_status):
    _start_push_to_talk(window)
    window._mic_button.click()
    window._bridge.listening_changed.emit(False)
    window._bridge.status_changed.emit(final_status)  # "Idle." = empty transcript
    assert window._session_button.isEnabled()


def test_start_session_is_re_enabled_after_an_error(window):
    _send_typed(window)
    window._bridge.error.emit("Garvis can't reach OmniRoute.")
    assert window._session_button.isEnabled()
    assert window._send_button.isEnabled()


def test_start_session_is_re_enabled_after_a_push_to_talk_error(window):
    _start_push_to_talk(window)
    window._bridge.error.emit("Microphone unavailable.")
    assert window._session_button.isEnabled()
    assert not window._mic_button.property("recording")


def test_a_session_cannot_start_while_a_turn_is_running(window):
    _send_typed(window)
    window._session_button.click()  # disabled -> click is ignored
    assert "start_session" not in window._bridge.calls
    assert window._session_button.text() == "Start Session"

    window._bridge.status_changed.emit("Done.")
    window._session_button.click()
    assert window._bridge.calls[-1] == "start_session"


def test_session_turns_do_not_disable_end_session(window):
    """Existing behavior: during a session the button is "End Session" and
    stays clickable through the session's own turns and errors."""
    window._session_button.click()
    assert window._session_button.text() == "End Session"
    for status in ("Thinking...", "Done.", "Idle."):
        window._bridge.status_changed.emit(status)
        assert window._session_button.isEnabled()
    window._bridge.error.emit("one failed transcription")
    assert window._session_button.isEnabled()
    assert not window._send_button.isEnabled()

    window._session_button.click()
    assert window._bridge.calls == ["start_session", "end_session"]
