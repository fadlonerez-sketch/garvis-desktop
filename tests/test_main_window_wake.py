"""MainWindow wake-word handling (app/ui/main_window.py): a wake goes
through the same busy gating as the Start Session button, starts the
session with a greeting, and the Standby indicator follows the bridge.

Same approach as tests/test_main_window_session_gating.py: the real
MainWindow against a fake AgentBridge exposing the same Qt signals.
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
        self.calls: list[tuple] = []

    def send_message(self, text: str) -> None:
        self.calls.append(("send_message",))

    def start_listening(self) -> None:
        self.calls.append(("start_listening",))

    def stop_listening(self) -> None:
        self.calls.append(("stop_listening",))

    def start_session(self, greet: bool = False) -> None:
        self.calls.append(("start_session", greet))

    def end_session(self) -> None:
        self.calls.append(("end_session",))

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


def _session_starts(win) -> list[tuple]:
    return [c for c in win._bridge.calls if c[0] == "start_session"]


def test_wake_when_idle_starts_a_greeted_session(window):
    window._bridge.wake_detected.emit("Hey Garvis")
    assert _session_starts(window) == [("start_session", True)]
    assert window._session_button.text() == "End Session"
    assert not window._send_button.isEnabled()
    assert not window._input_line.isEnabled()
    assert not window._mic_button.isEnabled()


def test_manual_start_session_does_not_greet(window):
    window._session_button.click()
    assert _session_starts(window) == [("start_session", False)]


def test_wake_during_a_typed_turn_is_ignored(window):
    window._input_line.setText("hello")
    window._send_button.click()
    window._bridge.wake_detected.emit("Hey Garvis")
    assert _session_starts(window) == []
    assert window._session_button.text() == "Start Session"


def test_wake_during_push_to_talk_is_ignored(window):
    window._mic_button.click()
    window._bridge.listening_changed.emit(True)
    window._bridge.wake_detected.emit("Hey Daddy")
    assert _session_starts(window) == []


def test_wake_is_accepted_again_once_the_turn_finishes(window):
    window._input_line.setText("hello")
    window._send_button.click()
    window._bridge.wake_detected.emit("Hey Garvis")
    window._bridge.status_changed.emit("Done.")
    window._bridge.wake_detected.emit("Hey Garvis")
    assert _session_starts(window) == [("start_session", True)]


def test_wake_during_an_active_session_does_not_start_a_second_one(window):
    window._session_button.click()
    window._bridge.wake_detected.emit("Hey Garvis")
    assert _session_starts(window) == [("start_session", False)]
    assert ("end_session",) not in window._bridge.calls


def test_end_session_after_a_wake_restores_controls(window):
    window._bridge.wake_detected.emit("Hey Garvis")
    window._session_button.click()
    assert window._bridge.calls[-1] == ("end_session",)
    window._bridge.session_state_changed.emit("idle")
    assert window._session_button.text() == "Start Session"
    assert window._send_button.isEnabled()
    assert window._mic_button.isEnabled()


def test_standby_indicator_follows_the_bridge(window):
    label = window._session_status_label
    window._bridge.standby_changed.emit(True)
    assert label.text() == "Standby"

    window._bridge.wake_detected.emit("Hey Garvis")
    window._bridge.standby_changed.emit(False)  # bridge releases the mic for the session
    assert label.text() == ""
    window._bridge.session_state_changed.emit("speaking")
    assert label.text() == "Speaking..."

    window._session_button.click()
    window._bridge.session_state_changed.emit("stopping")
    window._bridge.session_state_changed.emit("idle")
    window._bridge.standby_changed.emit(True)  # End Session -> back to standby
    assert label.text() == "Standby"


def test_standby_off_does_not_clear_a_session_state_label(window):
    window._session_button.click()
    window._bridge.session_state_changed.emit("listening")
    window._bridge.standby_changed.emit(False)
    assert window._session_status_label.text() == "Listening..."


def test_standby_on_does_not_overwrite_a_running_session_label(window):
    window._session_button.click()
    window._bridge.session_state_changed.emit("thinking")
    window._bridge.standby_changed.emit(True)
    assert window._session_status_label.text() == "Thinking..."


# ------------------------------------------------------- sleep / standby
def test_sleep_request_ends_the_session_like_the_end_button(window):
    window._bridge.wake_detected.emit("Hey Garvis")
    window._bridge.sleep_requested.emit()
    assert window._bridge.calls[-1] == ("end_session",)
    window._bridge.session_state_changed.emit("idle")
    window._bridge.standby_changed.emit(True)
    assert window._session_button.text() == "Start Session"
    assert window._send_button.isEnabled()
    assert window._session_status_label.text() == "Standby"


def test_sleep_request_without_a_session_is_ignored(window):
    window._bridge.sleep_requested.emit()
    assert ("end_session",) not in window._bridge.calls


def test_wake_after_sleep_starts_a_new_greeted_session(window):
    window._bridge.wake_detected.emit("Hey Garvis")
    window._bridge.sleep_requested.emit()
    window._bridge.session_state_changed.emit("idle")
    window._bridge.wake_detected.emit("Hey Daddy")
    assert _session_starts(window) == [("start_session", True), ("start_session", True)]
    assert window._session_button.text() == "End Session"


def test_sleep_request_never_closes_garvis(window):
    shutdowns: list[bool] = []
    window._bridge.shutdown = lambda: shutdowns.append(True)
    window.show()
    window._bridge.wake_detected.emit("Hey Garvis")
    window._bridge.sleep_requested.emit()
    window._bridge.session_state_changed.emit("idle")
    window._bridge.standby_changed.emit(True)
    assert shutdowns == []
    assert window.isVisible()
    assert window._session_status_label.text() == "Standby"


def test_typed_standby_reply_restores_the_controls(window):
    window._input_line.setText("Good night, Jarvis.")
    window._send_button.click()
    assert window._bridge.calls[-1] == ("send_message",)
    assert not window._send_button.isEnabled()
    # What AgentBridge emits for a typed standby command (no Agent turn).
    window._bridge.assistant_message.emit("Goodnight, Sir.")
    window._bridge.status_changed.emit("Done.")
    assert window._send_button.isEnabled()
    assert window._session_button.isEnabled()  # a wake can start a session again
    assert "Goodnight, Sir." in window._conversation_view.toPlainText()
