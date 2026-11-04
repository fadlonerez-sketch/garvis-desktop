"""Desktop side of the mobile interface (app/ui/main_window.py): the
confirmation dialog closes when the phone answers first, phone turns are
shown (escaped), and the iPhone pairing dialog. Offscreen Qt, fake bridge."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

import app.ui.main_window as main_window
from app.mobile.pairing import PairingCode


class _FakePairing:
    def __init__(self) -> None:
        self.cancelled = 0
        self.revoked = 0

    def new_pairing_code(self) -> PairingCode:
        return PairingCode(code="12345678", expires_at=time.time() + 300)

    def cancel_pairing_code(self) -> None:
        self.cancelled += 1

    def revoke_all(self) -> int:
        self.revoked += 1
        return 2

    def device_count(self) -> int:
        return 1


class _FakeMobile:
    is_running = True
    urls = ["https://192.168.1.20:8443/"]
    ca_fingerprint = "AA:BB"

    def __init__(self) -> None:
        self.pairing = _FakePairing()


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
        self.resolutions: list[tuple[str, bool]] = []
        self.mobile = _FakeMobile()

    def resolve_confirmation(self, request_id: str, approved: bool) -> None:
        self.resolutions.append((request_id, approved))

    def set_voice_replies_enabled(self, enabled: bool) -> None:
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


def test_dialog_closes_without_answering_when_the_phone_answered_first(window):
    bridge = window._bridge
    QTimer.singleShot(50, lambda: bridge.confirmation_resolved.emit("req-1", "approved"))
    window._on_confirmation_requested("req-1", "send_email", {"to": "x"}, "high risk")
    assert bridge.resolutions == []  # the phone's answer stands; no second answer
    assert window._confirmation_boxes == {}


def test_answering_on_the_desktop_still_resolves_the_request(window):
    bridge = window._bridge

    def click_yes() -> None:
        window._confirmation_boxes["req-2"].button(QMessageBox.StandardButton.Yes).click()

    QTimer.singleShot(50, click_yes)
    window._on_confirmation_requested("req-2", "send_email", {}, "high risk")
    assert bridge.resolutions == [("req-2", True)]
    bridge.confirmation_resolved.emit("req-2", "approved")  # late event: harmless
    assert bridge.resolutions == [("req-2", True)]


def test_phone_messages_are_shown_as_plain_text(window):
    window._bridge.remote_user_message.emit("<img src=x> hi", "mobile")
    text = window._conversation_view.toPlainText()
    assert "You (iPhone): <img src=x> hi" in text


def test_pairing_dialog_shows_url_fingerprint_and_a_code_then_cancels_it():
    QApplication.instance() or QApplication([])
    mobile = _FakeMobile()
    dialog = main_window.MobileDialog(mobile)
    text = dialog._text.toPlainText()
    assert "https://192.168.1.20:8443/" in text
    assert "https://192.168.1.20:8443/garvis-ca.crt" in text
    assert "AA:BB" in text and "1234 5678" in text
    dialog._unpair_all()
    assert mobile.pairing.revoked == 1 and "Unpaired 2" in dialog._text.toPlainText()
    dialog._new_code()
    dialog.accept()
    assert mobile.pairing.cancelled == 1


def test_pairing_dialog_explains_how_to_enable_when_off():
    QApplication.instance() or QApplication([])
    dialog = main_window.MobileDialog(None)
    assert "GARVIS_MOBILE_ENABLED=true" in dialog._text.toPlainText()
    dialog.accept()
