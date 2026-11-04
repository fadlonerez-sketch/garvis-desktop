"""Single instance + global hotkey (Ctrl+Alt+G): registration, conflicts,
cleanup, activating the running instance, and never starting a second one."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid

import pytest

import app.main as main_module
from app.ui.global_hotkey import (
    ERROR_HOTKEY_ALREADY_REGISTERED,
    HOTKEY_ID,
    MOD_ALT,
    MOD_CONTROL,
    WM_HOTKEY,
    GlobalHotkey,
    parse_global_hotkey,
)
from app.ui.single_instance import ForwardResult, InstanceLock, forward_activation


# ------------------------------------------------------------ fakes
class FakeInstanceApi:
    def __init__(self, existing=False, windows=None, hung=False, post_ok=True):
        self.existing = existing
        self.windows = windows or []
        self.hung = hung
        self.post_ok = post_ok
        self.closed = []
        self.allowed = []
        self.posted = []

    def create_mutex(self, name):
        return 42, self.existing

    def close_handle(self, handle):
        self.closed.append(handle)

    def activate_message(self):
        return 0xC123

    def tagged_windows(self):
        return list(self.windows)

    def is_hung(self, hwnd):
        return self.hung

    def allow_foreground(self, pid):
        self.allowed.append(pid)

    def post(self, hwnd, message):
        self.posted.append((hwnd, message))
        return self.post_ok


class FakeHotkeyApi:
    def __init__(self, error=0):
        self.error = error
        self.registered = []
        self.unregistered = []

    def register(self, hwnd, hotkey_id, modifiers, vk):
        if self.error:
            return self.error
        self.registered.append((hwnd, hotkey_id, modifiers, vk))
        return 0

    def unregister(self, hwnd, hotkey_id):
        self.unregistered.append((hwnd, hotkey_id))


# ------------------------------------------------------- single instance
def test_first_instance_takes_the_lock_and_releases_it():
    api = FakeInstanceApi(existing=False)
    lock = InstanceLock(api)
    assert lock.acquire() is True
    lock.release()
    assert api.closed == [42]


def test_a_second_instance_does_not_get_the_lock():
    api = FakeInstanceApi(existing=True)
    assert InstanceLock(api).acquire() is False
    assert api.closed == [42]  # its own handle to the existing mutex is closed


def test_forwarding_activates_the_running_window_and_passes_foreground_rights():
    api = FakeInstanceApi(windows=[(1001, 777)])
    result = forward_activation(api, wait_seconds=0)
    assert result == ForwardResult("activated", 777)
    assert api.allowed == [777] and api.posted == [(1001, 0xC123)]


def test_a_hung_instance_is_reported_not_duplicated():
    api = FakeInstanceApi(windows=[(1001, 777)], hung=True)
    result = forward_activation(api, wait_seconds=0)
    assert result.status == "not_responding" and "777" in result.message
    assert api.posted == []


def test_an_instance_whose_window_never_appears_is_reported():
    now = [0.0]
    result = forward_activation(FakeInstanceApi(), wait_seconds=1.0, poll_seconds=0.5,
                                clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    assert result.status == "no_window"


def test_launching_again_hands_over_and_never_starts_a_second_garvis(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("app.ui.single_instance.InstanceLock.acquire", lambda self: False)
    monkeypatch.setattr("app.ui.single_instance.forward_activation", lambda: ForwardResult("activated", 1))
    # Any attempt to import the Qt app (i.e. to start a second Garvis) would raise.
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    assert main_module.main() == 0


def test_a_stuck_instance_is_explained_to_the_user(monkeypatch, capsys):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("app.ui.single_instance.InstanceLock.acquire", lambda self: False)
    monkeypatch.setattr("app.ui.single_instance.forward_activation", lambda: ForwardResult("not_responding", 55))
    assert main_module.main() == 1
    assert "not responding" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "win32", reason="Windows named mutex")
def test_real_mutex_allows_exactly_one_instance_per_session(tmp_path):
    name = f"Local\\Garvis.Test.{uuid.uuid4().hex}"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(os.getcwd())!r})
            from app.ui.single_instance import InstanceLock
            lock = InstanceLock(name={name!r})
            print("acquired" if lock.acquire() else "refused", flush=True)
            sys.stdin.readline()
        """)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "acquired"
        assert InstanceLock(name=name).acquire() is False  # a second Garvis would be refused
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(10)
    again = InstanceLock(name=name)
    assert again.acquire() is True  # released when the first process exited: no stale lock
    again.release()


# ---------------------------------------------------------- global hotkey
def test_hotkey_spec_parsing():
    modifiers, vk = parse_global_hotkey("ctrl+alt+g")
    assert modifiers == MOD_CONTROL | MOD_ALT and vk == ord("G")
    for bad in ("g", "ctrl+alt", "ctrl+alt+g+h", ""):
        with pytest.raises(ValueError):
            parse_global_hotkey(bad)


def test_registration_success_activation_and_cleanup():
    activations = []
    api = FakeHotkeyApi()
    hotkey = GlobalHotkey("ctrl+alt+g", lambda: activations.append(1), activate_message=0xC123, api=api)
    status = hotkey.register(555)
    assert status.registered and api.registered == [(555, HOTKEY_ID, MOD_CONTROL | MOD_ALT, ord("G"))]
    assert hotkey.register(555).registered and len(api.registered) == 1  # no duplicate registration
    assert hotkey.handle_message(WM_HOTKEY, HOTKEY_ID) and activations == [1]
    assert not hotkey.handle_message(WM_HOTKEY, 999)                     # someone else's hotkey
    assert hotkey.handle_message(0xC123, 0) and activations == [1, 1]   # a second launch said "activate"
    hotkey.unregister()
    hotkey.unregister()  # idempotent
    assert api.unregistered == [(555, HOTKEY_ID)]
    assert not hotkey.handle_message(WM_HOTKEY, HOTKEY_ID)               # nothing after shutdown


def test_conflicting_registration_is_explained():
    hotkey = GlobalHotkey("ctrl+alt+g", lambda: None, api=FakeHotkeyApi(error=ERROR_HOTKEY_ALREADY_REGISTERED))
    status = hotkey.register(555)
    assert not status.registered and not hotkey.is_registered
    assert "already taken by another program" in status.message and "GARVIS_GLOBAL_HOTKEY" in status.message
    other = GlobalHotkey("ctrl+alt+g", lambda: None, api=FakeHotkeyApi(error=5)).register(1)
    assert "error 5" in other.message


@pytest.mark.skipif(sys.platform != "win32", reason="Windows RegisterHotKey")
def test_real_registerhotkey_detects_conflicts_and_cleans_up():
    """Ctrl+Alt+Shift+F23 is a combination no keyboard produces, so this
    can't steal a key the user relies on. hwnd 0 = the calling thread."""
    first = GlobalHotkey("ctrl+alt+shift+f23", lambda: None)
    second = GlobalHotkey("ctrl+alt+shift+f23", lambda: None)
    assert first.register(0).registered
    try:
        status = second.register(0)
        assert not status.registered and "already taken" in status.message
    finally:
        first.unregister()
    third = GlobalHotkey("ctrl+alt+shift+f23", lambda: None)
    assert third.register(0).registered  # freed by unregister: no orphaned registration
    third.unregister()


# ------------------------------------------------------- main window
@pytest.fixture
def window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import app.ui.main_window as main_window
    from tests.test_main_window_mobile import _FakeBridge

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(main_window, "AgentBridge", _FakeBridge)
    win = main_window.MainWindow()
    yield win, main_window
    win.close()
    win.deleteLater()


def test_main_window_reports_a_hotkey_conflict_once(window, monkeypatch, tmp_path):
    win, main_window = window
    monkeypatch.setattr("app.ui.global_hotkey.HotkeyApi", lambda: FakeHotkeyApi(error=ERROR_HOTKEY_ALREADY_REGISTERED))
    monkeypatch.setattr(main_window, "_start_menu_shortcut", lambda: tmp_path / "missing.lnk")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    hotkey = win.install_global_hotkey(app, "ctrl+alt+g")
    try:
        assert "already taken by another program" in win._conversation_view.toPlainText()
    finally:
        hotkey.unregister()
        app.removeNativeEventFilter(hotkey)


def test_activation_restores_a_minimized_window(window, monkeypatch):
    win, _ = window
    monkeypatch.setattr("ctypes.windll.user32.SetForegroundWindow", lambda hwnd: 1, raising=False)
    win.show()
    win.showMinimized()
    win.bring_to_front()
    assert not win.isMinimized() and win.isVisible()
    win.hide()
    win.bring_to_front()
    assert win.isVisible()
