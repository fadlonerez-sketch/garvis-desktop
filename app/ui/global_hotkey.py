"""System-wide hotkey (default Ctrl+Alt+G) that brings Garvis to the front.

Uses the native Windows API: RegisterHotKey on the main window's handle, so
Windows posts WM_HOTKEY to that window whatever app is in the foreground.
The message is picked up by a QAbstractNativeEventFilter on Qt's own event
loop -- nothing polls the keyboard. The same filter handles the "activate"
message a second launch posts (app/ui/single_instance.py). The hotkey is
unregistered on shutdown; Windows also drops it when the process ends.

Starting Garvis when it ISN'T running can't be done by a running process,
so that half is the Start-menu shortcut created by
scripts/install_shortcut.py, whose "Shortcut key" Explorer handles: it
starts Garvis, and if Garvis is already running that launch just forwards
"activate" to it.
"""
from __future__ import annotations

import ctypes
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

from PySide6.QtCore import QAbstractNativeEventFilter

from app.desktop.keys import key_to_vk

WM_HOTKEY = 0x0312
HOTKEY_ID = 0xB0A1
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}


def parse_global_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+g' -> (modifier flags, virtual-key code). Needs at least one
    modifier and exactly one non-modifier key."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    modifiers = 0
    keys = []
    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        else:
            keys.append(part)
    if not modifiers or len(keys) != 1:
        raise ValueError(f"'{spec}' isn't a usable global hotkey: use modifiers plus one key, e.g. ctrl+alt+g.")
    return modifiers, key_to_vk(keys[0])


class HotkeyApi:
    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]

    def register(self, hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> int:
        """0 on success, else the Windows error code."""
        if self._user32.RegisterHotKey(hwnd, hotkey_id, modifiers | MOD_NOREPEAT, vk):
            return 0
        return ctypes.get_last_error() or -1

    def unregister(self, hwnd: int, hotkey_id: int) -> None:
        self._user32.UnregisterHotKey(hwnd, hotkey_id)


@dataclass(frozen=True)
class HotkeyStatus:
    registered: bool
    message: str


class GlobalHotkey(QAbstractNativeEventFilter):
    def __init__(self, spec: str, on_activate: Callable[[], None], activate_message: int = 0,
                 api: HotkeyApi | None = None) -> None:
        super().__init__()
        self._spec = spec
        self._on_activate = on_activate
        self._activate_message = activate_message
        self._api = api or HotkeyApi()
        self._hwnd: int | None = None

    @property
    def is_registered(self) -> bool:
        return self._hwnd is not None

    def register(self, hwnd: int) -> HotkeyStatus:
        if self._hwnd is not None:
            return HotkeyStatus(True, f"{self._spec} is already active.")  # idempotent
        try:
            modifiers, vk = parse_global_hotkey(self._spec)
        except ValueError as exc:
            return HotkeyStatus(False, str(exc))
        error = self._api.register(hwnd, HOTKEY_ID, modifiers, vk)
        if error == 0:
            self._hwnd = hwnd
            return HotkeyStatus(True, f"{self._spec.upper()} brings Garvis to the front.")
        if error == ERROR_HOTKEY_ALREADY_REGISTERED:
            return HotkeyStatus(
                False,
                f"{self._spec.upper()} is already taken by another program, so Garvis couldn't register it. "
                "Close that program or remove its shortcut, or choose another key with GARVIS_GLOBAL_HOTKEY "
                "(e.g. ctrl+alt+j). A Start-menu shortcut with the same key also counts: scripts/install_shortcut.py "
                "--remove clears Garvis's own.",
            )
        return HotkeyStatus(False, f"Windows refused to register {self._spec.upper()} (error {error}).")

    def unregister(self) -> None:
        if self._hwnd is not None:
            self._api.unregister(self._hwnd, HOTKEY_ID)
            self._hwnd = None

    def handle_message(self, message: int, wparam: int) -> bool:
        if message == WM_HOTKEY and wparam == HOTKEY_ID and self._hwnd is not None:
            self._on_activate()
            return True
        if self._activate_message and message == self._activate_message:
            self._on_activate()
            return True
        return False

    def nativeEventFilter(self, event_type, message):  # noqa: N802 (Qt override)
        if bytes(event_type) not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            return False, 0
        msg = wintypes.MSG.from_address(int(message))
        return self.handle_message(int(msg.message), int(msg.wParam or 0)), 0
