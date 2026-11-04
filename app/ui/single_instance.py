"""One Garvis per Windows session, and "launch again" = "show the running one".

Deliberately ctypes-only (no Qt, no app imports): it runs first thing in
main(), so a second launch -- e.g. Ctrl+Alt+G through the Start-menu
shortcut while Garvis is already running -- hands over and exits in
milliseconds instead of loading the whole app.

- A named mutex ("Local\\Garvis.SingleInstance", i.e. per logon session)
  says whether an instance exists. Windows releases it when that process
  exits for any reason, so a crash never leaves a stale lock behind.
- The running instance tags its main window with a window property; the
  new launch finds it (even hidden or minimized), grants it the right to
  take the foreground (AllowSetForegroundWindow -- the launch came from the
  user's key press, so it may pass that right on) and posts a registered
  "activate" message. The main window restores and focuses itself.
- If that window is hung, or never appears, the launch reports it instead
  of starting a second Garvis.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

MUTEX_NAME = "Local\\Garvis.SingleInstance"
WINDOW_PROPERTY = "Garvis.MainWindow"
ACTIVATE_MESSAGE_NAME = "Garvis.ActivateMainWindow"
ERROR_ALREADY_EXISTS = 183

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class Win32Api:
    """The handful of user32/kernel32 calls involved (a seam for tests)."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u, k = self.user32, self.kernel32
        k.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k.CreateMutexW.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        u.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
        u.RegisterWindowMessageW.restype = wintypes.UINT
        u.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
        u.GetPropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        u.GetPropW.restype = wintypes.HANDLE
        u.SetPropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.HANDLE]
        u.RemovePropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.IsHungAppWindow.argtypes = [wintypes.HWND]
        u.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
        u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    def create_mutex(self, name: str) -> tuple[int, bool]:
        """(handle, already_existed)."""
        handle = self.kernel32.CreateMutexW(None, False, name)
        return int(handle or 0), ctypes.get_last_error() == ERROR_ALREADY_EXISTS

    def close_handle(self, handle: int) -> None:
        self.kernel32.CloseHandle(handle)

    def activate_message(self) -> int:
        return int(self.user32.RegisterWindowMessageW(ACTIVATE_MESSAGE_NAME))

    def tagged_windows(self) -> list[tuple[int, int]]:
        """(hwnd, pid) of every top-level window carrying WINDOW_PROPERTY."""
        found: list[tuple[int, int]] = []

        @_WNDENUMPROC
        def collect(hwnd, _lparam):
            if self.user32.GetPropW(hwnd, WINDOW_PROPERTY):
                pid = wintypes.DWORD(0)
                self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                found.append((int(hwnd), int(pid.value)))
            return True

        self.user32.EnumWindows(collect, 0)
        return found

    def tag_window(self, hwnd: int) -> None:
        self.user32.SetPropW(hwnd, WINDOW_PROPERTY, wintypes.HANDLE(1))

    def untag_window(self, hwnd: int) -> None:
        self.user32.RemovePropW(hwnd, WINDOW_PROPERTY)

    def is_hung(self, hwnd: int) -> bool:
        return bool(self.user32.IsHungAppWindow(hwnd))

    def allow_foreground(self, pid: int) -> None:
        self.user32.AllowSetForegroundWindow(pid)

    def post(self, hwnd: int, message: int) -> bool:
        return bool(self.user32.PostMessageW(hwnd, message, 0, 0))


@dataclass(frozen=True)
class ForwardResult:
    status: str
    """"activated", "not_responding" or "no_window"."""
    pid: int | None = None

    @property
    def message(self) -> str:
        if self.status == "activated":
            return "Garvis is already running; brought its window to the front."
        if self.status == "not_responding":
            return (f"Garvis is already running (process {self.pid}) but its window is not responding. "
                    "Wait a moment, or end that process in Task Manager and start Garvis again.")
        return ("Garvis is already running but its window couldn't be found (it may still be starting). "
                "Try again in a few seconds.")


class InstanceLock:
    def __init__(self, api: Win32Api | None = None, name: str = MUTEX_NAME) -> None:
        self._api = api or Win32Api()
        self._name = name
        self._handle = 0

    def acquire(self) -> bool:
        """True if this is the only Garvis in the session (keep the lock for
        the process lifetime); False if another instance holds it."""
        handle, existed = self._api.create_mutex(self._name)
        if existed:
            if handle:
                self._api.close_handle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle:
            self._api.close_handle(self._handle)
            self._handle = 0


def forward_activation(api: Win32Api | None = None, wait_seconds: float = 10.0, poll_seconds: float = 0.25,
                       clock=time.monotonic, sleep=time.sleep) -> ForwardResult:
    """Ask the running instance to show its window."""
    api = api or Win32Api()
    deadline = clock() + wait_seconds
    while True:
        windows = api.tagged_windows()
        if windows:
            hwnd, pid = windows[0]
            if api.is_hung(hwnd):
                return ForwardResult("not_responding", pid)
            api.allow_foreground(pid)
            if api.post(hwnd, api.activate_message()):
                return ForwardResult("activated", pid)
        if clock() >= deadline:
            return ForwardResult("no_window")
        sleep(poll_seconds)  # the other instance may still be starting up
