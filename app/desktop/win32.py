"""Win32 implementation of DesktopBackend, via ctypes only (no pywin32).

This is the ONLY module that calls user32/gdi32/dwmapi/kernel32 -- the same
isolation rule app/browser/manager.py follows for Playwright. It is imported
lazily by app/desktop/build.py, so importing the rest of Garvis on a
non-Windows machine (or in tests) never loads Windows DLLs.

Input is injected with SendInput (the supported API; mouse_event/
keybd_event are legacy wrappers of it). Text is typed as Unicode scan
events (KEYEVENTF_UNICODE), so what the model asks to type is exactly what
arrives, independent of the active keyboard layout (English, Hebrew, ...).
Screenshots are GDI BitBlt + a small stdlib PNG encoder -- no Pillow.
"""
from __future__ import annotations

import ctypes
import os
import struct
import time
import zlib
from ctypes import wintypes
from pathlib import Path

from app.desktop.backend import DesktopBackend, InputContext, MouseButton, Rect, WindowInfo, WindowState
from app.desktop.keys import EXTENDED_KEYS

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi")

ULONG_PTR = ctypes.c_size_t

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
WHEEL_DELTA = 120

VK_RETURN = 0x0D
VK_TAB = 0x09
VK_MENU = 0x12

SM_CXSCREEN, SM_CYSCREEN = 0, 1
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
SW_RESTORE = 9
SW_SHOWNOACTIVATE = 4
SW_MAXIMIZE = 3
SW_MINIMIZE = 6
WM_CLOSE = 0x0010
DESKTOP_READOBJECTS = 0x0001
UOI_NAME = 2
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0

_BUTTON_FLAGS = {
    MouseButton.LEFT: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    MouseButton.RIGHT: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    MouseButton.MIDDLE: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}
_IGNORED_WINDOW_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _declare() -> None:
    """argtypes/restype for every handle-returning call -- without these,
    ctypes defaults to a 32-bit int return and silently truncates 64-bit
    HWND/HDC/HBITMAP values."""
    u, g, k = _user32, _gdi32, _kernel32
    u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    u.SendInput.restype = wintypes.UINT
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.IsWindow.argtypes = [wintypes.HWND]
    u.IsIconic.argtypes = [wintypes.HWND]
    u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.GetWindowThreadProcessId.restype = wintypes.DWORD
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    u.OpenInputDesktop.restype = wintypes.HANDLE
    u.CloseDesktop.argtypes = [wintypes.HANDLE]
    u.GetThreadDesktop.argtypes = [wintypes.DWORD]
    u.GetThreadDesktop.restype = wintypes.HANDLE
    u.GetUserObjectInformationW.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    k.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    k.WTSGetActiveConsoleSessionId.restype = wintypes.DWORD
    u.BringWindowToTop.argtypes = [wintypes.HWND]
    u.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    u.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.UINT,
    ]
    u.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    u.GetDC.argtypes = [wintypes.HWND]
    u.GetDC.restype = wintypes.HDC
    u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    g.CreateCompatibleDC.argtypes = [wintypes.HDC]
    g.CreateCompatibleDC.restype = wintypes.HDC
    g.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    g.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    g.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    g.SelectObject.restype = wintypes.HGDIOBJ
    g.BitBlt.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
    ]
    g.GetDIBits.argtypes = [
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
        ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT,
    ]
    g.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    g.DeleteDC.argtypes = [wintypes.HDC]
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.GetCurrentThreadId.restype = wintypes.DWORD
    _dwmapi.DwmGetWindowAttribute.argtypes = [
        wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
    ]


_declare()


def _enable_dpi_awareness() -> None:
    """Make coordinates physical pixels. Qt already sets per-monitor-v2
    awareness for the GUI process, in which case this call simply fails
    (awareness can only be set once) -- harmless either way."""
    try:
        _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except (AttributeError, OSError):
        pass


def _encode_png(width: int, height: int, bgra: bytes) -> bytes:
    rgb = bytearray(width * height * 3)
    rgb[0::3] = bgra[2::4]
    rgb[1::3] = bgra[1::4]
    rgb[2::3] = bgra[0::4]
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(rgb[y * stride:(y + 1) * stride]) for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class Win32DesktopBackend(DesktopBackend):
    def __init__(self) -> None:
        _enable_dpi_awareness()

    # ------------------------------------------------------------ input
    @staticmethod
    def _send(inputs: list[INPUT]) -> None:
        array = (INPUT * len(inputs))(*inputs)
        sent = _user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
        if sent != len(inputs):
            # Typically UIPI: the foreground window belongs to an elevated
            # process, which a non-elevated Garvis may not inject into.
            raise OSError(
                f"SendInput injected {sent}/{len(inputs)} events "
                f"(error {ctypes.get_last_error()}); the target may be an elevated/admin window."
            )

    @staticmethod
    def _key_input(vk: int, up: bool) -> INPUT:
        flags = KEYEVENTF_KEYUP if up else 0
        if vk in EXTENDED_KEYS:
            flags |= KEYEVENTF_EXTENDEDKEY
        return INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(wVk=vk, dwFlags=flags)))

    @staticmethod
    def _mouse_input(flags: int, data: int = 0) -> INPUT:
        return INPUT(
            type=INPUT_MOUSE,
            u=_INPUTUNION(mi=MOUSEINPUT(mouseData=ctypes.c_ulong(data & 0xFFFFFFFF).value, dwFlags=flags)),
        )

    def move_mouse(self, x: int, y: int) -> None:
        if not _user32.SetCursorPos(int(x), int(y)):
            raise OSError(f"SetCursorPos failed (error {ctypes.get_last_error()}).")

    def click(self, button: MouseButton, count: int) -> None:
        down, up = _BUTTON_FLAGS[button]
        events: list[INPUT] = []
        for _ in range(count):
            events += [self._mouse_input(down), self._mouse_input(up)]
        self._send(events)

    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        flag = MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL
        self._send([self._mouse_input(flag, clicks * WHEEL_DELTA)])

    def type_text(self, text: str) -> None:
        events: list[INPUT] = []
        for char in text.replace("\r\n", "\n"):
            if char == "\n":
                events += [self._key_input(VK_RETURN, False), self._key_input(VK_RETURN, True)]
                continue
            if char == "\t":
                events += [self._key_input(VK_TAB, False), self._key_input(VK_TAB, True)]
                continue
            encoded = char.encode("utf-16-le")
            for (unit,) in struct.iter_unpack("<H", encoded):  # surrogate pairs -> 2 units
                for up in (False, True):
                    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
                    events.append(
                        INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(wScan=unit, dwFlags=flags)))
                    )
        # Chunked so a long string doesn't overflow the target's input queue.
        for start in range(0, len(events), 200):
            self._send(events[start:start + 200])
            time.sleep(0.01)

    def press_keys(self, vks: list[int], repeat: int = 1) -> None:
        for i in range(repeat):
            events = [self._key_input(vk, False) for vk in vks]
            events += [self._key_input(vk, True) for vk in reversed(vks)]
            self._send(events)
            if i + 1 < repeat:
                time.sleep(0.03)

    # ---------------------------------------------------------- screen
    def virtual_screen(self) -> Rect:
        left = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        top = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        return Rect(
            left=left,
            top=top,
            right=left + _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            bottom=top + _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    def primary_screen_size(self) -> tuple[int, int]:
        return _user32.GetSystemMetrics(SM_CXSCREEN), _user32.GetSystemMetrics(SM_CYSCREEN)

    def cursor_position(self) -> tuple[int, int]:
        point = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def capture_screen(self, path: Path, region: Rect | None = None) -> tuple[int, int]:
        area = region or self.virtual_screen()
        width, height = area.width, area.height
        if width <= 0 or height <= 0:
            raise OSError("Capture region is empty.")
        screen_dc = _user32.GetDC(None)
        mem_dc = _gdi32.CreateCompatibleDC(screen_dc)
        bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        old = _gdi32.SelectObject(mem_dc, bitmap)
        try:
            if not _gdi32.BitBlt(
                mem_dc, 0, 0, width, height, screen_dc, area.left, area.top, SRCCOPY | CAPTUREBLT
            ):
                raise OSError(f"BitBlt failed (error {ctypes.get_last_error()}).")
            header = BITMAPINFOHEADER(
                biSize=ctypes.sizeof(BITMAPINFOHEADER),
                biWidth=width,
                biHeight=-height,  # negative = top-down rows
                biPlanes=1,
                biBitCount=32,
            )
            buffer = ctypes.create_string_buffer(width * height * 4)
            if not _gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, ctypes.byref(header), DIB_RGB_COLORS):
                raise OSError("GetDIBits failed.")
        finally:
            _gdi32.SelectObject(mem_dc, old)
            _gdi32.DeleteObject(bitmap)
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(None, screen_dc)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_encode_png(width, height, buffer.raw))
        return width, height

    # --------------------------------------------------------- windows
    @staticmethod
    def _title(hwnd: int) -> str:
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    @staticmethod
    def _class_name(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _is_cloaked(hwnd: int) -> bool:
        """Cloaked windows exist but aren't shown -- chiefly windows on a
        different virtual desktop, plus suspended UWP frames."""
        cloaked = wintypes.DWORD(0)
        result = _dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        )
        return result == 0 and cloaked.value != 0

    @staticmethod
    def _process(hwnd: int) -> tuple[int, str]:
        pid = wintypes.DWORD(0)
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = ""
        handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if handle:
            try:
                size = wintypes.DWORD(1024)
                buf = ctypes.create_unicode_buffer(size.value)
                if _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    name = os.path.basename(buf.value)
            finally:
                _kernel32.CloseHandle(handle)
        return pid.value, name

    @staticmethod
    def _rect(hwnd: int) -> Rect | None:
        rect = wintypes.RECT()
        if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return Rect(left=rect.left, top=rect.top, right=rect.right, bottom=rect.bottom)

    def _info(self, hwnd: int, foreground: int | None) -> WindowInfo:
        pid, process_name = self._process(hwnd)
        return WindowInfo(
            handle=int(hwnd),
            title=self._title(hwnd),
            process_name=process_name,
            pid=pid,
            rect=self._rect(hwnd),
            is_minimized=bool(_user32.IsIconic(hwnd)),
            is_foreground=foreground is not None and int(hwnd) == int(foreground),
            on_current_desktop=not self._is_cloaked(hwnd),
        )

    @staticmethod
    def _desktop_name(handle) -> str | None:
        buf = ctypes.create_unicode_buffer(256)
        needed = wintypes.DWORD(0)
        if not _user32.GetUserObjectInformationW(handle, UOI_NAME, buf, ctypes.sizeof(buf), ctypes.byref(needed)):
            return None
        return buf.value

    def input_context(self) -> InputContext:
        """Real input only reaches the user's apps when this process runs in
        the active console session and the desktop receiving input is the
        normal one ("Default") -- not the lock screen or a UAC prompt
        (secure "Winlogon" desktop), which a normal process can't even open."""
        session = wintypes.DWORD(0)
        _kernel32.ProcessIdToSessionId(_kernel32.GetCurrentProcessId(), ctypes.byref(session))
        console = int(_kernel32.WTSGetActiveConsoleSessionId())
        own_desktop = self._desktop_name(_user32.GetThreadDesktop(_kernel32.GetCurrentThreadId()))
        input_desktop = None
        handle = _user32.OpenInputDesktop(0, False, DESKTOP_READOBJECTS)
        if handle:
            try:
                input_desktop = self._desktop_name(handle)
            finally:
                _user32.CloseDesktop(handle)
        reason = None
        if console != 0xFFFFFFFF and session.value != console:
            reason = (f"Garvis runs in Windows session {session.value}, but the screen in front of you is session "
                      f"{console} (e.g. Garvis was started over Remote Desktop or as another user). Start Garvis "
                      "from your own desktop session.")
        elif input_desktop is None:
            reason = ("Windows is showing the lock screen or a secure prompt (UAC / Ctrl+Alt+Del), which normal "
                      "apps can't control. Unlock or answer it, then try again.")
        elif own_desktop and input_desktop.lower() != own_desktop.lower():
            reason = (f"Input currently goes to the '{input_desktop}' desktop, not Garvis's '{own_desktop}' desktop.")
        return InputContext(
            session_id=session.value, console_session_id=console, input_desktop=input_desktop,
            interactive=reason is None, reason=reason,
        )

    def list_windows(self, include_other_desktops: bool = False) -> list[WindowInfo]:
        foreground = _user32.GetForegroundWindow()
        handles: list[int] = []

        @_WNDENUMPROC
        def _collect(hwnd, _lparam):
            if (
                _user32.IsWindowVisible(hwnd)
                and _user32.GetWindowTextLengthW(hwnd) > 0
                and not (_user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW)
                and self._class_name(hwnd) not in _IGNORED_WINDOW_CLASSES
                and (include_other_desktops or not self._is_cloaked(hwnd))
            ):
                handles.append(hwnd)
            return True

        _user32.EnumWindows(_collect, 0)
        return [self._info(h, foreground) for h in handles]

    def foreground_window(self) -> WindowInfo | None:
        hwnd = _user32.GetForegroundWindow()
        return self._info(hwnd, hwnd) if hwnd else None

    def window_info(self, handle: int) -> WindowInfo | None:
        if not _user32.IsWindow(handle):
            return None
        return self._info(handle, _user32.GetForegroundWindow())

    def focus_window(self, handle: int) -> bool:
        if not _user32.IsWindow(handle):
            return False
        if _user32.IsIconic(handle):
            _user32.ShowWindow(handle, SW_RESTORE)

        # Windows only lets the foreground process move the foreground.
        # Attaching our input queue to the current foreground thread's
        # satisfies that rule without synthesizing any keystroke.
        current = _user32.GetForegroundWindow()
        our_thread = _kernel32.GetCurrentThreadId()
        fg_thread = _user32.GetWindowThreadProcessId(current, None) if current else 0
        attached = bool(fg_thread and fg_thread != our_thread and
                        _user32.AttachThreadInput(our_thread, fg_thread, True))
        try:
            _user32.BringWindowToTop(handle)
            _user32.SetForegroundWindow(handle)
        finally:
            if attached:
                _user32.AttachThreadInput(our_thread, fg_thread, False)

        if int(_user32.GetForegroundWindow() or 0) != int(handle):
            # Fallback: an injected (harmless) ALT tap makes us the process
            # that "received the last input event", which also unlocks it.
            self._send([self._key_input(VK_MENU, False), self._key_input(VK_MENU, True)])
            _user32.SetForegroundWindow(handle)
        time.sleep(0.05)
        return int(_user32.GetForegroundWindow() or 0) == int(handle)

    def set_window_state(self, handle: int, state: WindowState) -> None:
        if not _user32.IsWindow(handle):
            raise OSError("That window no longer exists.")
        command = {WindowState.MINIMIZE: SW_MINIMIZE, WindowState.MAXIMIZE: SW_MAXIMIZE,
                   WindowState.SHOW_INACTIVE: SW_SHOWNOACTIVATE}.get(state, SW_RESTORE)
        _user32.ShowWindow(handle, command)

    def close_window(self, handle: int) -> None:
        if not _user32.IsWindow(handle):
            raise OSError("That window no longer exists.")
        if not _user32.PostMessageW(handle, WM_CLOSE, 0, 0):
            raise OSError(f"Windows refused to send the close request (error {ctypes.get_last_error()}).")

    def set_window_rect(self, handle: int, rect: Rect) -> None:
        if not _user32.SetWindowPos(
            handle, None, rect.left, rect.top, rect.width, rect.height, SWP_NOZORDER | SWP_NOACTIVATE
        ):
            raise OSError(f"SetWindowPos failed (error {ctypes.get_last_error()}).")
