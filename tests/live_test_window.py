"""A plain Win32 window for the live desktop-routing test
(tests/test_desktop_routing_live.py): `pythonw live_test_window.py <title>`.

It is shown with SW_SHOWDEFAULT, i.e. exactly as the launcher's STARTUPINFO
asks (minimized + not activated for the workspace), like a well-behaved
application. It does nothing, owns no data, and closes on WM_CLOSE or by
itself after 90 seconds, so a failed test can't leave it behind.
"""

import ctypes
import sys
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WM_DESTROY, WM_TIMER = 0x0002, 0x0113
WS_OVERLAPPEDWINDOW = 0x00CF0000
SW_SHOWDEFAULT = 10


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND


@WNDPROC
def window_proc(hwnd, msg, wparam, lparam):
    if msg == WM_TIMER:
        user32.DestroyWindow(hwnd)
        return 0
    if msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "Garvis routing test"
    instance = kernel32.GetModuleHandleW(None)
    wc = WNDCLASSW(lpfnWndProc=window_proc, hInstance=instance, lpszClassName="GarvisRoutingTestWindow")
    user32.RegisterClassW(ctypes.byref(wc))
    hwnd = user32.CreateWindowExW(
        0, wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, 120, 120, 360, 200, None, None, instance, None
    )
    user32.ShowWindow(hwnd, SW_SHOWDEFAULT)
    user32.SetTimer(hwnd, 1, 90_000, None)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
