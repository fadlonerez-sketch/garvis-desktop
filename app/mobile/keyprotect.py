"""Windows DPAPI for the mobile server's private keys.

`protect()` / `unprotect()` wrap CryptProtectData / CryptUnprotectData for
the CURRENT Windows user (no UI, never machine-wide), with Garvis-specific
entropy so another program running as the user can't decrypt the blob by
accident. Only that Windows account on this PC can open the result: a copy
of the file on another machine, in a backup or under another account is
useless.

Not Windows -> `available()` is False and callers keep their old behaviour.
"""

from __future__ import annotations

import ctypes
import sys

_ENTROPY = b"garvis/mobile/tls-keys/v1"
_UI_FORBIDDEN = 0x1


class KeyProtectionError(Exception):
    """DPAPI refused (another Windows account, a reset profile, a damaged blob)."""


def available() -> bool:
    return sys.platform == "win32"


if sys.platform == "win32":
    from ctypes import wintypes

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _BLOB_P = ctypes.POINTER(_Blob)
    _crypt32.CryptProtectData.argtypes = [
        _BLOB_P,
        wintypes.LPCWSTR,
        _BLOB_P,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        _BLOB_P,
    ]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        _BLOB_P,
        ctypes.c_void_p,
        _BLOB_P,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        _BLOB_P,
    ]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    def _blob(data: bytes) -> tuple[_Blob, ctypes.Array]:
        buffer = ctypes.create_string_buffer(data, len(data))
        return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer

    def _run(protecting: bool, data: bytes) -> bytes:
        source, _keep_source = _blob(data)
        entropy, _keep_entropy = _blob(_ENTROPY)
        result = _Blob()
        if protecting:
            ok = _crypt32.CryptProtectData(
                ctypes.byref(source),
                "Garvis mobile TLS keys",
                ctypes.byref(entropy),
                None,
                None,
                _UI_FORBIDDEN,
                ctypes.byref(result),
            )
        else:
            ok = _crypt32.CryptUnprotectData(
                ctypes.byref(source), None, ctypes.byref(entropy), None, None, _UI_FORBIDDEN, ctypes.byref(result)
            )
        if not ok:
            raise KeyProtectionError(f"DPAPI error {ctypes.get_last_error()}")
        try:
            return ctypes.string_at(result.pbData, result.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(result.pbData, ctypes.c_void_p))


def protect(data: bytes) -> bytes:
    if not available():
        raise KeyProtectionError("DPAPI is only available on Windows.")
    return _run(True, data)


def unprotect(blob: bytes) -> bytes:
    if not available():
        raise KeyProtectionError("DPAPI is only available on Windows.")
    return _run(False, blob)
