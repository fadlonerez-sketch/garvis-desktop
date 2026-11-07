"""open_path: show an existing folder in File Explorer, or open a document
with the app Windows associates with it.

Deliberately not a way to run things: executables, scripts, installers,
shortcuts and other launchable types are refused outright (applications go
through open_application, which resolves only installed apps), as are
credential files and network (UNC) paths. Nothing is created, changed or
deleted.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.logging import get_logger, log_extra
from app.security.sensitive import sensitive_path_reason
from app.tools.base import RiskLevel, Tool, ToolResult

logger = get_logger("tools.open_path")

BLOCKED_SUFFIXES = frozenset(
    ".exe .com .bat .cmd .ps1 .psm1 .psd1 .ps1xml .psc1 .vbs .vbe .js .jse .wsf .wsh .wsc .msi .msp .mst "
    ".scr .pif .lnk .url .reg .hta .cpl .jar .py .pyw .pyc .application .appref-ms .msc .scf .inf .sys "
    ".dll .ocx .gadget .appx .msix .appxbundle .msixbundle .diagcab .settingcontent-ms .library-ms "
    ".search-ms .xll .iqy .slk .chm .ins .isp .mde .mdb .accde .vb .sct .shb .shs .ws .cer "
    ".sh .msu .appinstaller .wsb .rdp .iso .img .vhd .vhdx .theme .themepack .deskthemepack "
    ".docm .dotm .xlsm .xltm .xlam .pptm .potm .ppam".split()
)
"""Types that execute or configure something when "opened" -- never opened here."""


def expand(raw: str) -> Path:
    text = os.path.expandvars(os.path.expanduser(raw.strip().strip('"')))
    if text.startswith(("\\\\", "//")):
        return Path(text)  # not resolved: resolving a UNC path contacts the server; refusal() rejects it
    return Path(text).resolve(strict=False)


def refusal(path: Path) -> str | None:
    """Why `path` must not be opened, or None if it may be."""
    text = str(path)
    if text.startswith("\\\\"):
        return "Network (UNC) paths are not opened."
    reason = sensitive_path_reason(path)
    if reason:
        return reason
    if not path.exists():
        return f"Nothing exists at '{text}'."
    if path.is_file() and path.suffix.lower() in BLOCKED_SUFFIXES:
        return (
            f"'{path.name}' is a {path.suffix.lower()} file, which runs or changes something when opened; "
            "Garvis only opens folders and documents. Use open_application for installed apps."
        )
    if not (path.is_file() or path.is_dir()):
        return f"'{text}' is neither a regular file nor a folder."
    return None


class OpenPathInput(BaseModel):
    path: str = Field(
        description="Absolute path of an existing folder or document, e.g. "
        "'C:\\\\Users\\\\me\\\\Projects\\\\garvis' or '%USERPROFILE%\\\\Documents\\\\notes.txt'."
    )


class OpenPathTool(Tool):
    name = "open_path"
    description = (
        "Open an existing folder in File Explorer, or a document (PDF, text, image, Office file...) in the app "
        "Windows uses for it, on the user's screen. Does not run programs or scripts -- use open_application "
        "for apps."
    )
    risk_level = RiskLevel.LOW
    presentation = "prominent"  # the user asked to see it
    input_model = OpenPathInput

    def __init__(self, opener=None) -> None:
        self._opener = opener or (os.startfile if sys.platform == "win32" else None)

    async def execute(self, args: OpenPathInput) -> ToolResult:
        if self._opener is None:
            return ToolResult.fail("Opening files and folders is only supported on Windows.")
        path = expand(args.path)
        reason = refusal(path)
        if reason:
            return ToolResult.fail(reason)
        try:
            await asyncio.to_thread(self._opener, str(path))
        except OSError as exc:
            log_extra(logger, 30, "open_path.failed", error=str(exc))
            return ToolResult.fail(f"Windows could not open '{path}': {exc}")
        kind = "folder" if path.is_dir() else "file"
        log_extra(logger, 20, "open_path.opened", kind=kind)
        return ToolResult.ok({"opened": str(path), "path": str(path), "name": path.name or str(path), "kind": kind})
