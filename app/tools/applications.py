"""Open an installed Windows application by friendly name.

The LLM can never supply an arbitrary command line -- it only supplies a
friendly name (e.g. "whatsapp", "notepad"), which this tool resolves to a
concrete, launchable target through a fixed resolution order (below), then
launches via `subprocess.Popen`/`os.startfile` with no shell interpretation
and no string concatenation, so there is no command-injection surface.

Resolution order (first match wins):
  1. config/applications.json -- a curated allowlist/alias map. Checked
     first so an explicit override (a specific path, or a friendly name
     that would otherwise be ambiguous) always wins.
  2. Windows 'App Paths' registry (HKCU then HKLM) -- most installers
     (Chrome, Spotify, Discord, WhatsApp, ...) register their real
     executable path here under App Paths\\<name>.exe, keyed by the exe's
     own filename. This is the same mechanism Win+R relies on to resolve a
     bare app name.
  3. Start Menu shortcuts (the all-users ProgramData and current user's
     AppData Start Menu folders, searched recursively for *.lnk). Returns
     the shortcut itself, launched via os.startfile() -- which resolves a
     .lnk exactly like Explorer double-clicking it would -- rather than
     parsing the binary shortcut format ourselves.
  4. shutil.which() -- a plain PATH lookup, for console-registered tools
     (e.g. a JetBrains launcher script) not covered by the above.
  5. The Start menu's "All apps" list (the shell AppsFolder, via
     Get-StartApps) -- the only place Store/MSIX-packaged apps appear.
     WhatsApp Desktop (package 5319275A.WhatsAppDesktop) is the motivating
     case: it has no App Paths entry, no Start Menu .lnk and nothing on
     PATH, so stages 1-4 all miss it. Every entry there has an
     AppUserModelID, launched as `shell:AppsFolder\\<AUMID>` through
     os.startfile() -- the same way Start itself launches it. Last in the
     order so nothing that already resolved via stages 1-4 changes.

This keeps config/applications.json as an explicit, auditable override
layer rather than the sole source of truth, so most installed apps ("Open
WhatsApp", "Open Spotify", "Open Discord") resolve without a human having
to hand-maintain a path for every app on the machine.

Target desktop (app/desktop/routing.py): resolved BEFORE launching.
"user_desktop" (the default -- opening an app is normally for the user)
puts the app's window on the desktop the user is looking at, focused and
verified. "garvis_workspace" launches it minimized without activation
(SW_SHOWMINNOACTIVE), moves it to Garvis's virtual desktop and verifies
the user's focus is unchanged; if the workspace is unavailable it is an
error and nothing is launched.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger, log_extra
from app.desktop.routing import TARGET_FIELD_DESCRIPTION, DesktopRouting, DesktopTarget
from app.tools.base import RiskLevel, Tool, ToolResult

logger = get_logger("tools.applications")

_DEFAULT_ALLOWLIST_PATH = Path(__file__).resolve().parents[2] / "config" / "applications.json"
_NORMALIZE_RE = re.compile(r"[^a-z0-9]")


def load_allowlist(path: Path = _DEFAULT_ALLOWLIST_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("", name.lower())


def _import_winreg():
    # Not a sys.platform check: attempting the import and treating
    # ImportError as "not supported here" is functionally identical on
    # real Windows (winreg always imports) and real non-Windows (winreg
    # never exists), but also lets tests substitute a fake `winreg` module
    # via sys.modules -- same pattern as app/voice/tts/player.py's winsound.
    try:
        import winreg
    except ImportError:
        return None
    return winreg


def resolve_from_app_paths_registry(name: str) -> str | None:
    """Look up `name` under the Windows 'App Paths' registry -- the same
    mechanism Win+R uses to resolve a bare app name to its real
    executable. Most installers (Chrome, Spotify, Discord, WhatsApp, ...)
    register themselves here."""
    winreg = _import_winreg()
    if winreg is None:
        return None
    candidate = name if name.lower().endswith(".exe") else f"{name}.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{candidate}"
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        if value:
            return value
    return None


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for env_var in ("ProgramData", "APPDATA"):
        base = os.environ.get(env_var)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def resolve_from_start_menu(name: str, search_dirs: list[Path] | None = None) -> str | None:
    """Search Start Menu shortcuts (recursively) for one matching `name`.
    Returns the .lnk path itself -- os.startfile() resolves and launches a
    shortcut exactly like Explorer would, so there's no need to parse the
    binary shortcut format to find its target."""
    target = _normalize(name)
    if not target:
        return None
    dirs = _start_menu_dirs() if search_dirs is None else search_dirs
    best_match: Path | None = None
    for base in dirs:
        if not base.exists():
            continue
        try:
            candidates = sorted(base.rglob("*.lnk"))
        except OSError:
            continue
        for lnk in candidates:
            stem = _normalize(lnk.stem)
            if stem == target:
                return str(lnk)
            if best_match is None and target in stem:
                best_match = lnk
    return str(best_match) if best_match else None


_START_APPS_CACHE_SECONDS = 300.0
_START_APPS_TIMEOUT_SECONDS = 15.0
_start_apps_cache: tuple[float, list[tuple[str, str]]] | None = None

# Fixed command (no user input is ever interpolated into it). UTF-8 output
# so non-English app names survive; Select-Object keeps the JSON small.
_GET_START_APPS_COMMAND = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
)


def _list_start_apps() -> list[tuple[str, str]]:
    """(display name, AppUserModelID) for every app in Start's "All apps"
    list, packaged Store/MSIX apps included. Empty on any failure."""
    if sys.platform != "win32":
        return []
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _GET_START_APPS_COMMAND],
            capture_output=True,
            timeout=_START_APPS_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_extra(logger, 30, "open_application.start_apps_unavailable", error=str(exc))
        return []
    if completed.returncode != 0:
        log_extra(logger, 30, "open_application.start_apps_failed", returncode=completed.returncode)
        return []
    try:
        data = json.loads(completed.stdout.decode("utf-8", errors="replace") or "[]")
    except ValueError:
        return []
    if isinstance(data, dict):  # ConvertTo-Json emits a bare object for one item
        data = [data]
    return [
        (str(item.get("Name") or ""), str(item.get("AppID") or ""))
        for item in data
        if isinstance(item, dict) and item.get("AppID")
    ]


def list_start_apps() -> list[tuple[str, str]]:
    """Cached _list_start_apps() -- the PowerShell call takes ~1s, and the
    installed-app set rarely changes within a few minutes."""
    global _start_apps_cache
    now = time.monotonic()
    if _start_apps_cache is not None and now - _start_apps_cache[0] < _START_APPS_CACHE_SECONDS:
        return _start_apps_cache[1]
    apps = _list_start_apps()
    if apps:  # don't cache a failure
        _start_apps_cache = (now, apps)
    return apps


def resolve_from_apps_folder(name: str, apps: list[tuple[str, str]] | None = None) -> str | None:
    """Match `name` against the Start "All apps" list and return a
    launchable `shell:AppsFolder\\<AUMID>` target. Ranking: exact display
    name > display-name substring (shortest name wins, so "WhatsApp" beats
    "WhatsApp Beta") > package-name substring (e.g. "whatsapp" in
    "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App")."""
    target = _normalize(name)
    if not target:
        return None
    entries = list_start_apps() if apps is None else apps
    substring: list[tuple[int, str]] = []
    package: list[tuple[int, str]] = []
    for display_name, app_id in entries:
        normalized_name = _normalize(display_name)
        if normalized_name == target:
            return f"shell:AppsFolder\\{app_id}"
        if target in normalized_name:
            substring.append((len(normalized_name), app_id))
            continue
        package_name = _normalize(app_id.split("!", 1)[0].split("_", 1)[0])
        if target in package_name:
            package.append((len(package_name), app_id))
    for candidates in (substring, package):
        if candidates:
            return f"shell:AppsFolder\\{min(candidates)[1]}"
    return None


def resolve_application(name: str, allowlist: dict[str, str]) -> tuple[str, bool] | None:
    """Resolve a friendly application name to a launchable target.

    Returns `(path, is_shortcut)` -- `is_shortcut` tells the caller to
    launch via os.startfile() rather than subprocess.Popen() -- or None if
    nothing matched. See module docstring for the resolution order.
    """
    key = name.strip().lower()
    if not key:
        return None
    # A friendly name, never a path: shutil.which() returns any existing
    # executable/script path as-is, which would let the model launch a file
    # it just wrote (open_path refuses programs and scripts for this reason).
    if any(sep in key for sep in ("\\", "/", ":")):
        return None
    if key in allowlist:
        return allowlist[key], False

    from_registry = resolve_from_app_paths_registry(key)
    if from_registry:
        return from_registry, False

    from_start_menu = resolve_from_start_menu(key)
    if from_start_menu:
        return from_start_menu, True

    which_result = shutil.which(key) or shutil.which(f"{key}.exe")
    if which_result:
        return which_result, False

    from_apps_folder = resolve_from_apps_folder(key)
    if from_apps_folder:
        return from_apps_folder, True

    return None


class OpenApplicationInput(BaseModel):
    application: str = Field(
        description="Friendly name of the application to open, e.g. 'whatsapp', 'chrome', 'notepad', 'spotify'."
    )
    target_desktop: Literal["user_desktop", "garvis_workspace"] | None = Field(
        default=None, description=TARGET_FIELD_DESCRIPTION + " Default: user_desktop."
    )


SW_SHOWMINNOACTIVE = 7


def make_launcher(target: str, is_shortcut: bool, background: bool, args: tuple[str, ...] = ()):
    """A zero-argument launcher. `background` asks Windows to show the
    app's first window minimized and NOT activated, so starting it can't
    take the user's focus (most apps honor this; the controller verifies
    and corrects either way)."""

    def launch() -> None:
        if is_shortcut:
            if background:
                os.startfile(target, show_cmd=SW_SHOWMINNOACTIVE)
            else:
                os.startfile(target)
        elif background:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = SW_SHOWMINNOACTIVE
            subprocess.Popen([target, *args], shell=False, startupinfo=startupinfo)
        else:
            subprocess.Popen([target, *args], shell=False)

    return launch


class OpenApplicationTool(Tool):
    name = "open_application"
    description = (
        "Open an installed Windows application by its common/friendly name (e.g. 'whatsapp', "
        "'chrome', 'discord', 'spotify', 'notepad'). Resolves against what's actually "
        "installed on this machine -- call this tool rather than assuming an app can't be "
        "opened. By default the app opens on the user's desktop, focused; to type into it pass "
        "that window title to keyboard_type. target_desktop='garvis_workspace' starts it in the "
        "background on Garvis's desktop without taking focus."
    )
    risk_level = RiskLevel.LOW
    presentation = "prominent"  # the user asked to see it
    input_model = OpenApplicationInput

    def __init__(
        self, allowlist: dict[str, str] | None = None, desktop=None, routing: DesktopRouting | None = None
    ) -> None:
        """`desktop` (a DesktopController), when given, is used to find the
        window the launch produced, put it on its target desktop and verify
        it -- so the next step (typing) has a real target."""
        self._allowlist = allowlist if allowlist is not None else load_allowlist()
        self._desktop = desktop
        self._routing = routing or DesktopRouting()

    @property
    def supported_applications(self) -> list[str]:
        return sorted(self._allowlist)

    async def execute(self, args: OpenApplicationInput) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult.fail("Opening applications is only supported on Windows.")

        key = args.application.strip().lower()
        resolution = self._routing.resolve("application", args.target_desktop, DesktopTarget.USER)
        background = resolution.target == DesktopTarget.WORKSPACE
        if background and self._desktop is None:
            return ToolResult.fail("Garvis's workspace desktop isn't available here, so the app was not started.")
        # Off the event loop: resolution can scan the Start Menu tree and
        # (for packaged apps) query the Start apps list via PowerShell.
        resolved = await asyncio.to_thread(resolve_application, args.application, self._allowlist)
        if resolved is None:
            return ToolResult.fail(
                f"Could not find an installed application matching '{args.application}' -- "
                "it may not be installed, or its name doesn't match what Garvis searched for."
            )
        target, is_shortcut = resolved
        launch = make_launcher(target, is_shortcut, background)

        if self._desktop is None:
            try:
                launch()
            except OSError as exc:
                log_extra(logger, 40, "open_application.failed", application=key, error=str(exc))
                return ToolResult.fail(f"Could not launch '{key}': {exc}")
            log_extra(logger, 20, "open_application.launched", application=key, resolved_path=target)
            return ToolResult.ok({"opened": key, "window_verified": False, **resolution.as_result()})

        from app.core.exceptions import DesktopError

        try:
            if background:
                window = await self._desktop.launch_in_workspace(key, launch)
            else:
                window = await self._desktop.launch_and_activate(key, launch)
        except OSError as exc:
            log_extra(logger, 40, "open_application.failed", application=key, error=str(exc))
            return ToolResult.fail(f"Could not launch '{key}': {exc}")
        except DesktopError as exc:
            log_extra(logger, 30, "open_application.window_not_ready", application=key, error=str(exc))
            return ToolResult.fail(str(exc))
        log_extra(
            logger,
            20,
            "open_application.launched",
            application=key,
            resolved_path=target,
            window=window["window"],
            target_desktop=resolution.target.value,
        )
        return ToolResult.ok({"opened": key, "window_verified": True, **window, **resolution.as_result()})
