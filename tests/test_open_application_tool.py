"""Tests for the application-launch tool (app/tools/applications.py).

`winreg` (Windows-only stdlib) is faked via sys.modules injection, same
pattern app/voice/tts/player.py's winsound tests use -- hermetic and
platform-independent regardless of what's installed. Start Menu scanning
is exercised against real tmp_path directories rather than the actual
filesystem.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from app.tools import applications
from app.tools.applications import (
    OpenApplicationInput,
    OpenApplicationTool,
    load_allowlist,
    resolve_application,
    resolve_from_app_paths_registry,
    resolve_from_apps_folder,
    resolve_from_start_menu,
)


@pytest.fixture(autouse=True)
def _no_real_start_apps(monkeypatch):
    """Keep every test hermetic: resolution stage 5 (Start "All apps" via
    PowerShell) returns nothing unless a test supplies its own list."""
    monkeypatch.setattr(applications, "_list_start_apps", lambda: [])
    monkeypatch.setattr(applications, "_start_apps_cache", None)


class _FakeKeyHandle:
    def __init__(self, value: str) -> None:
        self._value = value

    def __enter__(self) -> "_FakeKeyHandle":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _FakeWinReg:
    HKEY_CURRENT_USER = "HKCU"
    HKEY_LOCAL_MACHINE = "HKLM"

    def __init__(self, entries: dict[tuple[str, str], str]) -> None:
        self._entries = entries

    def OpenKey(self, hive, key_path):
        try:
            return _FakeKeyHandle(self._entries[(hive, key_path)])
        except KeyError as exc:
            raise FileNotFoundError(key_path) from exc

    def QueryValueEx(self, key: _FakeKeyHandle, value_name: str):
        return key._value, 1


# ------------------------------------------------------------- load_allowlist
def test_load_allowlist_returns_empty_dict_when_missing(tmp_path):
    assert load_allowlist(tmp_path / "does_not_exist.json") == {}


def test_load_allowlist_reads_json(tmp_path):
    path = tmp_path / "applications.json"
    path.write_text('{"notepad": "notepad.exe"}', encoding="utf-8")
    assert load_allowlist(path) == {"notepad": "notepad.exe"}


# ---------------------------------------------------- App Paths registry
def test_resolve_from_app_paths_registry_finds_entry(monkeypatch):
    fake = _FakeWinReg(
        {
            (
                "HKCU",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\whatsapp.exe",
            ): r"C:\Users\me\AppData\Local\WhatsApp\WhatsApp.exe",
        }
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert resolve_from_app_paths_registry("whatsapp") == r"C:\Users\me\AppData\Local\WhatsApp\WhatsApp.exe"


def test_resolve_from_app_paths_registry_accepts_name_with_exe_suffix(monkeypatch):
    fake = _FakeWinReg(
        {
            (
                "HKCU",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
            ): r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        }
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert resolve_from_app_paths_registry("chrome.exe") == r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def test_resolve_from_app_paths_registry_falls_back_to_hklm(monkeypatch):
    fake = _FakeWinReg(
        {
            (
                "HKLM",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\discord.exe",
            ): r"C:\Program Files\Discord\Discord.exe",
        }
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert resolve_from_app_paths_registry("discord") == r"C:\Program Files\Discord\Discord.exe"


def test_resolve_from_app_paths_registry_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    assert resolve_from_app_paths_registry("nonexistent") is None


def test_resolve_from_app_paths_registry_returns_none_when_winreg_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", None)
    assert resolve_from_app_paths_registry("chrome") is None


# ------------------------------------------------------------ Start Menu scan
def test_resolve_from_start_menu_matches_exact_stem(tmp_path):
    programs = tmp_path / "Programs"
    sub = programs / "WhatsApp"
    sub.mkdir(parents=True)
    lnk = sub / "WhatsApp.lnk"
    lnk.write_bytes(b"")
    assert resolve_from_start_menu("whatsapp", search_dirs=[programs]) == str(lnk)


def test_resolve_from_start_menu_falls_back_to_substring_match(tmp_path):
    programs = tmp_path / "Programs"
    programs.mkdir()
    lnk = programs / "WhatsApp Desktop.lnk"
    lnk.write_bytes(b"")
    assert resolve_from_start_menu("whatsapp", search_dirs=[programs]) == str(lnk)


def test_resolve_from_start_menu_returns_none_when_nothing_matches(tmp_path):
    programs = tmp_path / "Programs"
    programs.mkdir()
    (programs / "Notepad++.lnk").write_bytes(b"")
    assert resolve_from_start_menu("spotify", search_dirs=[programs]) is None


def test_resolve_from_start_menu_ignores_missing_directories(tmp_path):
    assert resolve_from_start_menu("whatsapp", search_dirs=[tmp_path / "missing"]) is None


def test_resolve_from_start_menu_blank_name_returns_none(tmp_path):
    assert resolve_from_start_menu("   ", search_dirs=[tmp_path]) is None


# --------------------------------------------------------- resolve_application
def test_resolve_application_prefers_allowlist_over_registry(monkeypatch):
    fake = _FakeWinReg(
        {
            (
                "HKCU",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\notepad.exe",
            ): r"C:\wrong\notepad.exe",
        }
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    result = resolve_application("notepad", allowlist={"notepad": "notepad.exe"})
    assert result == ("notepad.exe", False)


def test_resolve_application_falls_back_to_registry(monkeypatch):
    fake = _FakeWinReg(
        {
            (
                "HKCU",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\whatsapp.exe",
            ): r"C:\WhatsApp\WhatsApp.exe",
        }
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    result = resolve_application("WhatsApp", allowlist={})
    assert result == (r"C:\WhatsApp\WhatsApp.exe", False)


def test_resolve_application_falls_back_to_start_menu(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    programs = tmp_path / "Programs"
    programs.mkdir()
    lnk = programs / "Discord.lnk"
    lnk.write_bytes(b"")
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [programs])
    result = resolve_application("discord", allowlist={})
    assert result == (str(lnk), True)


def test_resolve_application_falls_back_to_path_lookup(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(
        applications.shutil,
        "which",
        lambda name: r"C:\tools\pycharm.exe" if name == "pycharm.exe" else None,
    )
    result = resolve_application("PyCharm", allowlist={})
    assert result == (r"C:\tools\pycharm.exe", False)


def test_resolve_application_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(applications.shutil, "which", lambda name: None)
    assert resolve_application("totally-made-up-app", allowlist={}) is None


def test_resolve_application_blank_name_returns_none():
    assert resolve_application("   ", allowlist={"notepad": "notepad.exe"}) is None


# --------------------------------------------------------------- execute()
@pytest.mark.asyncio
async def test_execute_launches_resolved_executable_via_subprocess(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(
        applications.subprocess,
        "Popen",
        lambda args, shell=False: calls.append((args, shell)),
    )
    tool = OpenApplicationTool(allowlist={"notepad": "notepad.exe"})
    result = await tool.execute(OpenApplicationInput(application="Notepad"))

    assert result.success
    assert result.data == {
        "opened": "notepad",
        "window_verified": False,
        "target_desktop": "user_desktop",
        "target_reason": "default for this kind of window",
    }
    assert calls == [(["notepad.exe"], False)]


@pytest.mark.asyncio
async def test_execute_launches_start_menu_shortcut_via_startfile(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    lnk = tmp_path / "WhatsApp.lnk"
    lnk.write_bytes(b"")
    monkeypatch.setattr(applications, "resolve_application", lambda name, allowlist: (str(lnk), True))
    startfile_calls = []
    monkeypatch.setattr(applications.os, "startfile", lambda path: startfile_calls.append(path))

    tool = OpenApplicationTool(allowlist={})
    result = await tool.execute(OpenApplicationInput(application="WhatsApp"))

    assert result.success
    assert startfile_calls == [str(lnk)]


@pytest.mark.asyncio
async def test_execute_fails_with_clear_message_when_unresolved(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(applications, "resolve_application", lambda name, allowlist: None)

    tool = OpenApplicationTool(allowlist={})
    result = await tool.execute(OpenApplicationInput(application="totally-unknown-app"))

    assert not result.success
    assert "totally-unknown-app" in result.error


@pytest.mark.asyncio
async def test_execute_fails_on_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    tool = OpenApplicationTool(allowlist={"notepad": "notepad.exe"})
    result = await tool.execute(OpenApplicationInput(application="notepad"))

    assert not result.success
    assert "Windows" in result.error


@pytest.mark.asyncio
async def test_execute_reports_launch_oserror(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def _raise(args, shell=False):
        raise OSError("boom")

    monkeypatch.setattr(applications.subprocess, "Popen", _raise)
    tool = OpenApplicationTool(allowlist={"notepad": "notepad.exe"})
    result = await tool.execute(OpenApplicationInput(application="notepad"))

    assert not result.success
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_execute_uses_default_allowlist_when_none_given(monkeypatch):
    """OpenApplicationTool() with no explicit allowlist loads from
    config/applications.json (the real curated allowlist)."""
    tool = OpenApplicationTool()
    assert "notepad" in tool.supported_applications


# ------------------------------------------- stage 5: Start "All apps" (AUMID)
_WHATSAPP_AUMID = "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"
_APPS_FOLDER = "shell:AppsFolder\\"
_WHATSAPP_TARGET = _APPS_FOLDER + _WHATSAPP_AUMID
_START_APPS = [
    ("Google Chrome", "Chrome"),
    ("WhatsApp Beta", "5319275A.WhatsAppDesktopBeta_cv1g1gvanyjgm!App"),
    ("WhatsApp", _WHATSAPP_AUMID),
    ("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
]


def test_resolve_from_apps_folder_exact_display_name():
    assert resolve_from_apps_folder("WhatsApp", _START_APPS) == _WHATSAPP_TARGET


def test_resolve_from_apps_folder_prefers_shortest_substring_match():
    apps = [("WhatsApp Beta", "beta!App"), ("WhatsApp Desktop", _WHATSAPP_AUMID)]
    assert resolve_from_apps_folder("whats", apps) == _APPS_FOLDER + "beta!App"


def test_resolve_from_apps_folder_falls_back_to_package_name():
    apps = [("Chats", _WHATSAPP_AUMID)]  # display name doesn't match, package does
    assert resolve_from_apps_folder("whatsapp", apps) == _WHATSAPP_TARGET


def test_resolve_from_apps_folder_none_when_no_match_or_blank():
    assert resolve_from_apps_folder("spotify", _START_APPS) is None
    assert resolve_from_apps_folder("  ", _START_APPS) is None


def test_resolve_application_finds_packaged_whatsapp_when_stages_1_to_4_miss(monkeypatch):
    """Regression test for the WhatsApp bug: a Store/MSIX app has no App
    Paths entry, no Start Menu .lnk, nothing on PATH -- only an AUMID."""
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(applications.shutil, "which", lambda name: None)
    monkeypatch.setattr(applications, "_list_start_apps", lambda: list(_START_APPS))
    assert resolve_application("WhatsApp", allowlist={}) == (_WHATSAPP_TARGET, True)


def test_apps_folder_is_consulted_last(monkeypatch):
    """Existing resolutions must not change: PATH (stage 4) still wins."""
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(
        applications.shutil, "which", lambda name: r"C:\bin\calculator.exe" if name == "calculator" else None
    )
    monkeypatch.setattr(applications, "_list_start_apps", lambda: list(_START_APPS))
    assert resolve_application("calculator", allowlist={}) == (r"C:\bin\calculator.exe", False)


def test_list_start_apps_caches_success_but_not_failure(monkeypatch):
    calls = []

    def fake_list():
        calls.append(1)
        return [] if len(calls) == 1 else list(_START_APPS)

    monkeypatch.setattr(applications, "_list_start_apps", fake_list)
    assert applications.list_start_apps() == []  # failure: not cached
    assert applications.list_start_apps() == _START_APPS
    assert applications.list_start_apps() == _START_APPS  # served from cache
    assert len(calls) == 2


def test_real_list_start_apps_parses_powershell_json(monkeypatch):
    monkeypatch.undo()  # use the real _list_start_apps, with subprocess faked
    monkeypatch.setattr(sys, "platform", "win32")
    payload = '[{"Name":"WhatsApp","AppID":"%s"},{"Name":"Broken","AppID":null}]' % _WHATSAPP_AUMID

    class _Completed:
        returncode = 0
        stdout = payload.encode("utf-8")

    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return _Completed()

    monkeypatch.setattr(applications.subprocess, "run", fake_run)
    assert applications._list_start_apps() == [("WhatsApp", _WHATSAPP_AUMID)]
    assert seen["args"][0] == "powershell.exe" and "Get-StartApps" in seen["args"][-1]


def test_real_list_start_apps_single_object_and_failures(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(sys, "platform", "win32")

    class _Completed:
        def __init__(self, code, out):
            self.returncode, self.stdout = code, out

    monkeypatch.setattr(
        applications.subprocess, "run", lambda args, **kw: _Completed(0, b'{"Name":"WhatsApp","AppID":"x!App"}')
    )
    assert applications._list_start_apps() == [("WhatsApp", "x!App")]
    monkeypatch.setattr(applications.subprocess, "run", lambda args, **kw: _Completed(1, b""))
    assert applications._list_start_apps() == []

    def _timeout(args, **kw):
        raise subprocess.TimeoutExpired(args, 15)

    monkeypatch.setattr(applications.subprocess, "run", _timeout)
    assert applications._list_start_apps() == []


@pytest.mark.asyncio
async def test_execute_launches_packaged_app_via_startfile(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinReg({}))
    monkeypatch.setattr(applications, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(applications.shutil, "which", lambda name: None)
    monkeypatch.setattr(applications, "_list_start_apps", lambda: list(_START_APPS))
    started = []
    monkeypatch.setattr(applications.os, "startfile", lambda path: started.append(path))

    result = await OpenApplicationTool(allowlist={}).execute(OpenApplicationInput(application="WhatsApp"))

    assert result.success and result.data == {
        "opened": "whatsapp",
        "window_verified": False,
        "target_desktop": "user_desktop",
        "target_reason": "default for this kind of window",
    }
    assert started == [_WHATSAPP_TARGET]
