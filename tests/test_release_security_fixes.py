"""Regression tests for the security fixes made while preparing the public release.

Each check only escalates (refuses or asks first) -- nothing that asked before runs silently now.
Pure logic: no PowerShell, network, desktop input or file outside tmp_path.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app.mobile.server import COOKIE_NAME
from app.security.commands import CommandAnalysis, ParsedCommand, classify
from app.tools import filesystem, open_path
from app.tools.applications import resolve_application
from app.tools.desktop import DEFAULT_COMMIT_APPS
from app.tools.webpage import UnsafeURLError, _validate_url_is_public
from tests.test_eye_station_api import _pair_station
from tests.test_mobile_native_api import _start


# ---------------------------------------------------------- open_application
def test_open_application_refuses_a_path(tmp_path):
    script = tmp_path / "x.bat"
    script.write_text("@echo off\n", encoding="utf-8")
    assert resolve_application(str(script), allowlist={}) is None
    assert resolve_application(script.as_posix(), allowlist={}) is None
    assert resolve_application("notepad", allowlist={"notepad": "notepad.exe"}) == ("notepad.exe", False)


# ------------------------------------------------------- PowerShell policy
def _cmd(name: str, *args: str) -> CommandAnalysis:
    return CommandAnalysis(commands=[ParsedCommand(name=name, args=list(args))])


@pytest.mark.parametrize("args", [["Delete"], ["-MemberName", "Kill"], ["Create", "calc.exe"]])
def test_foreach_object_member_name_form_needs_confirmation(args):
    assert classify("Get-ChildItem | % " + " ".join(args), _cmd("%", *args)) is not None


def test_foreach_object_script_block_is_still_read_only():
    assert classify("Get-Process | ForEach-Object { $_.Name }", _cmd("ForEach-Object", "{ $_.Name }")) is None


@pytest.mark.parametrize("command", [
    "$env:GIT_EXTERNAL_DIFF='cmd /c calc'; git diff",
    "${env:GIT_CONFIG_PARAMETERS} = 'x'; git status",
    "$Env:PATH += ';C:\\evil'; git log",
])
def test_environment_variable_assignment_needs_confirmation(command):
    assert classify(command, _cmd("git", "status")) is not None


def test_reading_an_environment_variable_is_still_read_only():
    assert classify("if ($env:USERNAME -eq 'x') { git status }", _cmd("git", "status")) is None


@pytest.mark.parametrize("args", [
    ["grep", "--no-index", "-Ocalc.exe", "x"],
    ["grep", "--open-files-in-pager=calc", "x"],
    ["diff", "--output=C:\\Users\\me\\file.txt"],
    ["diff", "--ext-diff"],
    ["log", "--output", "out.txt"],
])
def test_git_flags_that_run_programs_or_write_files_need_confirmation(args):
    assert classify("git " + " ".join(args), _cmd("git", *args)) is not None


@pytest.mark.parametrize("args", [["log", "--oneline", "-5"], ["status"], ["grep", "-o", "TODO"]])
def test_ordinary_git_reads_are_still_read_only(args):
    assert classify("git " + " ".join(args), _cmd("git", *args)) is None


# ------------------------------------------------------------- file writes
@pytest.mark.parametrize("name", ["run.bat", "profile.ps1", "x.lnk", "a.vbs", "setup.EXE"])
def test_writing_a_launchable_file_needs_confirmation(tmp_path, name):
    assert filesystem._write_reason((tmp_path / name).resolve()) is not None


def test_writing_into_git_config_or_hooks_needs_confirmation(tmp_path):
    assert filesystem._write_reason((tmp_path / "repo" / ".git" / "hooks" / "pre-commit").resolve())
    assert filesystem._write_reason((tmp_path / ".gitconfig").resolve())


def test_writing_into_the_startup_folder_needs_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    startup = tmp_path / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    assert filesystem._write_reason((startup / "notes.txt").resolve()) is not None


def test_writing_garvis_own_code_needs_confirmation_but_its_data_does_not():
    root = Path(filesystem.__file__).resolve().parents[2]
    assert filesystem._write_reason(root / "app" / "security" / "commands.txt") is not None
    assert filesystem._write_reason(root / "data" / "notes.txt") is None


def test_ordinary_documents_and_source_files_are_still_written_directly(tmp_path):
    for name in ("notes.txt", "app.py", "README.md", "data.json"):
        assert filesystem._write_reason((tmp_path / name).resolve()) is None, name


@pytest.mark.parametrize("raw", ["\\\\attacker@80\\share\\x.txt", "//attacker/share/x.txt"])
def test_unc_paths_are_refused_before_they_are_resolved(tmp_path, monkeypatch, raw):
    def no_resolve(self, *a, **k):
        raise AssertionError("a UNC path must not be resolved (that contacts the server)")

    monkeypatch.setattr(Path, "resolve", no_resolve)
    with pytest.raises(ValueError, match="UNC"):
        filesystem.resolve_path(tmp_path, raw)
    assert "UNC" in open_path.refusal(open_path.expand(raw))


@pytest.mark.parametrize("name", ["setup.iso", "disk.vhdx", "work.rdp", "dark.theme", "sandbox.wsb",
                                  "update.msu", "app.appinstaller", "run.sh", "budget.xlsm", "notes.docm"])
def test_open_path_refuses_mounting_and_launching_types(tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"x")
    assert open_path.refusal(target)


# --------------------------------------------------------- desktop commit guard
def test_explorer_and_start_search_are_commit_guarded():
    apps = {a.strip() for a in DEFAULT_COMMIT_APPS.split(",")}
    assert {"explorer", "searchhost", "startmenuexperiencehost"} <= apps


# ------------------------------------------------------------------- SSRF
@pytest.mark.parametrize("address", ["100.101.102.103", "100.64.0.1"])
def test_fetch_refuses_cgnat_and_tailscale_addresses(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", (address, 0))])
    with pytest.raises(UnsafeURLError):
        _validate_url_is_public("http://nas.example/")


def test_fetch_still_allows_public_addresses(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))])
    _validate_url_is_public("http://public.example/")


# --------------------------------------------------- phone web app vs stations
async def test_a_station_token_is_not_a_web_app_session(tmp_path):
    h = await _start(tmp_path)
    try:
        await _pair_station(h)
        station_cookie = {"Cookie": f"{COOKIE_NAME}={h.token}"}
        assert (await (await h.client.get("/api/session", headers=station_cookie)).json()) == {"paired": False}
        await h.pair()  # a phone
        phone_cookie = {"Cookie": f"{COOKIE_NAME}={h.token}"}
        assert (await (await h.client.get("/api/session", headers=phone_cookie)).json()) == {"paired": True}
    finally:
        await h.service.stop()
        await h.tasks.stop()
        await h.client.close()
