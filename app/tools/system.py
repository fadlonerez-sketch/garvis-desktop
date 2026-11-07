"""System inspection and command tools.

- list_processes / system_status / recent_system_errors /
  list_installed_apps: fixed read-only queries -- LOW risk.
- run_powershell: MEDIUM. Read-only commands (as proven by
  app/security/commands.py, which parses the command with PowerShell's own
  parser) run immediately; anything that could change the system, reveal
  credentials or can't be verified asks the user first.

All output is passed through redact_secrets() before it reaches the LLM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import DesktopError
from app.security.commands import CommandPolicy
from app.security.sensitive import redact_secrets
from app.system.info import SystemInfo
from app.system.powershell import PowerShellError, PowerShellRunner
from app.tools.base import RiskLevel, Tool, ToolResult

_MAX_OUTPUT_CHARS = 8000


def _clip(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    text = redact_secrets(text.strip())
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters omitted] ...\n{text[-half:]}"


class ListProcessesInput(BaseModel):
    name: str | None = Field(default=None, description="Only processes whose name or window title contains this")
    sort_by: Literal["cpu", "memory", "name"] = Field(default="cpu")
    limit: int = Field(default=25, ge=1, le=200)
    not_responding_only: bool = Field(default=False, description="Only processes that are hung")


class ListProcessesTool(Tool):
    name = "list_processes"
    description = (
        "List running processes with live CPU %, memory (MB), whether they respond, and their main "
        "window title. Filter by name, sort by cpu/memory/name, or show only hung processes."
    )
    risk_level = RiskLevel.LOW
    input_model = ListProcessesInput

    def __init__(self, info: SystemInfo) -> None:
        self._info = info

    async def execute(self, args: ListProcessesInput) -> ToolResult:
        try:
            rows = await self._info.processes()
        except PowerShellError as exc:
            return ToolResult.fail(str(exc))
        if args.name:
            needle = args.name.lower()
            rows = [
                r
                for r in rows
                if needle in str(r.get("name", "")).lower() or needle in str(r.get("window_title") or "").lower()
            ]
        if args.not_responding_only:
            rows = [r for r in rows if r.get("responding") is False]
        key = {
            "cpu": lambda r: -(r.get("cpu_percent") or 0),
            "memory": lambda r: -(r.get("memory_mb") or 0),
            "name": lambda r: str(r.get("name", "")).lower(),
        }[args.sort_by]
        rows.sort(key=key)
        for row in rows:
            if not row.get("window_title"):
                row.pop("window_title", None)
        return ToolResult.ok({"total": len(rows), "processes": rows[: args.limit]})


class NoInput(BaseModel):
    pass


class SystemStatusTool(Tool):
    name = "system_status"
    description = "OS version, uptime, CPU load, memory, free space on every disk and battery level."
    risk_level = RiskLevel.LOW
    input_model = NoInput

    def __init__(self, info: SystemInfo) -> None:
        self._info = info

    async def execute(self, args: NoInput) -> ToolResult:
        try:
            return ToolResult.ok(await self._info.status())
        except PowerShellError as exc:
            return ToolResult.fail(str(exc))


class RecentErrorsInput(BaseModel):
    hours: float = Field(default=24, gt=0, le=24 * 30)
    filter: str | None = Field(
        default=None, description="Only events whose source or message contains this, e.g. an app name"
    )
    limit: int = Field(default=20, ge=1, le=100)


class RecentSystemErrorsTool(Tool):
    name = "recent_system_errors"
    description = (
        "Recent errors and critical events from the Windows Application and System event logs "
        "(crashes, failed services, driver errors), newest first. Filter by app/source name."
    )
    risk_level = RiskLevel.LOW
    input_model = RecentErrorsInput

    def __init__(self, info: SystemInfo) -> None:
        self._info = info

    async def execute(self, args: RecentErrorsInput) -> ToolResult:
        try:
            events = await self._info.recent_errors(args.hours, args.filter, args.limit)
        except PowerShellError as exc:
            return ToolResult.fail(str(exc))
        for event in events:
            event["message"] = redact_secrets(str(event.get("message", "")))
        return ToolResult.ok({"count": len(events), "events": events})


class InstalledAppsInput(BaseModel):
    filter: str | None = Field(default=None, description="Only apps whose name contains this")
    limit: int = Field(default=60, ge=1, le=500)


class ListInstalledAppsTool(Tool):
    name = "list_installed_apps"
    description = "Installed applications (name, version, publisher), optionally filtered by name."
    risk_level = RiskLevel.LOW
    input_model = InstalledAppsInput

    def __init__(self, info: SystemInfo) -> None:
        self._info = info

    async def execute(self, args: InstalledAppsInput) -> ToolResult:
        import asyncio

        apps = await asyncio.to_thread(self._info.installed_apps)
        if args.filter:
            needle = args.filter.lower()
            apps = [a for a in apps if needle in a["name"].lower()]
        return ToolResult.ok({"total": len(apps), "apps": apps[: args.limit]})


class RunPowerShellInput(BaseModel):
    command: str = Field(description="PowerShell command or short script to run")
    working_directory: str | None = Field(default=None, description="Folder to run it in (default: your home folder)")
    timeout_seconds: float = Field(default=60, ge=1, le=600)
    target_desktop: Literal["user_desktop", "garvis_workspace"] | None = Field(
        default=None,
        description="Only matters if the command starts programs with windows: 'garvis_workspace' "
        "moves the windows it opens to Garvis's background desktop; 'user_desktop' keeps them on the user's "
        "screen. PowerShell itself always runs hidden. Prefer open_application for starting apps.",
    )


class RunPowerShellTool(Tool):
    name = "run_powershell"
    description = (
        "Run a PowerShell command on this PC and get its output (exit code, stdout, stderr). Best for "
        "system facts, diagnostics, git, running tests and anything without a dedicated tool. "
        "Read-only commands run immediately; commands that change things or contact another computer "
        "(lookups/pings to hosts other than localhost or well-known ones, UNC paths, URLs) ask the user first. "
        "Windows it opens are reported with the desktop they're on (see target_desktop)."
    )
    risk_level = RiskLevel.MEDIUM
    input_model = RunPowerShellInput

    def __init__(
        self, runner: PowerShellRunner | None = None, policy: CommandPolicy | None = None, desktop=None
    ) -> None:
        """`desktop` (DesktopController with the workspace attached) lets
        windows opened by a command be found and put on their target desktop."""
        self._runner = runner or PowerShellRunner()
        self._policy = policy or CommandPolicy(self._runner)
        self._desktop = desktop

    async def _window_snapshot(self) -> tuple[set[int], set[int], int | None] | None:
        if self._desktop is None:
            return None
        try:
            windows = await self._desktop.list_windows(include_other_desktops=True)
            foreground = await self._desktop.foreground_window()
        except DesktopError:
            return None
        return {w.handle for w in windows}, {w.pid for w in windows}, foreground.handle if foreground else None

    async def _place_new_windows(self, before: tuple[set[int], set[int], int | None], target: str | None) -> list[dict]:
        """Windows that appeared during the run AND belong to a process that
        had no window before it -- i.e. programs this command started, not
        anything the user already had open. Put on `target` when given."""
        handles, pids, user_foreground = before
        windows = await self._desktop.list_windows(include_other_desktops=True)
        fresh = [
            w for w in windows if w.handle not in handles and w.pid not in pids and not self._desktop.is_own_window(w)
        ]
        if not fresh:
            return []
        report = []
        for window in fresh:
            if target == "garvis_workspace":
                await self._desktop.move_to_workspace(window.handle)
            elif target == "user_desktop" and not window.on_current_desktop:
                await self._desktop.show_window(window.handle)
            report.append({"title": window.title, "app": window.process_name})
        if target == "garvis_workspace" and user_foreground is not None:
            await self._desktop.restore_foreground(user_foreground)
        locations = await self._desktop.locate_windows(fresh)
        for row, window in zip(report, fresh, strict=True):
            row["desktop"] = {"user": "user_desktop", "garvis_workspace": "garvis_workspace"}.get(
                locations.get(window.handle), "other_desktop"
            )
        return report

    async def confirmation_reason(self, args: RunPowerShellInput) -> str | None:
        return await self._policy.confirmation_reason(args.command)

    async def execute(self, args: RunPowerShellInput) -> ToolResult:
        if args.working_directory and os.path.expandvars(args.working_directory.strip()).startswith(("\\\\", "//")):
            # Checked before touching the path: even is_dir() on a UNC path contacts that server.
            return ToolResult.fail("Network (UNC) folders are not supported as a working directory.")
        cwd = (
            Path(os.path.expandvars(os.path.expanduser(args.working_directory)))
            if args.working_directory
            else Path.home()
        )
        if not cwd.is_dir():
            return ToolResult.fail(f"Folder not found: {cwd}")
        if args.target_desktop == "garvis_workspace":
            if self._desktop is None or not await self._desktop.workspace_available():
                return ToolResult.fail(
                    "Garvis's workspace desktop isn't available, so the command was not run "
                    "(its windows would otherwise open on the user's screen)."
                )
        before = await self._window_snapshot()
        try:
            result = await self._runner.run(args.command, cwd=cwd, timeout_seconds=args.timeout_seconds)
        except PowerShellError as exc:
            return ToolResult.fail(str(exc))
        windows: list[dict] = []
        if before is not None:
            try:
                windows = await self._place_new_windows(before, args.target_desktop)
            except DesktopError as exc:
                windows = [{"error": f"Windows the command opened couldn't be placed: {exc}"}]
        payload = {
            "exit_code": result.exit_code,
            "stdout": _clip(result.stdout),
            "stderr": _clip(result.stderr, 3000),
            "seconds": result.elapsed_seconds,
        }
        if windows:
            payload["opened_windows"] = windows
        if result.timed_out:
            payload["timed_out"] = True
            return ToolResult(success=False, data=payload, error=f"Timed out after {args.timeout_seconds:.0f} s.")
        if result.exit_code != 0:
            return ToolResult(success=False, data=payload, error=f"Exited with code {result.exit_code}.")
        return ToolResult.ok(payload)
