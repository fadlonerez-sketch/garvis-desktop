"""Runs Windows PowerShell for Garvis.

Scripts are passed with -EncodedCommand (UTF-16LE base64), so there is no
shell quoting or interpolation of the text on the Python side. Every run is
non-interactive, without the user's profile, without a console window, with
UTF-8 output and a hard timeout (the whole process tree is killed when it
expires). Values that a script needs (paths, filters) are embedded as
base64 strings via `ps_string()`, never by string concatenation of raw text.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
_PREAMBLE = (
    "$ProgressPreference='SilentlyContinue';"
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "$OutputEncoding=[System.Text.Encoding]::UTF8;"
)


class PowerShellError(Exception):
    pass


@dataclass(frozen=True)
class PowerShellResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_seconds: float


def ps_string(value: str) -> str:
    """A PowerShell expression that evaluates to `value`, safe for any text."""
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"([System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{encoded}')))"


class PowerShellRunner:
    def __init__(self, executable: str = "powershell.exe") -> None:
        self._executable = executable

    def _command(self, script: str) -> list[str]:
        encoded = base64.b64encode((_PREAMBLE + script).encode("utf-16-le")).decode("ascii")
        return [self._executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]

    def run_sync(self, script: str, cwd: Path | None = None, timeout_seconds: float = 60.0) -> PowerShellResult:
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                self._command(script),
                cwd=str(cwd) if cwd else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as exc:
            raise PowerShellError(f"PowerShell could not be started: {exc}") from exc
        try:
            out, err = process.communicate(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                    capture_output=True,
                    creationflags=CREATE_NO_WINDOW,
                )
            process.kill()
            out, err = process.communicate()
        return PowerShellResult(
            exit_code=process.returncode if not timed_out else -1,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            timed_out=timed_out,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    async def run(self, script: str, cwd: Path | None = None, timeout_seconds: float = 60.0) -> PowerShellResult:
        return await asyncio.to_thread(self.run_sync, script, cwd, timeout_seconds)

    async def run_json(self, script: str, timeout_seconds: float = 60.0):
        """Run a script whose output is JSON (ConvertTo-Json); parsed result."""
        result = await self.run(script, timeout_seconds=timeout_seconds)
        if result.timed_out:
            raise PowerShellError(f"The query took longer than {timeout_seconds:.0f} s.")
        text = result.stdout.strip()
        if result.exit_code != 0 and not text:
            raise PowerShellError(result.stderr.strip()[:400] or f"PowerShell exited with {result.exit_code}.")
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise PowerShellError(f"Unexpected PowerShell output: {text[:200]}") from exc
