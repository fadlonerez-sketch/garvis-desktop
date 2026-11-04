"""System inspection, PowerShell and file tools."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.security.permissions import PermissionChecker, PermissionDecision
from app.system.info import SystemInfo
from app.system.powershell import PowerShellError, PowerShellResult, PowerShellRunner
from app.tools.filesystem import (
    EditTextFileTool,
    FindFilesTool,
    ListDirectoryTool,
    ReadTextFileTool,
    WriteTextFileTool,
)
from app.tools.system import (
    ListInstalledAppsTool,
    ListProcessesTool,
    RecentSystemErrorsTool,
    RunPowerShellTool,
    SystemStatusTool,
)


# --------------------------------------------------------------- processes
class _Info(SystemInfo):
    def __init__(self):
        super().__init__(runner=None, registry_apps=lambda: [{"name": "Python 3.14", "version": "3.14.5", "publisher": "PSF"}],
                         start_apps=lambda: [("WhatsApp", "5319275A.WhatsAppDesktop!App"), ("python 3.14", "x")])

    async def processes(self):
        return [
            {"name": "chrome", "pid": 1, "cpu_percent": 3.0, "memory_mb": 900.0, "responding": True, "window_title": "Gmail"},
            {"name": "Code", "pid": 2, "cpu_percent": 25.0, "memory_mb": 400.0, "responding": True, "window_title": ""},
            {"name": "badapp", "pid": 3, "cpu_percent": 0.0, "memory_mb": 50.0, "responding": False, "window_title": "Bad"},
        ]

    async def status(self):
        return {"os": "Windows 11", "disks": [{"drive": "C:", "free_gb": 10.0}]}

    async def recent_errors(self, hours, filter_text, limit):
        return [{"source": "Application Error", "message": "Faulting app badapp.exe token=abc123"}]


async def test_processes_are_sorted_filtered_and_hung_ones_found():
    tool = ListProcessesTool(_Info())
    by_cpu = await tool.execute(tool.validate_arguments({}))
    assert [p["name"] for p in by_cpu.data["processes"]] == ["Code", "chrome", "badapp"]
    assert "window_title" not in by_cpu.data["processes"][0]  # empty titles are dropped
    by_memory = await tool.execute(tool.validate_arguments({"sort_by": "memory", "limit": 1}))
    assert by_memory.data["processes"][0]["name"] == "chrome" and by_memory.data["total"] == 3
    hung = await tool.execute(tool.validate_arguments({"not_responding_only": True}))
    assert [p["name"] for p in hung.data["processes"]] == ["badapp"]
    by_title = await tool.execute(tool.validate_arguments({"name": "gmail"}))
    assert [p["name"] for p in by_title.data["processes"]] == ["chrome"]


async def test_status_errors_and_installed_apps():
    info = _Info()
    assert (await SystemStatusTool(info).execute(SystemStatusTool.input_model())).data["os"] == "Windows 11"
    errors = await RecentSystemErrorsTool(info).execute(RecentSystemErrorsTool.input_model(filter="badapp"))
    assert "abc123" not in errors.data["events"][0]["message"]
    apps = await ListInstalledAppsTool(info).execute(ListInstalledAppsTool.input_model(filter="python"))
    assert apps.data["total"] == 1 and apps.data["apps"][0]["version"] == "3.14.5"  # registry entry wins
    every = await ListInstalledAppsTool(info).execute(ListInstalledAppsTool.input_model())
    assert {a["name"] for a in every.data["apps"]} == {"Python 3.14", "WhatsApp"}


@pytest.mark.parametrize("tool_cls", [ListProcessesTool, SystemStatusTool, RecentSystemErrorsTool, ListInstalledAppsTool])
def test_inspection_tools_never_ask(tool_cls):
    assert PermissionChecker().evaluate(tool_cls(_Info())).decision == PermissionDecision.ALLOW


# --------------------------------------------------------------- powershell
class _FakeRunner(PowerShellRunner):
    def __init__(self, result: PowerShellResult | None = None, error: str | None = None):
        super().__init__()
        self.result = result
        self.error = error
        self.calls = []

    async def run(self, script, cwd=None, timeout_seconds=60.0):
        self.calls.append((script, cwd, timeout_seconds))
        if self.error:
            raise PowerShellError(self.error)
        return self.result


class _Policy:
    def __init__(self, reason=None):
        self.reason = reason

    async def confirmation_reason(self, command):
        return self.reason


def _ok(stdout="", stderr="", code=0, timed_out=False):
    return PowerShellResult(exit_code=code, stdout=stdout, stderr=stderr, timed_out=timed_out, elapsed_seconds=0.1)


async def test_run_powershell_returns_structured_redacted_output(tmp_path):
    runner = _FakeRunner(_ok(stdout="build ok\r\nGROQ=gsk_" + "a" * 40))
    tool = RunPowerShellTool(runner, _Policy())
    result = await tool.execute(tool.validate_arguments({"command": "Get-Date", "working_directory": str(tmp_path)}))
    assert result.success and result.data["exit_code"] == 0
    assert "build ok" in result.data["stdout"] and "gsk_aaaa" not in result.data["stdout"]
    assert runner.calls[0][1] == tmp_path


async def test_run_powershell_reports_failures_honestly(tmp_path):
    failing = RunPowerShellTool(_FakeRunner(_ok(stderr="boom", code=1)), _Policy())
    result = await failing.execute(failing.validate_arguments({"command": "x"}))
    assert not result.success and "code 1" in result.error and result.data["stderr"] == "boom"
    slow = RunPowerShellTool(_FakeRunner(_ok(timed_out=True, code=-1)), _Policy())
    result = await slow.execute(slow.validate_arguments({"command": "x", "timeout_seconds": 5}))
    assert not result.success and result.data["timed_out"]
    missing = RunPowerShellTool(_FakeRunner(_ok()), _Policy())
    result = await missing.execute(missing.validate_arguments({"command": "x", "working_directory": str(tmp_path / "nope")}))
    assert not result.success and "not found" in result.error


async def test_long_output_keeps_the_start_and_the_end():
    tool = RunPowerShellTool(_FakeRunner(_ok(stdout="HEAD" + "x" * 20000 + "TAIL")), _Policy())
    out = (await tool.execute(tool.validate_arguments({"command": "x"}))).data["stdout"]
    assert out.startswith("HEAD") and out.endswith("TAIL") and "omitted" in out


async def test_run_powershell_confirmation_comes_from_the_command_policy():
    tool = RunPowerShellTool(_FakeRunner(_ok()), _Policy("'Remove-Item' can change the system."))
    decision = await PermissionChecker().evaluate_call(tool, tool.validate_arguments({"command": "Remove-Item x"}))
    assert decision.decision == PermissionDecision.REQUIRE_CONFIRMATION
    quiet = RunPowerShellTool(_FakeRunner(_ok()), _Policy(None))
    assert (await PermissionChecker().evaluate_call(quiet, quiet.validate_arguments({"command": "Get-Date"}))).decision \
        == PermissionDecision.ALLOW


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell")
async def test_real_powershell_and_system_queries():
    tool = RunPowerShellTool()
    result = await tool.execute(tool.validate_arguments({"command": "'garvis-' + (2 + 3)"}))
    assert result.success and result.data["stdout"] == "garvis-5"
    info = SystemInfo()
    processes = await info.processes()
    assert any(p["pid"] for p in processes) and {"cpu_percent", "memory_mb", "responding"} <= set(processes[0])
    status = await info.status()
    assert status["memory"]["total_gb"] > 0 and status["disks"]


# ------------------------------------------------------------------- files
@pytest.fixture
def root(tmp_path):
    (tmp_path / "garvis_root").mkdir()
    return tmp_path / "garvis_root"


async def test_absolute_paths_and_line_ranges(tmp_path, root):
    log = tmp_path / "app.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 1001)), encoding="utf-8")
    tool = ReadTextFileTool(root)
    tail = await tool.execute(tool.validate_arguments({"path": str(log), "start_line": 991, "max_lines": 50}))
    assert tail.success and tail.data["content"].splitlines() == [f"line {i}" for i in range(991, 1001)]
    assert tail.data["total_lines"] == 1000
    listing = ListDirectoryTool(root)
    entries = await listing.execute(listing.validate_arguments({"path": str(tmp_path)}))
    assert {"app.log", "garvis_root"} <= {e["name"] for e in entries.data}


async def test_relative_paths_still_cannot_escape_the_garvis_folder(root):
    tool = ReadTextFileTool(root)
    result = await tool.execute(tool.validate_arguments({"relative_path": "../../x.txt"}))
    assert not result.success and "escapes" in result.error


async def test_credential_files_need_confirmation_and_are_redacted(tmp_path, root):
    env = tmp_path / ".env"
    env.write_text("GARVIS_GROQ_API_KEY=gsk_" + "b" * 40 + "\nDEBUG=1\n", encoding="utf-8")
    tool = ReadTextFileTool(root)
    args = tool.validate_arguments({"path": str(env)})
    assert (await PermissionChecker().evaluate_call(tool, args)).decision == PermissionDecision.REQUIRE_CONFIRMATION
    content = (await tool.execute(args)).data["content"]  # even when the user approves
    assert "gsk_bbbb" not in content and "DEBUG=1" in content
    ordinary = tool.validate_arguments({"path": str(tmp_path / "notes.txt")})
    assert (await PermissionChecker().evaluate_call(tool, ordinary)).decision == PermissionDecision.ALLOW


async def test_find_files_by_name_and_content(tmp_path, root):
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / ".venv").mkdir()
    (project / "tests" / "test_math.py").write_text("def test_add():\n    assert add(1, 2) == 4\n", encoding="utf-8")
    (project / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (project / ".venv" / "test_skip.py").write_text("assert add", encoding="utf-8")
    tool = FindFilesTool(root)
    by_name = await tool.execute(tool.validate_arguments({"directory": str(project), "pattern": "test_*.py"}))
    assert [Path(r["path"]).name for r in by_name.data["results"]] == ["test_math.py"]
    by_content = await tool.execute(tool.validate_arguments({"directory": str(project), "pattern": "*.py", "contains": "return a - b"}))
    assert by_content.data["results"][0]["line"] == 2


async def test_write_creates_then_backs_up_before_overwriting(tmp_path, root):
    backups = tmp_path / "backups"
    tool = WriteTextFileTool(root, backups)
    target = tmp_path / "out" / "report.txt"
    created = await tool.execute(tool.validate_arguments({"path": str(target), "content": "v1"}))
    assert created.success and created.data["created"] and created.data["backup"] is None
    replaced = await tool.execute(tool.validate_arguments({"path": str(target), "content": "v2"}))
    assert target.read_text(encoding="utf-8") == "v2"
    assert Path(replaced.data["backup"]).read_text(encoding="utf-8") == "v1"
    assert (await PermissionChecker().evaluate_call(tool, tool.validate_arguments({"path": str(target), "content": "x"}))).decision \
        == PermissionDecision.ALLOW


async def test_writes_to_system_folders_or_credential_files_need_confirmation(tmp_path, root, monkeypatch):
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    tool = WriteTextFileTool(root, tmp_path / "b")
    system_file = tool.validate_arguments({"path": str(tmp_path / "Windows" / "System32" / "x.ini"), "content": "x"})
    assert (await PermissionChecker().evaluate_call(tool, system_file)).decision == PermissionDecision.REQUIRE_CONFIRMATION
    key_file = tool.validate_arguments({"path": str(tmp_path / "id_rsa"), "content": "x"})
    assert (await PermissionChecker().evaluate_call(tool, key_file)).decision == PermissionDecision.REQUIRE_CONFIRMATION


async def test_edit_replaces_exactly_once_and_keeps_line_endings(tmp_path, root):
    source = tmp_path / "app.py"
    source.write_bytes(b"def add(a, b):\r\n    return a - b\r\n")
    tool = EditTextFileTool(root, tmp_path / "backups")
    result = await tool.execute(tool.validate_arguments({"path": str(source), "old_text": "return a - b", "new_text": "return a + b"}))
    assert result.success and result.data["first_changed_line"] == 2
    assert source.read_bytes() == b"def add(a, b):\r\n    return a + b\r\n"
    assert Path(result.data["backup"]).read_bytes() == b"def add(a, b):\r\n    return a - b\r\n"

    missing = await tool.execute(tool.validate_arguments({"path": str(source), "old_text": "nope", "new_text": "x"}))
    assert not missing.success and "not found" in missing.error
    source.write_text("x = 1\nx = 1\n", encoding="utf-8")
    ambiguous = await tool.execute(tool.validate_arguments({"path": str(source), "old_text": "x = 1", "new_text": "x = 2"}))
    assert not ambiguous.success and "2 times" in ambiguous.error
    every = await tool.execute(tool.validate_arguments({"path": str(source), "old_text": "x = 1", "new_text": "x = 2", "replace_all": True}))
    assert every.data["replacements"] == 2 and source.read_text(encoding="utf-8") == "x = 2\nx = 2\n"
