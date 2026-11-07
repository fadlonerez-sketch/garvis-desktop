"""Safe autonomy: which calls run directly and which need the user's
confirmation -- decided by deterministic code, never by the LLM."""

from __future__ import annotations

import asyncio
import sys

import pytest
from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.security.commands import CommandAnalysis, CommandPolicy, ParsedCommand, classify
from app.security.confirmations import ConfirmationManager
from app.security.permissions import PermissionChecker, PermissionDecision
from app.security.sensitive import consequential_reason, redact_secrets, sensitive_path_reason
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.browser import ClickTool, PressKeyTool, SubmitFormTool
from app.tools.registry import ToolRegistry
from tests.conftest import final_response, tool_call_response


# ------------------------------------------------------------ labels / paths
@pytest.mark.parametrize(
    "label",
    [
        "Send",
        "Send now",
        "btn-send",
        "#sendButton",
        "Delete",
        "Uninstall",
        "Install updates",
        "Place your order",
        "Buy now",
        "Pay",
        "Publish",
        "Post",
        "Empty Recycle Bin",
        "button[type=submit]",
        "Discard",
        "שלח",
        "מחק",
    ],
)
def test_committing_labels_are_recognized(label):
    assert consequential_reason(label)


@pytest.mark.parametrize(
    "label",
    [
        "Open",
        "Save",
        "Search",
        "Sender",
        "Deleted Items",
        "Posts",
        "Sign in",
        "Accept all cookies",
        "Find and Replace",
        "Format",
        "Next page",
        "a.next-page",
        "Settings",
        "",
        None,
    ],
)
def test_navigation_labels_are_not_committing(label):
    assert consequential_reason(label) is None


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\me\.ssh\id_ed25519",
        r"C:\proj\.env",
        r"C:\proj\.env.local",
        r"C:\certs\server.pem",
        r"C:\Users\me\AppData\Local\Google\Chrome\User Data\Default\Login Data",
        r"C:\Users\me\.aws\credentials",
        r"C:\vault\passwords.kdbx",
        r"C:\Users\me\.ssh",
    ],
)
def test_credential_paths_are_sensitive(path):
    assert sensitive_path_reason(path)


@pytest.mark.parametrize("path", [r"C:\proj\app.py", r"C:\proj\README.md", r"C:\logs\app.log", r"C:\proj\keys.txt"])
def test_ordinary_paths_are_not_sensitive(path):
    assert sensitive_path_reason(path) is None


def test_redaction_masks_secrets_but_keeps_ordinary_text():
    text = (
        "API_KEY=abc123\nkey AIzaSyA1234567890123456789012345678901 ok\npassword: hunter2\n"
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----\n"
        "build passed in 3.2s"
    )
    out = redact_secrets(text)
    for secret in ("abc123", "AIzaSyA123", "hunter2", "ghp_abc", "MIIE"):
        assert secret not in out
    assert "build passed in 3.2s" in out


# --------------------------------------------------------- per-call checks
class _NoArgs(BaseModel):
    pass


class _EscalatingTool(Tool):
    name = "escalating"
    description = "test"
    risk_level = RiskLevel.MEDIUM
    input_model = _NoArgs

    def __init__(self, reason=None, raises=False):
        self.reason = reason
        self.raises = raises
        self.runs = 0

    async def confirmation_reason(self, args):
        if self.raises:
            raise RuntimeError("probe failed")
        return self.reason

    async def execute(self, args):
        self.runs += 1
        return ToolResult.ok("ran")


async def test_per_call_reason_escalates_allow_to_confirmation():
    checker = PermissionChecker()
    assert (await checker.evaluate_call(_EscalatingTool(), _NoArgs())).decision == PermissionDecision.ALLOW
    result = await checker.evaluate_call(_EscalatingTool("it sends"), _NoArgs())
    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION and result.reason == "it sends"


async def test_a_failing_per_call_check_fails_closed():
    result = await PermissionChecker().evaluate_call(_EscalatingTool(raises=True), _NoArgs())
    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION


async def test_per_call_check_can_never_relax_high_risk():
    tool = _EscalatingTool(None)
    tool.risk_level = RiskLevel.HIGH
    assert (
        await PermissionChecker().evaluate_call(tool, _NoArgs())
    ).decision == PermissionDecision.REQUIRE_CONFIRMATION


async def _run_agent_with(tool, decide):
    events = EventBus()
    seen: list[AgentEvent] = []
    events.subscribe(seen.append)
    registry = ToolRegistry()
    registry.register(tool)

    class _LLM:
        def __init__(self):
            self.responses = [tool_call_response(tool.name, {}), final_response("done")]

        async def chat(self, messages, tools=None):
            return self.responses.pop(0)

    import tempfile
    from pathlib import Path

    agent = Agent(
        llm=_LLM(),
        registry=registry,
        memory=SQLiteMemoryStore(Path(tempfile.mkdtemp()) / "m.db"),
        events=events,
        confirmations=ConfirmationManager(default_timeout_seconds=2),
    )
    turn = asyncio.create_task(agent.handle_user_message("go"))
    if decide is not None:
        while not any(e.type == EventType.CONFIRMATION_REQUESTED for e in seen):
            await asyncio.sleep(0.01)
        request = next(e for e in seen if e.type == EventType.CONFIRMATION_REQUESTED)
        agent.confirmations.resolve(request.payload["id"], decide)
    await turn
    return seen


async def test_agent_asks_only_for_calls_the_tool_flags():
    routine = _EscalatingTool(None)
    seen = await _run_agent_with(routine, decide=None)
    assert routine.runs == 1 and not any(e.type == EventType.CONFIRMATION_REQUESTED for e in seen)

    committing = _EscalatingTool("'Send' looks like a committing action (send).")
    seen = await _run_agent_with(committing, decide=False)
    request = next(e for e in seen if e.type == EventType.CONFIRMATION_REQUESTED)
    assert "Send" in request.payload["reason"]
    assert committing.runs == 0  # declined -> never executed


# -------------------------------------------------------------- browser
class _Browser:
    def __init__(self, label="", form=""):
        self.label = label
        self.form = form

    async def describe_element(self, selector):
        return self.label

    async def describe_focused_form(self):
        return self.form


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Place your order", PermissionDecision.REQUIRE_CONFIRMATION),
        ("Send | send-btn", PermissionDecision.REQUIRE_CONFIRMATION),
        ("Next page", PermissionDecision.ALLOW),
        ("Search", PermissionDecision.ALLOW),
    ],
)
async def test_browser_click_is_judged_by_the_real_element_label(label, expected):
    tool = ClickTool(_Browser(label=label))
    decision = await PermissionChecker().evaluate_call(tool, tool.validate_arguments({"selector": "#b1"}))
    assert decision.decision == expected


@pytest.mark.parametrize(
    "key, form, expected",
    [
        ("Enter", "Place order | Back", PermissionDecision.REQUIRE_CONFIRMATION),
        ("Enter", "Google Search | I'm Feeling Lucky", PermissionDecision.ALLOW),
        ("Enter", "", PermissionDecision.ALLOW),
        ("Tab", "Place order", PermissionDecision.ALLOW),
    ],
)
async def test_browser_enter_is_judged_by_the_form_it_submits(key, form, expected):
    tool = PressKeyTool(_Browser(form=form))
    decision = await PermissionChecker().evaluate_call(tool, tool.validate_arguments({"key": key}))
    assert decision.decision == expected


async def test_browser_submit_form_still_always_confirms():
    tool = SubmitFormTool(_Browser(label="Search"))
    decision = await PermissionChecker().evaluate_call(tool, tool.validate_arguments({"selector": "#go"}))
    assert decision.decision == PermissionDecision.REQUIRE_CONFIRMATION


# ------------------------------------------------------- command policy
def _analysis(*commands, **extra) -> CommandAnalysis:
    return CommandAnalysis(commands=[ParsedCommand(name, list(args)) for name, *args in commands], **extra)


@pytest.mark.parametrize(
    "command, analysis",
    [
        ("Get-Process | Sort-Object CPU", _analysis(("Get-Process",), ("Sort-Object", "CPU"))),
        ("git status", _analysis(("git", "status"))),
        ("git log -5", _analysis(("git", "log", "-5"))),
        ("ipconfig /all", _analysis(("ipconfig", "/all"))),
        ("nvidia-smi", _analysis(("nvidia-smi",))),
        ("ls | % { $_.Name }", _analysis(("ls",), ("%", "{ $_.Name }"), invocations=[(False, "", "ToString")])),
        ("[math]::Round(1.5)", _analysis(invocations=[(True, "[math]", "Round")])),
    ],
)
def test_read_only_commands_run_without_confirmation(command, analysis):
    assert classify(command, analysis) is None


@pytest.mark.parametrize(
    "command, analysis, fragment",
    [
        ("Remove-Item x", _analysis(("Remove-Item", "x")), "change the system"),
        ("Stop-Process -Name x", _analysis(("Stop-Process", "-Name", "x")), "change the system"),
        ("git push", _analysis(("git", "push")), "may change"),
        ("git branch -D main", _analysis(("git", "branch", "-D", "main")), "may change"),
        ("winget install x", _analysis(("winget", "install", "x")), "may change"),
        ("python -m pytest -q", _analysis(("python", "-m", "pytest", "-q")), "test suite"),
        ("mytool.exe --go", _analysis(("mytool.exe", "--go")), "can't verify"),
        (r".\ping.exe 1.1.1.1", _analysis((r".\ping.exe", "1.1.1.1")), "from a path"),
        (r"C:\Users\me\Downloads\whoami.bat", _analysis((r"C:\Users\me\Downloads\whoami.bat",)), "from a path"),
        ("& $x", _analysis((None,)), "dynamic"),
        ("Get-Process > p.txt", _analysis(("Get-Process",), redirections=1), "redirection"),
        ("(gi x).Attributes = 'Hidden'", _analysis(("gi", "x"), member_assignments=1), "properties"),
        ("[IO.File]::Delete('x')", _analysis(invocations=[(True, "[IO.File]", "Delete")]), ".NET method"),
        ("(ps x).Kill()", _analysis(("ps", "x"), invocations=[(False, "", "Kill")]), ".Kill()"),
        ("Get-Content C:\\p\\.env", _analysis(("Get-Content", "C:\\p\\.env")), "credential file"),
        ("Get-Clipboard", _analysis(("Get-Clipboard",)), "credentials or clipboard"),
        (
            "netsh wlan show profile x key=clear",
            _analysis(("netsh", "wlan", "show", "profile", "x", "key=clear")),
            "credentials",
        ),
        ("Get-Process |", _analysis(("Get-Process",), parse_errors=1), "parse"),
    ],
)
def test_commands_that_change_things_or_touch_secrets_need_confirmation(command, analysis, fragment):
    reason = classify(command, analysis)
    assert reason and fragment.lower() in reason.lower()


class _BrokenRunner:
    async def run_json(self, script, timeout_seconds=60):
        from app.system.powershell import PowerShellError

        raise PowerShellError("powershell missing")


async def test_command_policy_fails_closed_when_it_cannot_analyze():
    assert "couldn't analyze" in (await CommandPolicy(_BrokenRunner()).confirmation_reason("Get-Date"))


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell")
@pytest.mark.parametrize(
    "command, safe",
    [
        ("Get-ChildItem C:\\Windows | Where-Object { $_.Length -gt 1MB } | Measure-Object", True),
        ("cd C:\\; git status", True),
        ('Get-Date; "$(Remove-Item x)"', False),  # hidden in a sub-expression
        ("Get-ChildItem | ForEach-Object { $_.Delete() }", False),
        ("Get-Process | Stop-Process", False),
    ],
)
async def test_real_powershell_parser_finds_every_command(command, safe):
    reason = await CommandPolicy().confirmation_reason(command)
    assert (reason is None) is safe, reason
