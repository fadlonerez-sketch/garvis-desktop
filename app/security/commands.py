"""Which PowerShell commands may run without asking the user.

The command text is parsed by PowerShell's own parser (not regexes), which
reports every command invoked anywhere in it -- inside pipelines, script
blocks and sub-expressions -- plus .NET method calls, file redirections
and property assignments. A command runs without confirmation only when
EVERYTHING in it is known to be read-only: cmdlets like Get-*/Select-*/
Where-*, and a short list of external tools with read-only arguments
(git status/log/diff, ipconfig, ping, ...). Test runners (pytest, npm test,
dotnet test, ...) always ask: a test suite runs the project's own code.

Anything else -- unknown programs, writes, deletes, process/service
changes, installs, network uploads, dynamic code, parse errors, or any
reference to credential files -- gets a reason, and the Agent asks the user
first (PermissionChecker.evaluate_call). This only decides confirmation;
nothing is ever silently blocked or silently allowed by the LLM.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.security.sensitive import sensitive_path_reason
from app.system.powershell import PowerShellError, PowerShellRunner, ps_string

_ALIASES = {
    "ls": "get-childitem", "dir": "get-childitem", "gci": "get-childitem", "cat": "get-content",
    "type": "get-content", "gc": "get-content", "ps": "get-process", "gps": "get-process",
    "gsv": "get-service", "select": "select-object", "where": "where-object", "?": "where-object",
    "%": "foreach-object", "foreach": "foreach-object", "sort": "sort-object", "ft": "format-table",
    "fl": "format-list", "fw": "format-wide", "measure": "measure-object", "echo": "write-output",
    "write": "write-output", "sls": "select-string", "cd": "set-location", "chdir": "set-location",
    "sl": "set-location", "pwd": "get-location", "gl": "get-location", "gm": "get-member",
    "group": "group-object", "h": "get-history", "history": "get-history", "gal": "get-alias",
    "gcm": "get-command", "gwmi": "get-wmiobject", "gcim": "get-ciminstance", "sleep": "start-sleep",
    "diff": "compare-object", "compare": "compare-object", "gi": "get-item", "gp": "get-itemproperty",
    "rvpa": "resolve-path", "cls": "clear-host", "clear": "clear-host", "pushd": "push-location",
    "popd": "pop-location", "gdr": "get-psdrive", "gjb": "get-job", "help": "get-help", "man": "get-help",
}
_SAFE_VERB_PREFIXES = (
    "get-", "test-", "measure-", "select-", "where-", "sort-", "group-", "format-", "convertto-",
    "convertfrom-", "compare-", "resolve-", "find-",
)
_SAFE_CMDLETS = {
    "foreach-object", "out-string", "out-host", "out-null", "write-output", "write-host", "start-sleep",
    "split-path", "join-path", "set-location", "push-location", "pop-location", "clear-host",
}
_CREDENTIAL_CMDLETS = {
    "get-credential", "get-clipboard", "get-storedcredential", "get-secret", "get-secretinfo",
    "convertfrom-securestring", "export-clixml",
}
_CREDENTIAL_TEXT = ("key=clear", "cmdkey", "dpapi", "securestring", "vaultcmd", "mimikatz")
_SAFE_STATIC_TYPES = {
    "[math]", "[system.math]", "[datetime]", "[system.datetime]", "[timespan]", "[string]", "[system.string]",
    "[char]", "[convert]", "[system.convert]", "[io.path]", "[system.io.path]", "[guid]", "[version]",
    "[int]", "[double]", "[regex]", "[environment]", "[system.environment]",
}
_SAFE_MEMBERS = {
    "tostring", "trim", "trimstart", "trimend", "split", "replace", "substring", "tolower", "toupper",
    "contains", "startswith", "endswith", "indexof", "lastindexof", "padleft", "padright", "join",
    "gettype", "equals", "compareto", "format", "toshortdatestring", "tolongdatestring",
    "toshorttimestring", "adddays", "addhours", "addminutes", "addseconds", "where", "foreach",
    "containskey", "getenumerator", "tochararray", "normalize", "isnullorempty", "isnullorwhitespace",
    "round", "floor", "ceiling", "abs", "max", "min", "pow", "sqrt", "parse", "tryparse", "frombase64string",
    "getenvironmentvariable", "getfolderpath", "getfilename", "getextension", "getdirectoryname",
    "combine", "getfullpath", "newguid", "now", "today", "matches", "match", "escape",
}


def _strip(arg: str) -> str:
    return arg.strip().strip("'\"")


def _positional(args: list[str]) -> list[str]:
    return [a for a in (_strip(x) for x in args) if a and not a.startswith("-")]


_GIT_UNSAFE_FLAGS = ("--open-files-in-pager", "--output", "--ext-diff", "--exec", "--upload-pack", "--config-env")


def _git(args: list[str]) -> bool:
    words = _positional(args)
    if not words:
        return True
    # -O<pager> (git grep) runs a program; --output writes a file; --ext-diff runs GIT_EXTERNAL_DIFF.
    if any(a.startswith("-O") or a.lower().startswith(_GIT_UNSAFE_FLAGS) for a in (_strip(x) for x in args)):
        return False
    sub, rest = words[0].lower(), [a.lower() for a in args[1:]]
    if sub in {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog", "grep"}:
        return True
    if sub == "branch":
        return not any(a.startswith(("-d", "-m", "-c", "--delete", "--move", "--copy", "--set")) for a in rest)
    if sub == "remote":
        return all(a in ("-v", "--verbose") for a in rest)
    if sub == "stash":
        return [a for a in rest if not a.startswith("-")][:1] == ["list"]
    if sub == "config":
        return any(a in ("--get", "--list", "-l", "--get-all") for a in rest)
    return False


def _python(args: list[str]) -> bool:
    cleaned = [_strip(a) for a in args]
    return cleaned in (["--version"], ["-V"])


# Test runners execute project-controlled code (conftest.py, package.json
# scripts, test assemblies...), so they are never "read-only", whatever the
# folder. Checked before the read-only rules below, by program name after
# path/extension/alias normalization.
_TEST_RUNNER_PROGRAMS = {
    "pytest", "py.test", "pytest-3", "tox", "nox", "nose2", "ward", "jest", "vitest", "mocha", "ava", "karma",
    "playwright", "cypress", "phpunit", "rspec", "ctest", "vstest.console", "invoke-pester",
}
_TEST_SUBCOMMAND_TOOLS = {
    "npm", "pnpm", "yarn", "bun", "deno", "npx", "dotnet", "cargo", "go", "mvn", "mvnw", "gradle", "gradlew",
    "make", "nmake", "mix", "swift", "flutter", "dart",
}
_TEST_SUBCOMMANDS = {"test", "t", "tst", "it", "cit", "install-test", "install-ci-test", "clean-install-test", "vstest"}
_SCRIPT_RUNNERS = {"run", "run-script", "rum", "urn", "exec", "x", "dlx"}
_PYTHON_TEST_MODULES = {"pytest", "unittest", "tox", "nox", "nose2", "doctest", "ward"}


def _is_python(name: str) -> bool:
    return name in ("python", "python3", "py", "pythonw", "pyw") or re.fullmatch(r"python\d+(\.\d+)?", name) is not None


def _runs_tests(name: str, args: list[str]) -> bool:
    if name in _TEST_RUNNER_PROGRAMS:
        return True
    cleaned = [_strip(a).lower() for a in args]
    if _is_python(name):
        for i, arg in enumerate(cleaned):
            module = cleaned[i + 1] if arg == "-m" and i + 1 < len(cleaned) else arg[2:] if arg.startswith("-m") else ""
            if module.split(".")[0] in _PYTHON_TEST_MODULES:
                return True
        return False
    if name in _TEST_SUBCOMMAND_TOOLS:
        # Flag values (`--prefix app test`) are positional to this simple split, so
        # look a few words in; a false match only means one more question.
        words = [a for a in cleaned if not a.startswith("-")][:3]
        if any(w in _TEST_SUBCOMMANDS for w in words):
            return True
        if any(w in _SCRIPT_RUNNERS and any("test" in x for x in words[i + 1:i + 2]) for i, w in enumerate(words)):
            return True
        if name == "npx" and words:
            return words[0] in _TEST_RUNNER_PROGRAMS
    return False


def _subcommand_in(allowed: set[str]) -> Callable[[list[str]], bool]:
    def check(args: list[str]) -> bool:
        words = _positional(args)
        if words:
            return words[0].lower() in allowed
        return "--version" in [_strip(a).lower() for a in args]

    return check


def _flags_in(allowed: set[str]) -> Callable[[list[str]], bool]:
    return lambda args: all(_strip(a).lower() in allowed for a in args)


_EXTERNAL_RULES: dict[str, Callable[[list[str]], bool]] = {
    "git": _git,
    "python": _python, "python3": _python, "py": _python,
    "pip": _subcommand_in({"list", "show", "freeze", "check"}),
    "npm": _subcommand_in({"ls", "list", "outdated", "view"}),
    "node": _flags_in({"--version", "-v"}),
    "dotnet": lambda args: [_strip(a).lower() for a in args][:1] in (["--info"], ["--version"], ["--list-sdks"]),
    "winget": _subcommand_in({"list", "search", "show"}),
    "ipconfig": _flags_in({"/all", "/displaydns"}),
    "sc": _subcommand_in({"query", "queryex", "qc", "qdescription"}),
    "reg": _subcommand_in({"query"}),
    "powercfg": _flags_in({"/list", "/l", "/query", "/q", "/a", "/availablesleepstates", "/getactivescheme"}),
    "netsh": lambda args: "show" in [_strip(a).lower() for a in args],
    **{name: (lambda args: True) for name in (
        "systeminfo", "tasklist", "whoami", "hostname", "where", "nslookup", "ping", "tracert", "pathping",
        "netstat", "getmac", "driverquery", "nvidia-smi", "findstr", "fc", "tree", "ver", "arp",
    )},
}


@dataclass
class ParsedCommand:
    name: str | None
    args: list[str] = field(default_factory=list)


@dataclass
class CommandAnalysis:
    commands: list[ParsedCommand]
    redirections: int = 0
    invocations: list[tuple[bool, str, str]] = field(default_factory=list)
    """(static, type text, member name) for every .NET method call."""
    member_assignments: int = 0
    parse_errors: int = 0


_ANALYZE_SCRIPT = r"""
$src = __SOURCE__
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$tokens, [ref]$errors)
$commands = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) | ForEach-Object {
    [pscustomobject]@{ name = $_.GetCommandName(); args = @($_.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text }) }
})
$redirections = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst] }, $true)).Count
$invocations = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] }, $true) | ForEach-Object {
    [pscustomobject]@{ static = [bool]$_.Static; type = $(if ($_.Static) { $_.Expression.Extent.Text } else { '' }); member = $_.Member.Extent.Text }
})
$assignments = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left -is [System.Management.Automation.Language.MemberExpressionAst] }, $true)).Count
[pscustomobject]@{ commands = $commands; redirections = $redirections; invocations = $invocations; assignments = $assignments; errors = @($errors).Count } | ConvertTo-Json -Depth 5 -Compress
"""


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_analysis(data: dict) -> CommandAnalysis:
    return CommandAnalysis(
        commands=[
            ParsedCommand(name=(c.get("name") or None), args=[str(a) for a in _as_list(c.get("args"))])
            for c in _as_list(data.get("commands"))
        ],
        redirections=int(data.get("redirections") or 0),
        invocations=[
            (bool(i.get("static")), str(i.get("type") or ""), str(i.get("member") or ""))
            for i in _as_list(data.get("invocations"))
        ],
        member_assignments=int(data.get("assignments") or 0),
        parse_errors=int(data.get("errors") or 0),
    )


def _normalize_name(name: str) -> str:
    base = name.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


_ENV_ASSIGNMENT = re.compile(r"\$\{?env:[\w()]+\}?\s*(?:[-+*/%]|\?\?)?=(?!=)", re.IGNORECASE)
_FOREACH_BLOCK_PARAMS = {"-begin", "-process", "-end", "-parallel"}


def _foreach_is_script_blocks(args: list[str]) -> bool:
    """`ForEach-Object { ... }` is checked like any other code (its commands and
    method calls are in the same AST); `ForEach-Object Delete` / `-MemberName Kill`
    calls a method by name, which the AST doesn't show as an invocation."""
    return all(a.strip().startswith("{") or a.strip().lower() in _FOREACH_BLOCK_PARAMS for a in args)


def classify(command: str, analysis: CommandAnalysis) -> str | None:
    """None when the command is read-only; otherwise why it needs confirmation."""
    lowered = command.lower()
    if any(marker in lowered for marker in _CREDENTIAL_TEXT):
        return "It may reveal stored credentials."
    for parsed in analysis.commands:
        for arg in parsed.args:
            reason = sensitive_path_reason(_strip(arg)) if any(ch in arg for ch in "\\/.") else None
            if reason:
                return f"It touches a credential file: {reason}"
    if analysis.parse_errors:
        return "Garvis couldn't fully parse this command."
    if analysis.redirections:
        return "It writes output to a file (redirection)."
    if analysis.member_assignments:
        return "It changes an object's properties."
    if _ENV_ASSIGNMENT.search(command):
        return "It changes an environment variable (which can change what later commands run)."
    for static, type_text, member in analysis.invocations:
        if static and type_text.lower() not in _SAFE_STATIC_TYPES:
            return f"It calls the .NET method {type_text}::{member}."
        if member.lower() not in _SAFE_MEMBERS:
            return f"It calls the method .{member}(), which may change something."
    if not analysis.commands and not command.strip():
        return "The command is empty."
    for parsed in analysis.commands:
        if parsed.name is None:
            return "It runs a command chosen at runtime (dynamic invocation)."
        name = _ALIASES.get(parsed.name.lower(), parsed.name.lower())
        if name in _CREDENTIAL_CMDLETS:
            return f"'{parsed.name}' can expose credentials or clipboard contents."
        if _runs_tests(_normalize_name(name), parsed.args):
            return (f"'{parsed.name}' runs a test suite, which executes the project's own code "
                    "(approve it only for a project you trust).")
        if name == "foreach-object" and not _foreach_is_script_blocks(parsed.args):
            return "ForEach-Object with a member name calls a method on every item, which may change something."
        rule = _EXTERNAL_RULES.get(_normalize_name(name))
        if rule is not None:
            if not rule(parsed.args):
                return f"'{parsed.name} {' '.join(parsed.args)[:60]}' may change something."
            continue
        if name in _SAFE_CMDLETS or ("-" in name and name.startswith(_SAFE_VERB_PREFIXES)):
            continue
        if "-" in name and "\\" not in name and "." not in name:
            return f"'{parsed.name}' can change the system."
        return f"It runs '{parsed.name}', which Garvis can't verify is read-only."
    return None


class CommandPolicy:
    def __init__(self, runner: PowerShellRunner | None = None) -> None:
        self._runner = runner or PowerShellRunner()
        self._cache: dict[str, CommandAnalysis] = {}

    async def analyze(self, command: str) -> CommandAnalysis:
        cached = self._cache.get(command)
        if cached is not None:
            return cached
        script = _ANALYZE_SCRIPT.replace("__SOURCE__", ps_string(command))
        data = await self._runner.run_json(script, timeout_seconds=20)
        if not isinstance(data, dict):
            raise PowerShellError("The command parser returned nothing.")
        analysis = parse_analysis(data)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[command] = analysis
        return analysis

    async def confirmation_reason(self, command: str) -> str | None:
        try:
            analysis = await self.analyze(command)
        except PowerShellError as exc:
            return f"Garvis couldn't analyze the command ({exc})."
        return classify(command, analysis)
