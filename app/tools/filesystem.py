"""File tools: list, read, find, write and edit files on this PC.

Paths: an absolute path (C:\\..., ~\\..., %USERPROFILE%\\...) means exactly
that file; a relative path is resolved inside the Garvis files folder
(~/Documents/Garvis) and may not escape it.

Safety, decided by code rather than the LLM:
  - Credential stores (SSH keys, .env files, browser password databases,
    key/certificate files -- app/security/sensitive.py) are never read or
    written silently: those calls need the user's confirmation, and any
    output is still passed through redact_secrets().
  - Writes inside Windows/Program Files/ProgramData need confirmation, and so
    do writes that could make something run later: executable/launcher file
    types (.exe, .bat, .ps1, .lnk, ...), the Startup folder, git config and
    hooks, and Garvis's own code and configuration.
  - Network (UNC) paths are refused before they are resolved.
  - Every overwrite or edit of an existing file first copies it to the
    backup folder (Settings.file_backup_dir), so changes are reversible;
    the result says where the backup is.
  - There is no delete tool; deleting goes through run_powershell, which
    always asks first.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field

from app.security.sensitive import redact_secrets, sensitive_path_reason
from app.tools.base import RiskLevel, Tool, ToolResult

_DEFAULT_ROOT = Path.home() / "Documents" / "Garvis"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH_SUFFIXES = frozenset(
    ".exe .com .bat .cmd .ps1 .psm1 .psd1 .vbs .vbe .wsf .wsh .wsc .hta .lnk .url .reg .scr .pif .msi .msp "
    ".cpl .jar .scf .inf .application .appref-ms .msc .settingcontent-ms".split()
)
_AUTORUN_NAMES = frozenset({"sitecustomize.py", "usercustomize.py", ".bashrc", ".bash_profile", ".bash_login",
                            ".profile", ".zshrc", ".zprofile"})
_DEFAULT_BACKUP_DIR = Path(__file__).resolve().parents[2] / "data" / "file_backups"

_MAX_READ_BYTES = 2_000_000
_DEFAULT_MAX_LINES = 400
_MAX_WRITE_CHARS = 1_000_000
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build"}
_PATH_DESCRIPTION = "Absolute path (e.g. C:\\Users\\me\\project\\app.py), or a path relative to the Garvis files folder"


def _protected_roots() -> list[Path]:
    roots = []
    for var in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value).resolve())
    return roots


def _resolve_within_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError(f"Path '{relative}' escapes the allowed directory; use an absolute path instead")
    return candidate


def resolve_path(root: Path, raw: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw.strip().strip('"')))
    if expanded.startswith(("\\\\", "//")):
        # Checked before resolve(): resolving a UNC path already contacts that
        # server (SMB/WebDAV), which can leak the user's NTLM credentials.
        raise ValueError("Network (UNC) paths are not supported by the file tools")
    if Path(expanded).is_absolute():
        return Path(expanded).resolve()
    return _resolve_within_root(root, expanded)


def _autostart_dirs() -> list[Path]:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []
    return [(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup").resolve()]


def _write_reason(path: Path) -> str | None:
    reason = sensitive_path_reason(path)
    if reason:
        return f"Writing a credential file: {reason}"
    for protected in _protected_roots():
        if path == protected or protected in path.parents:
            return f"'{path}' is inside a system folder ({protected})."
    # Files that run something when launched, at logon or by git: a silent
    # write followed by any launch would otherwise be code execution.
    if path.suffix.lower() in _LAUNCH_SUFFIXES:
        return f"'{path.name}' is a {path.suffix.lower()} file, which runs something when it is opened."
    for autostart in _autostart_dirs():
        if autostart in path.parents:
            return f"'{path}' is in the Startup folder (runs at logon)."
    if ".git" in path.parts or path.name.lower() == ".gitconfig":
        return f"'{path}' is git configuration or a git hook, which can run commands."
    # Python runs .pth files and site/usercustomize from site-packages at every
    # start (Garvis's own included); shells run their startup files.
    if (path.suffix.lower() == ".pth" or path.name.lower() in _AUTORUN_NAMES
            or any(part.lower() in ("site-packages", "dist-packages") for part in path.parts)):
        return f"'{path}' is run automatically by Python or a shell at startup."
    if _PROJECT_ROOT in path.parents and _PROJECT_ROOT / "data" not in path.parents:
        return f"'{path}' is part of Garvis's own code or configuration."
    return None


def _is_probably_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return False
    return b"\x00" not in chunk


class _FileTool(Tool):
    def __init__(self, root: Path = _DEFAULT_ROOT, backup_dir: Path = _DEFAULT_BACKUP_DIR) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._backup_dir = backup_dir

    def _resolve(self, raw: str) -> Path:
        return resolve_path(self._root, raw)

    def _backup(self, path: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        relative = Path(*[part.replace(":", "") for part in path.parts])
        target = self._backup_dir / stamp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return target


class ListDirectoryInput(BaseModel):
    relative_path: str = Field(
        default=".", validation_alias=AliasChoices("path", "relative_path"), description=_PATH_DESCRIPTION
    )


class ListDirectoryTool(_FileTool):
    name = "list_directory"
    description = "List the files and folders in a directory (name, folder or not, size, modified time)."
    risk_level = RiskLevel.LOW
    input_model = ListDirectoryInput

    async def execute(self, args: ListDirectoryInput) -> ToolResult:
        try:
            target = self._resolve(args.relative_path)
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        if not target.exists():
            return ToolResult.fail(f"'{args.relative_path}' does not exist")
        if not target.is_dir():
            return ToolResult.fail(f"'{args.relative_path}' is not a directory")
        entries = []
        try:
            for p in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:500]:
                try:
                    stat = p.stat()
                except OSError:
                    continue
                entries.append({
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size_bytes": stat.st_size if p.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes"),
                })
        except OSError as exc:
            return ToolResult.fail(f"Could not list '{target}': {exc}")
        return ToolResult.ok(entries)


class ReadTextFileInput(BaseModel):
    relative_path: str = Field(validation_alias=AliasChoices("path", "relative_path"), description=_PATH_DESCRIPTION)
    start_line: int = Field(default=1, ge=1, description="First line to return (1-based)")
    max_lines: int = Field(default=_DEFAULT_MAX_LINES, ge=1, le=5000, description="How many lines to return")


class ReadTextFileTool(_FileTool):
    name = "read_text_file"
    description = (
        "Read a text file (source code, logs, configs). Returns up to max_lines lines starting at "
        "start_line, plus the total line count -- read long files in parts (e.g. the end of a log)."
    )
    risk_level = RiskLevel.LOW
    input_model = ReadTextFileInput

    async def confirmation_reason(self, args: ReadTextFileInput) -> str | None:
        try:
            reason = sensitive_path_reason(self._resolve(args.relative_path))
        except ValueError:
            return None
        return f"Reading a credential file: {reason}" if reason else None

    async def execute(self, args: ReadTextFileInput) -> ToolResult:
        try:
            target = self._resolve(args.relative_path)
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        if not target.exists() or not target.is_file():
            return ToolResult.fail(f"'{args.relative_path}' is not a readable file")
        size = target.stat().st_size
        if size > _MAX_READ_BYTES:
            return ToolResult.fail(f"'{target}' is too large to read ({size} bytes); use find_files or run_powershell")
        if not _is_probably_text(target):
            return ToolResult.fail(f"'{target}' looks like a binary file, not text")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.fail(f"Could not read '{args.relative_path}': {exc}")
        lines = content.splitlines()
        start = args.start_line - 1
        selected = lines[start:start + args.max_lines]
        return ToolResult.ok({
            "path": str(target),
            "content": redact_secrets("\n".join(selected)),
            "start_line": args.start_line,
            "end_line": start + len(selected),
            "total_lines": len(lines),
        })


class FindFilesInput(BaseModel):
    directory: str = Field(description=_PATH_DESCRIPTION)
    pattern: str = Field(default="*", description="File-name glob, e.g. '*.py', 'test_*.py', '*.log'")
    contains: str | None = Field(default=None, description="Only text files containing this text (case-insensitive)")
    max_results: int = Field(default=50, ge=1, le=500)


class FindFilesTool(_FileTool):
    name = "find_files"
    description = (
        "Search a folder tree for files by name pattern and optionally by content; returns paths "
        "(and the first matching line for content searches). Skips .git, node_modules, venvs."
    )
    risk_level = RiskLevel.LOW
    input_model = FindFilesInput

    async def execute(self, args: FindFilesInput) -> ToolResult:
        import asyncio
        import fnmatch

        try:
            base = self._resolve(args.directory)
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        if not base.is_dir():
            return ToolResult.fail(f"'{args.directory}' is not a directory")
        needle = args.contains.lower() if args.contains else None

        def search() -> tuple[list[dict], bool]:
            results: list[dict] = []
            scanned = 0
            for current, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
                for name in files:
                    if not fnmatch.fnmatch(name.lower(), args.pattern.lower()):
                        continue
                    path = Path(current) / name
                    scanned += 1
                    if scanned > 20000:
                        return results, True
                    if needle is None:
                        results.append({"path": str(path)})
                    elif not sensitive_path_reason(path) and path.stat().st_size < 1_000_000 and _is_probably_text(path):
                        try:
                            with path.open(encoding="utf-8", errors="replace") as handle:
                                for number, line in enumerate(handle, 1):
                                    if needle in line.lower():
                                        results.append({"path": str(path), "line": number,
                                                        "text": redact_secrets(line.strip()[:200])})
                                        break
                        except OSError:
                            continue
                    if len(results) >= args.max_results:
                        return results, True
            return results, False

        results, truncated = await asyncio.to_thread(search)
        return ToolResult.ok({"count": len(results), "truncated": truncated, "results": results})


class WriteTextFileInput(BaseModel):
    path: str = Field(description=_PATH_DESCRIPTION)
    content: str = Field(description="The complete new file content")


class WriteTextFileTool(_FileTool):
    name = "write_text_file"
    description = (
        "Create a text file, or replace an existing one's entire content (the old version is backed up "
        "first). For small changes to an existing file prefer edit_text_file."
    )
    risk_level = RiskLevel.MEDIUM
    input_model = WriteTextFileInput

    async def confirmation_reason(self, args: WriteTextFileInput) -> str | None:
        try:
            return _write_reason(self._resolve(args.path))
        except ValueError:
            return None

    async def execute(self, args: WriteTextFileInput) -> ToolResult:
        if len(args.content) > _MAX_WRITE_CHARS:
            return ToolResult.fail("Content is too large (over 1 MB).")
        try:
            target = self._resolve(args.path)
            if target.exists() and not target.is_file():
                return ToolResult.fail(f"'{target}' is not a file")
            backup = self._backup(target) if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args.content, encoding="utf-8")
        except (OSError, ValueError) as exc:
            return ToolResult.fail(f"Could not write '{args.path}': {exc}")
        return ToolResult.ok({
            "path": str(target), "created": backup is None, "bytes": target.stat().st_size,
            "backup": str(backup) if backup else None,
        })


class EditTextFileInput(BaseModel):
    path: str = Field(description=_PATH_DESCRIPTION)
    old_text: str = Field(description="Exact text to replace (include enough context to be unique)")
    new_text: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of exactly one")


class EditTextFileTool(_FileTool):
    name = "edit_text_file"
    description = (
        "Replace an exact piece of text in an existing file (e.g. fix a line of code). Fails if the text "
        "isn't found or isn't unique. The original is backed up first."
    )
    risk_level = RiskLevel.MEDIUM
    input_model = EditTextFileInput

    async def confirmation_reason(self, args: EditTextFileInput) -> str | None:
        try:
            return _write_reason(self._resolve(args.path))
        except ValueError:
            return None

    async def execute(self, args: EditTextFileInput) -> ToolResult:
        try:
            target = self._resolve(args.path)
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        if not target.is_file():
            return ToolResult.fail(f"'{args.path}' is not an existing file")
        try:
            raw = target.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.fail(f"Could not read '{target}' as UTF-8 text: {exc}")
        newline = "\r\n" if "\r\n" in text else "\n"
        old = args.old_text.replace("\r\n", "\n").replace("\n", newline)
        new = args.new_text.replace("\r\n", "\n").replace("\n", newline)
        count = text.count(old) if old else 0
        if count == 0:
            return ToolResult.fail("old_text was not found in the file; read the file again and copy it exactly")
        if count > 1 and not args.replace_all:
            return ToolResult.fail(f"old_text occurs {count} times; add more context or set replace_all")
        updated = text.replace(old, new) if args.replace_all else text.replace(old, new, 1)
        try:
            backup = self._backup(target)
            with target.open("w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
        except OSError as exc:
            return ToolResult.fail(f"Could not write '{target}': {exc}")
        first_line = text[: text.index(old)].count(newline) + 1
        return ToolResult.ok({
            "path": str(target), "replacements": count if args.replace_all else 1,
            "first_changed_line": first_line, "backup": str(backup),
        })


def build_file_tools(root: Path = _DEFAULT_ROOT, backup_dir: Path = _DEFAULT_BACKUP_DIR) -> list[Tool]:
    return [
        ListDirectoryTool(root, backup_dir),
        ReadTextFileTool(root, backup_dir),
        FindFilesTool(root, backup_dir),
        WriteTextFileTool(root, backup_dir),
        EditTextFileTool(root, backup_dir),
    ]
