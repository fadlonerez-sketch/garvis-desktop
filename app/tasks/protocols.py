"""Protocols: saved, reusable sequences of desktop actions.

A protocol is data, never code. Each step names one entry of the explicit
ACTIONS allowlist below; every action is either a registered tool (run
through Agent.run_tool, i.e. the Agent's own validation, PermissionChecker
and confirmation path) or one of four small built-ins (wait, notify,
confirm, ask Garvis). There is no shell, script or "run anything" step.

Storage: one JSON file per protocol in <data_dir>/protocols/<id>.json,
written atomically. A file that can't be read or doesn't validate is
reported as invalid (with the reason) and is never run; the others load
normally.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.logging import get_logger, log_extra
from app.tools.open_path import expand, refusal

logger = get_logger("tasks.protocols")

SCHEMA_VERSION = 1
MAX_STEPS = 40
MAX_URLS = 10
_ID_RE = re.compile(r"^[a-f0-9]{12,32}$")
_APP_RE = re.compile(r"^[\w .+&'()-]{1,80}$")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------
# The action allowlist
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # text | longtext | url | urls | path | app | window | seconds | choice
    required: bool = True
    default: Any = None
    help: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Action:
    key: str
    label: str
    description: str
    tool: str | None
    """The registered tool this action runs, or None for a built-in."""
    params: tuple[Param, ...]
    presentation: str = "none"
    """Default presentation of the step's result (a step may override)."""
    timeout: float = 60.0
    icon: str = "step"
    unavailable_reason: str = "Not available in this configuration."


_WHERE = Param(
    "where",
    "Where",
    "choice",
    required=False,
    default="user_desktop",
    choices=(("user_desktop", "On my screen"), ("garvis_workspace", "In Garvis's workspace (background)")),
    help="Background work never pulls your attention.",
)

ACTIONS: dict[str, Action] = {
    a.key: a
    for a in (
        Action(
            "open_application",
            "Open an application",
            "Start an installed app by its name.",
            "open_application",
            (
                Param(
                    "application", "Application", "app", help="As you'd find it in Start, e.g. 'Visual Studio Code'."
                ),
                _WHERE,
            ),
            presentation="prominent",
            timeout=60.0,
            icon="app",
        ),
        Action(
            "open_website",
            "Open a web page",
            "Open a page in your own browser.",
            "open_website",
            (Param("url", "Address", "url", help="A full http(s) address."),),
            presentation="prominent",
            timeout=45.0,
            icon="web",
        ),
        Action(
            "open_websites",
            "Open several web pages",
            "Open a set of pages in your browser, one after another.",
            "open_website",
            (Param("urls", "Addresses", "urls", help=f"One http(s) address per line, up to {MAX_URLS}."),),
            presentation="prominent",
            timeout=45.0,
            icon="web",
        ),
        Action(
            "open_path",
            "Open a folder or file",
            "Show a folder in File Explorer or open a document.",
            "open_path",
            (
                Param(
                    "path",
                    "Folder or file",
                    "path",
                    help="An existing folder or document. Programs and scripts are never opened.",
                ),
            ),
            presentation="prominent",
            timeout=30.0,
            icon="folder",
        ),
        Action(
            "focus_window",
            "Bring a window forward",
            "Switch to an open window by its title or app name.",
            "focus_window",
            (Param("window", "Window", "window", help="e.g. 'chrome' or part of the title."),),
            presentation="none",
            timeout=30.0,
            icon="window",
            unavailable_reason="Needs desktop control, which is turned off in Garvis's .env.",
        ),
        Action(
            "wait_for_window",
            "Wait for a window",
            "Wait until a window appears (or give up).",
            "wait_for_window",
            (
                Param("window", "Window", "window"),
                Param(
                    "timeout_seconds",
                    "Give up after (s)",
                    "seconds",
                    required=False,
                    default=15,
                    minimum=1,
                    maximum=120,
                ),
            ),
            presentation="none",
            timeout=130.0,
            icon="wait",
            unavailable_reason="Needs desktop control, which is turned off in Garvis's .env.",
        ),
        Action(
            "wait",
            "Pause",
            "Wait a fixed time before the next step.",
            None,
            (Param("seconds", "Seconds", "seconds", default=3, minimum=1, maximum=600),),
            timeout=610.0,
            icon="wait",
        ),
        Action(
            "notify",
            "Notify me",
            "Show a card on the desktop (and in the task on the phone).",
            None,
            (Param("title", "Title", "text"), Param("message", "Message", "longtext", required=False, default="")),
            presentation="prominent",
            timeout=5.0,
            icon="notify",
        ),
        Action(
            "confirm",
            "Ask me first",
            "Pause until you approve on the PC or the phone; stop if you decline.",
            None,
            (Param("message", "Question", "longtext", help="What you're approving, e.g. 'Open the work apps?'"),),
            timeout=130.0,
            icon="confirm",
        ),
        Action(
            "ask_garvis",
            "Ask Garvis",
            "Give Garvis an instruction in your own words; it uses its tools (and the AI model) like a chat request.",
            None,
            (Param("instruction", "Instruction", "longtext"),),
            timeout=300.0,
            icon="ai",
        ),
    )
}


def action_available(action: Action, tools: set[str] | None) -> bool:
    return action.tool is None or tools is None or action.tool in tools


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Step(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = Field(default="", max_length=80)
    present: Literal["default", "none", "subtle", "prominent"] = "default"
    on_failure: Literal["stop", "continue"] = "stop"
    enabled: bool = True

    def presentation(self) -> str:
        action = ACTIONS.get(self.type)
        return self.present if self.present != "default" else (action.presentation if action else "none")


class Protocol(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=new_id)
    name: str = Field(default="Untitled protocol", max_length=80)
    description: str = Field(default="", max_length=500)
    enabled: bool = True
    steps: list[Step] = Field(default_factory=list, max_length=MAX_STEPS)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("protocol ids are 12-32 lowercase hex characters")
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value > SCHEMA_VERSION:
            raise ValueError(f"saved by a newer Garvis (schema {value}); this one reads up to {SCHEMA_VERSION}")
        return value


@dataclass(frozen=True)
class Problem:
    message: str
    step: int | None = None  # 0-based index, None = the protocol itself
    blocking: bool = True

    def text(self) -> str:
        return (f"Step {self.step + 1}: " if self.step is not None else "") + self.message


def clean_params(step: Step) -> dict[str, Any]:
    """The step's parameters as the action defines them: defaults filled
    in, unknown keys dropped, values trimmed. Raises ValueError (user-facing)."""
    action = ACTIONS[step.type]
    out: dict[str, Any] = {}
    for param in action.params:
        value = step.params.get(param.name, param.default)
        if isinstance(value, str):
            value = value.strip()
        if param.kind == "urls":
            if isinstance(value, str):
                value = [line.strip() for line in value.splitlines() if line.strip()]
            value = [str(v).strip() for v in (value or []) if str(v).strip()]
        missing = value in (None, "", [])
        if missing:
            if param.required:
                raise ValueError(f"{param.label} is required.")
            out[param.name] = param.default if param.default is not None else ""
            continue
        if param.kind == "url":
            _check_url(value)
        elif param.kind == "urls":
            if len(value) > MAX_URLS:
                raise ValueError(f"At most {MAX_URLS} addresses per step.")
            for url in value:
                _check_url(url)
        elif param.kind == "app":
            if not _APP_RE.match(value):
                raise ValueError("Use the application's name (letters, numbers, spaces), not a path or command.")
        elif param.kind in ("window", "text"):
            if len(value) > 120:
                raise ValueError(f"{param.label} is too long.")
        elif param.kind == "longtext":
            if len(value) > 2000:
                raise ValueError(f"{param.label} is too long (2000 characters at most).")
        elif param.kind == "seconds":
            try:
                value = float(value)
            except TypeError, ValueError:
                raise ValueError(f"{param.label} must be a number.") from None
            if (param.minimum is not None and value < param.minimum) or (
                param.maximum is not None and value > param.maximum
            ):
                raise ValueError(f"{param.label} must be between {param.minimum:g} and {param.maximum:g}.")
        elif param.kind == "choice":
            if value not in {key for key, _ in param.choices}:
                raise ValueError(f"{param.label}: unknown option '{value}'.")
        elif param.kind == "path":
            if len(value) > 400:
                raise ValueError("The path is too long.")
        out[param.name] = value
    return out


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"'{url}' is not a full http(s) address.")


def validate(protocol: Protocol, tools: set[str] | None = None) -> list[Problem]:
    """What's wrong with a protocol before it runs. Blocking problems stop a
    run; non-blocking ones (a file that isn't there right now) are shown."""
    problems: list[Problem] = []
    if not protocol.name.strip():
        problems.append(Problem("Give the protocol a name."))
    enabled = [s for s in protocol.steps if s.enabled]
    if not enabled:
        problems.append(Problem("Add at least one enabled step."))
    for index, step in enumerate(protocol.steps):
        action = ACTIONS.get(step.type)
        if action is None:
            problems.append(Problem(f"Unknown action '{step.type}' — it can't run.", index))
            continue
        if not step.enabled:
            continue
        if not action_available(action, tools):
            problems.append(Problem(action.unavailable_reason, index))
        try:
            params = clean_params(step)
        except ValueError as exc:
            problems.append(Problem(str(exc), index))
            continue
        if action.key == "open_path":
            reason = refusal(expand(params["path"]))
            if reason:
                missing = reason.startswith("Nothing exists")
                problems.append(
                    Problem(reason + (" (checked again when it runs)" if missing else ""), index, blocking=not missing)
                )
    return problems


def blocking(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.blocking]


def describe_step(step: Step) -> str:
    """A plain-language line for a step (editor, previews, the phone)."""
    action = ACTIONS.get(step.type)
    if action is None:
        return f"Unknown action '{step.type}'"
    if step.label.strip():
        return step.label.strip()
    p = step.params
    if step.type == "open_application":
        where = " in the background" if p.get("where") == "garvis_workspace" else ""
        return f"Open {p.get('application') or 'an application'}{where}"
    if step.type == "open_website":
        return f"Open {urlparse(str(p.get('url', ''))).netloc or p.get('url') or 'a web page'}"
    if step.type == "open_websites":
        urls = p.get("urls") or []
        urls = [u for u in (urls.splitlines() if isinstance(urls, str) else urls) if str(u).strip()]
        return f"Open {len(urls)} web page{'s' if len(urls) != 1 else ''}"
    if step.type == "open_path":
        return f"Open {Path(str(p.get('path') or '')).name or p.get('path') or 'a folder or file'}"
    if step.type == "focus_window":
        return f"Bring {p.get('window') or 'a window'} forward"
    if step.type == "wait_for_window":
        return f"Wait for {p.get('window') or 'a window'}"
    if step.type == "wait":
        return f"Pause {float(p.get('seconds') or 0):g} s"
    if step.type == "notify":
        return f"Notify: {p.get('title') or '…'}"
    if step.type == "confirm":
        return f"Ask first: {p.get('message') or '…'}"
    if step.type == "ask_garvis":
        text = str(p.get("instruction") or "…")
        return f"Ask Garvis: {text if len(text) <= 60 else text[:59] + '…'}"
    return action.label


def public_summary(protocol: Protocol, tools: set[str] | None = None) -> dict:
    """What the phone may see: names and plain step descriptions -- no raw
    parameters (paths, instructions) beyond what the description shows."""
    problems = validate(protocol, tools)
    stopping = blocking(problems)
    return {
        "id": protocol.id,
        "name": protocol.name,
        "description": protocol.description,
        "enabled": protocol.enabled,
        "steps": [describe_step(s) for s in protocol.steps if s.enabled],
        "runnable": protocol.enabled and not stopping,
        "problem": (stopping[0].text() if stopping else ("Disabled on the PC." if not protocol.enabled else "")),
        "updated_at": protocol.updated_at,
    }


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
@dataclass
class StoredProtocol:
    path: Path
    protocol: Protocol | None
    error: str = ""


@dataclass
class ProtocolStore:
    directory: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def _path(self, protocol_id: str) -> Path:
        if not _ID_RE.match(protocol_id or ""):
            raise ValueError("Invalid protocol id.")
        return self.directory / f"{protocol_id}.json"

    def load_all(self) -> list[StoredProtocol]:
        with self._lock:
            if not self.directory.exists():
                return []
            entries = []
            for path in sorted(self.directory.glob("*.json")):
                entries.append(self._read(path))
            entries.sort(key=lambda e: (e.protocol is None, (e.protocol.name.lower() if e.protocol else e.path.name)))
            return entries

    def protocols(self) -> list[Protocol]:
        return [e.protocol for e in self.load_all() if e.protocol is not None]

    def get(self, protocol_id: str) -> Protocol | None:
        try:
            path = self._path(protocol_id)
        except ValueError:
            return None
        with self._lock:
            if not path.exists():
                return None
            return self._read(path).protocol

    @staticmethod
    def _read(path: Path) -> StoredProtocol:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            protocol = Protocol.model_validate(data)
        except (OSError, ValueError, ValidationError) as exc:
            log_extra(logger, 30, "protocols.invalid_file", file=path.name)
            first = str(exc).splitlines()
            return StoredProtocol(path, None, f"Can't be loaded: {first[0] if first else exc}")
        if path.stem != protocol.id:
            return StoredProtocol(path, None, "Can't be loaded: the file name doesn't match its id.")
        return StoredProtocol(path, protocol)

    def save(self, protocol: Protocol) -> Protocol:
        protocol = protocol.model_copy(update={"updated_at": time.time(), "schema_version": SCHEMA_VERSION})
        path = self._path(protocol.id)
        text = json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, indent=2)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".protocol-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        log_extra(logger, 20, "protocols.saved", protocol=protocol.id, steps=len(protocol.steps))
        return protocol

    def delete(self, protocol_id: str) -> bool:
        path = self._path(protocol_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
        log_extra(logger, 20, "protocols.deleted", protocol=protocol_id)
        return True

    def duplicate(self, protocol_id: str) -> Protocol | None:
        original = self.get(protocol_id)
        if original is None:
            return None
        now = time.time()
        copy = original.model_copy(
            deep=True,
            update={
                "id": new_id(),
                "name": f"{original.name} (copy)"[:80],
                "created_at": now,
                "updated_at": now,
                "steps": [s.model_copy(update={"id": uuid.uuid4().hex[:12]}) for s in original.steps],
            },
        )
        return self.save(copy)

    @staticmethod
    def export(protocol: Protocol, target: Path) -> None:
        Path(target).write_text(
            json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
