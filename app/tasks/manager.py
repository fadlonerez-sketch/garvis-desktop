"""The one owner of submitted tasks: queued, running and finished.

A task is anything Garvis was asked to do: an instruction sent from the
phone, a protocol run (from the phone or the PC), or -- recorded, not
executed here -- a chat turn typed or spoken on the PC. Everything runs
on AgentBridge's asyncio loop, one task at a time, through the existing
Agent (turns) and Agent.run_tool (protocol steps); this module only queues,
tracks and remembers.

Guarantees:
- A task's final status reflects what actually happened: "completed" only
  after the work finished, "cancelled" only once it really stopped.
- The same submission (client `request_id`) is accepted once; a retry gets
  the original task back. A protocol that is already queued or running
  can't be queued again.
- Tasks survive a phone disconnect and a Garvis restart (a small JSON file,
  no conversation history). A task that was queued or running when Garvis
  closed is marked "interrupted" and is never re-run automatically --
  repeating desktop actions unasked could do them twice.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.agent.agent import CancelToken
from app.core.events import AgentEvent, EventBus, EventType
from app.core.logging import get_logger, log_extra
from app.tasks.protocols import ProtocolStore, blocking, describe_step, validate
from app.tasks.runner import ProtocolRun

logger = get_logger("tasks.manager")

QUEUED, RUNNING, WAITING, CANCELLING = "queued", "running", "waiting_confirmation", "cancelling"
COMPLETED, FAILED, CANCELLED, INTERRUPTED = "completed", "failed", "cancelled", "interrupted"
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED, INTERRUPTED})
ACTIVE = frozenset({QUEUED, RUNNING, WAITING, CANCELLING})

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MOBILE = "mobile"
_MAX_TEXT = 4000
_MAX_SOURCES = 20
_MAX_OPENED = 20


class TaskError(Exception):
    """A task can't be accepted or changed; the message is user-facing."""


def _clip(text: Any, limit: int = _MAX_TEXT) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Task:
    id: str
    kind: str  # "instruction" | "protocol" | "chat"
    origin: str  # "desktop" | "mobile"
    title: str
    device: str = ""
    """For phone tasks: the paired device's type ("iPhone", "Android", ...)."""
    text: str = ""
    protocol_id: str = ""
    request_id: str = ""
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict = field(default_factory=dict)  # {"step": n, "total": m, "label": str}
    result: str = ""
    error: str = ""
    note: str = ""
    steps: list[dict] = field(default_factory=list)
    """Protocol steps, or the tool calls of an instruction/chat turn."""
    sources: list[dict] = field(default_factory=list)
    opened: list[dict] = field(default_factory=list)
    """PRESENTATION payloads: what this task put on the screen."""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "device": self.device,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": dict(self.progress),
            "note": self.note,
            "error": _clip(self.error, 300),
            "result_preview": _clip(self.result, 160),
            "opened_count": len(self.opened),
            "source_count": len(self.sources),
            "protocol_id": self.protocol_id,
            "cancellable": self.status in (QUEUED, RUNNING, WAITING) and self.kind != "chat",
        }

    def detail(self) -> dict:
        data = self.summary()
        data.update(
            text=self.text,
            result=self.result,
            error=self.error,
            steps=list(self.steps),
            sources=list(self.sources),
            opened=[_public_presentation(o) for o in self.opened],
        )
        return data


def _public_presentation(item: dict) -> dict:
    """What a task record keeps about something that was shown (no window handle)."""
    return {key: item.get(key) for key in ("kind", "title", "detail", "url", "verified", "level")}


class TaskManager:
    def __init__(
        self,
        agent,
        events: EventBus,
        store_path: Path,
        protocols: ProtocolStore,
        *,
        available_tools: Callable[[], set[str] | None] = lambda: None,
        is_desktop_session_active: Callable[[], bool] = lambda: False,
        max_queued: int = 20,
        keep: int = 200,
        max_text_chars: int = 4000,
    ) -> None:
        self._agent = agent
        self._events = events
        self._path = Path(store_path)
        self._protocols = protocols
        self._available_tools = available_tools
        self._is_session_active = is_desktop_session_active
        self._max_queued = max_queued
        self._keep = keep
        self._max_text = max_text_chars
        self._tasks: dict[str, Task] = {}
        self._order: deque[str] = deque()
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task | None = None
        self._current: Task | None = None
        self._passive: Task | None = None
        self._cancel: CancelToken | None = None
        self._confirmations: set[str] = set()
        self._listeners: list[Callable[[Task], None]] = []
        self._unsubscribe: Callable[[], None] | None = None
        self.version = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """On the loop that hosts the Agent."""
        if self._worker is not None:
            return
        self._load()
        self._queue = asyncio.Queue()
        self._unsubscribe = self._events.subscribe(self._on_event)
        self._worker = asyncio.get_running_loop().create_task(self._work())

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._cancel is not None:
            self._cancel.cancel()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError, Exception:
                pass
        for task in self._tasks.values():
            if task.status in ACTIVE:
                self._finish(task, INTERRUPTED, error="Garvis was closed before this finished.", notify=False)
        self._save()

    def add_listener(self, listener: Callable[[Task], None]) -> None:
        self._listeners.append(listener)

    # ---------------------------------------------------------------- reads
    def list(self, limit: int = 50) -> list[Task]:
        ids = list(self._order)[-limit:]
        return [self._tasks[i] for i in reversed(ids)]

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    @property
    def current(self) -> Task | None:
        return self._current

    def queued_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == QUEUED)

    # --------------------------------------------------------------- submit
    def _deduplicate(self, request_id: str | None) -> Task | None:
        if not request_id:
            return None
        if not REQUEST_ID_RE.match(request_id):
            raise TaskError("Invalid request id.")
        return next((t for t in self._tasks.values() if t.request_id == request_id), None)

    def _accept(self, task: Task) -> Task:
        if self._queue is None:
            raise TaskError("Garvis isn't ready to take tasks yet.")
        if self.queued_count() >= self._max_queued:
            raise TaskError(f"{self._max_queued} tasks are already waiting. Try again when some have run.")
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._trim()
        self._queue.put_nowait(task.id)
        log_extra(logger, 20, "task.accepted", task=task.id, kind=task.kind, origin=task.origin)
        self._changed(task)
        return task

    def submit_instruction(
        self, text: str, *, origin: str = MOBILE, request_id: str | None = None, device: str = ""
    ) -> Task:
        existing = self._deduplicate(request_id)
        if existing is not None:
            return existing
        text = (text or "").strip()
        if not text:
            raise TaskError("The instruction is empty.")
        if len(text) > self._max_text:
            raise TaskError(f"The instruction is too long (limit {self._max_text} characters).")
        title = text if len(text) <= 80 else text[:79] + "…"
        return self._accept(
            Task(
                id=secrets.token_urlsafe(9),
                kind="instruction",
                origin=origin,
                title=title,
                text=text,
                request_id=request_id or "",
                device=device[:20],
            )
        )

    def submit_protocol(
        self, protocol_id: str, *, origin: str = "desktop", request_id: str | None = None, device: str = ""
    ) -> Task:
        """The desktop decides: the protocol must exist, be enabled and pass
        validation now -- a phone holding an old id gets no shortcut."""
        existing = self._deduplicate(request_id)
        if existing is not None:
            return existing
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            raise TaskError("That protocol doesn't exist (any more).")
        if not protocol.enabled:
            raise TaskError(f"“{protocol.name}” is disabled on the PC.")
        stopping = blocking(validate(protocol, self._available_tools()))
        if stopping:
            raise TaskError(f"“{protocol.name}” can't run: {stopping[0].text()}")
        busy = next((t for t in self._tasks.values() if t.protocol_id == protocol.id and t.status in ACTIVE), None)
        if busy is not None:
            raise TaskError(f"“{protocol.name}” is already {'running' if busy.status != QUEUED else 'queued'}.")
        enabled = [s for s in protocol.steps if s.enabled]
        task = Task(
            id=secrets.token_urlsafe(9),
            kind="protocol",
            origin=origin,
            title=protocol.name,
            device=device[:20],
            text=protocol.description,
            protocol_id=protocol.id,
            request_id=request_id or "",
            progress={"step": 0, "total": len(enabled), "label": ""},
            steps=[
                {"index": i, "type": s.type, "label": describe_step(s), "status": "pending", "message": ""}
                for i, s in enumerate(enabled)
            ],
        )
        task._snapshot = protocol.model_copy(deep=True)  # what runs, even if the protocol is edited meanwhile
        return self._accept(task)

    # --------------------------------------------------------------- cancel
    def cancel(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskError("Unknown task.")
        if task.kind == "chat":
            raise TaskError("Chat turns from the PC are stopped on the PC.")
        if task.status == QUEUED:
            self._finish(task, CANCELLED, error="Cancelled before it started.")
            return task
        if task.status in (RUNNING, WAITING) and task is self._current and self._cancel is not None:
            self._cancel.cancel()
            task.status = CANCELLING
            task.note = "Stopping after the current step…"
            for request_id in list(self._confirmations):  # an open question would hold the task forever
                self._agent.confirmations.resolve(request_id, False)
            self._changed(task)
            return task
        if task.status == CANCELLING:
            return task
        raise TaskError("That task has already finished.")

    # --------------------------------------------------------------- worker
    async def _work(self) -> None:
        while True:
            task_id = await self._queue.get()
            task = self._tasks.get(task_id)
            if task is None or task.status != QUEUED:
                continue
            while self._is_session_active() and task.status == QUEUED:
                if task.note != "Waiting for the voice session on the PC to end.":
                    task.note = "Waiting for the voice session on the PC to end."
                    self._changed(task)
                await asyncio.sleep(1.0)
            if task.status != QUEUED:
                continue
            await self._run(task)

    async def _run(self, task: Task) -> None:
        self._current = task
        self._cancel = CancelToken()
        self._confirmations.clear()
        task.status, task.started_at, task.note = RUNNING, time.time(), ""
        self._changed(task)
        source = MOBILE if task.origin == MOBILE else None
        try:
            if task.kind == "instruction":
                reply = await self._agent.handle_user_message(task.text, source=source, cancel=self._cancel)
                if self._cancel.honored:
                    self._finish(task, CANCELLED, result=reply)
                elif task.error and not task.result:
                    self._finish(task, FAILED)
                else:
                    self._finish(task, COMPLETED, result=task.result or reply)
            else:
                await self._run_protocol(task, source)
        except asyncio.CancelledError:
            self._finish(task, INTERRUPTED, error="Garvis was closed before this finished.", notify=False)
            raise
        except Exception as exc:  # noqa: BLE001 -- never leave a task "running" forever
            logger.exception("task.crashed")
            self._finish(task, FAILED, error=f"Something went wrong: {exc}")
        finally:
            self._current = None
            self._cancel = None
            self._confirmations.clear()

    async def _run_protocol(self, task: Task, source: str | None) -> None:
        protocol = getattr(task, "_snapshot", None) or self._protocols.get(task.protocol_id)
        if protocol is None:
            self._finish(task, FAILED, error="The protocol was deleted before it could run.")
            return
        await self._events.publish(
            AgentEvent(
                type=EventType.REQUEST_STARTED,
                payload={
                    "text": f"Protocol: {protocol.name}",
                    "source": source,
                    "task_id": task.id,
                    "kind": "protocol",
                },
            )
        )

        def on_step(steps: list[dict]) -> None:
            task.steps = [dict(s) for s in steps]
            running = next((s for s in steps if s["status"] == "running"), None)
            done = sum(1 for s in steps if s["status"] not in ("pending", "running"))
            task.progress = {
                "step": (running["index"] + 1) if running else done,
                "total": len(steps),
                "label": running["label"] if running else "",
            }
            self._changed(task)

        run = ProtocolRun(
            self._agent, self._events, protocol, task_id=task.id, source=source, cancel=self._cancel, on_step=on_step
        )
        try:
            outcome, summary = await run.run()
        finally:
            await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Done."}))
        status = {"completed": COMPLETED, "cancelled": CANCELLED}.get(outcome, FAILED)
        self._finish(task, status, result=summary, error=summary if status == FAILED else "")

    # --------------------------------------------------------------- events
    def _on_event(self, event: AgentEvent) -> None:
        payload = event.payload
        kind = event.type
        if kind == EventType.USER_MESSAGE and self._current is None and payload.get("source") is None:
            # A chat turn typed or spoken on the PC: recorded so every interface sees it.
            if self._passive is not None:
                self._finish(self._passive, COMPLETED if self._passive.result else FAILED)
            text = _clip(payload.get("text"), self._max_text)
            self._passive = Task(
                id=secrets.token_urlsafe(9),
                kind="chat",
                origin="desktop",
                title=text if len(text) <= 80 else text[:79] + "…",
                text=text,
                status=RUNNING,
                started_at=time.time(),
            )
            self._tasks[self._passive.id] = self._passive
            self._order.append(self._passive.id)
            self._trim()
            self._changed(self._passive)
            return
        task = self._current or self._passive
        if task is None:
            return
        if kind == EventType.TOOL_STARTED:
            if task.kind != "protocol":
                task.steps.append(
                    {
                        "index": len(task.steps),
                        "type": "tool",
                        "label": str(payload.get("name", "")),
                        "status": "running",
                        "message": "",
                    }
                )
                self._changed(task)
        elif kind == EventType.TOOL_FINISHED:
            if task.kind != "protocol" and task.steps:
                step = task.steps[-1]
                step["status"] = "ok" if payload.get("success") else "failed"
                step["message"] = _clip(payload.get("error") or "", 300)
            for source in payload.get("sources") or []:
                if isinstance(source, dict) and source.get("url") and len(task.sources) < _MAX_SOURCES:
                    if all(s.get("url") != source["url"] for s in task.sources):
                        task.sources.append({k: source.get(k, "") for k in ("title", "url", "publisher", "kind")})
            self._changed(task)
        elif kind == EventType.PRESENTATION:
            if payload.get("task_id") in (None, task.id) and len(task.opened) < _MAX_OPENED:
                task.opened.append(dict(payload))
                self._changed(task)
        elif kind == EventType.ASSISTANT_MESSAGE:
            if task.kind != "protocol":
                task.result = _clip(payload.get("text"))
            if task is self._passive:
                self._finish(task, COMPLETED)
                self._passive = None
        elif kind == EventType.ERROR:
            task.error = _clip(payload.get("message"), 1000)
            self._changed(task)
        elif kind == EventType.CONFIRMATION_REQUESTED and task is self._current:
            self._confirmations.add(str(payload.get("id", "")))
            if task.status == RUNNING:
                task.status = WAITING
                task.note = "Waiting for your approval (on the PC or the phone)."
                self._changed(task)
        elif kind == EventType.CONFIRMATION_RESOLVED and task is self._current:
            self._confirmations.discard(str(payload.get("id", "")))
            if task.status == WAITING and not self._confirmations:
                task.status, task.note = RUNNING, ""
                self._changed(task)
        elif kind == EventType.STATUS and task is self._passive:
            if payload.get("status") in ("Done.", "Idle."):
                self._finish(task, COMPLETED if task.result and not task.error else FAILED)
                self._passive = None

    # ------------------------------------------------------------- helpers
    def _finish(self, task: Task, status: str, *, result: str = "", error: str = "", notify: bool = True) -> None:
        if task.terminal:
            return
        task.status = status
        task.finished_at = time.time()
        task.note = ""
        if result:
            task.result = _clip(result)
        if error:
            task.error = _clip(error, 1000)
        log_extra(logger, 20, "task.finished", task=task.id, status=status)
        if notify:
            self._changed(task)
        self._save()

    def _changed(self, task: Task) -> None:
        self.version += 1
        if task.status in (QUEUED, CANCELLING) or task.terminal:
            self._save()
        for listener in list(self._listeners):
            try:
                listener(task)
            except Exception:  # noqa: BLE001
                logger.exception("task.listener_failed")
        loop_publish = self._events.publish(AgentEvent(type=EventType.TASK_UPDATED, payload={"task": task.summary()}))
        try:
            asyncio.get_running_loop().create_task(loop_publish)
        except RuntimeError:
            loop_publish.close()

    def _trim(self) -> None:
        while len(self._order) > self._keep:
            oldest = self._order[0]
            if self._tasks[oldest].status in ACTIVE:
                break
            self._order.popleft()
            self._tasks.pop(oldest, None)

    # ---------------------------------------------------------- persistence
    def _save(self) -> None:
        records = []
        for task_id in self._order:
            data = asdict(self._tasks[task_id])
            data["opened"] = [_public_presentation(o) for o in data["opened"]]
            records.append(data)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".tasks-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "tasks": records}, handle, ensure_ascii=False)
            os.replace(tmp, self._path)
        except OSError:
            log_extra(logger, 30, "task.save_failed")

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            records = data.get("tasks", []) if isinstance(data, dict) else []
        except FileNotFoundError:
            return
        except OSError, ValueError:
            log_extra(logger, 30, "task.load_failed")
            return
        fields = set(Task.__dataclass_fields__)
        for record in records[-self._keep :]:
            if not isinstance(record, dict) or not record.get("id"):
                continue
            try:
                task = Task(**{k: v for k, v in record.items() if k in fields})
            except TypeError:
                continue
            if task.status in ACTIVE:
                task.status = INTERRUPTED
                task.finished_at = task.finished_at or time.time()
                task.note = ""
                task.error = "Garvis was closed before this finished. It was not re-run."
            self._tasks[task.id] = task
            self._order.append(task.id)
