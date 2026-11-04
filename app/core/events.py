"""A tiny in-process event bus decoupling the agent core from the UI.

The agent core must never import Qt. Instead it publishes ``AgentEvent``
objects here; the UI subscribes and renders them. This keeps the agent
usable headlessly (tests, future API layer) while still driving a rich UI.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class EventType(StrEnum):
    STATUS = "status"  # e.g. "Thinking...", "Listening...", "Done."
    ASSISTANT_MESSAGE = "assistant_message"
    USER_MESSAGE = "user_message"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_RESOLVED = "confirmation_resolved"
    """Published once a CONFIRMATION_REQUESTED request is settled, with its
    `id` and `outcome` ("approved", "denied", "timeout" or "cancelled"). Only
    informational: the decision itself goes through ConfirmationManager."""
    ERROR = "error"
    REQUEST_STARTED = "request_started"
    """A request that is not a chat turn began -- a protocol run. Payload:
    `text` (a description, e.g. the protocol's name), `source`, `task_id`.
    Starts a new request for per-request state (desktop routing, the
    dashboard's activity) the way USER_MESSAGE does for chat turns, without
    adding anything to the conversation."""
    PRESENTATION = "presentation"
    """Something the user should see: a window/page/file a tool opened, or
    an explicit notice. Payload from app/agent/presentation.py (level,
    kind, title, detail, url, handle, source, task_id). Drives the desktop
    attention overlay and the task record; one event per action."""
    TASK_UPDATED = "task_updated"
    """A task (app/tasks/manager.py) changed. Payload: `task` (summary dict)."""
    SESSION_STATE = "session_state"
    """Hands-free conversation session state transitions -- see
    app/voice/session.py:ConversationSession/SessionState. Independent of
    STATUS: STATUS reflects one Agent turn's fine-grained progress,
    SESSION_STATE reflects the session's own high-level state (listening/
    thinking/speaking/interrupted/stopping), which spans multiple turns."""


class AgentEvent(BaseModel):
    type: EventType
    payload: dict[str, Any] = {}


Listener = Callable[[AgentEvent], Awaitable[None] | None]


class EventBus:
    """Simple synchronous-dispatch pub/sub bus, safe to call from asyncio code."""

    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def publish(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            result = listener(event)
            if isinstance(result, Awaitable):
                await result

    def publish_threadsafe(self, event: AgentEvent, loop: asyncio.AbstractEventLoop) -> None:
        """Publish from a non-asyncio thread onto ``loop``."""
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)
