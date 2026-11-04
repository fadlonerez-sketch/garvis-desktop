"""Pending-confirmation bookkeeping.

When the permission layer decides a tool call needs confirmation, the
agent registers a `ConfirmationRequest` here and *awaits* the user's
decision before executing anything. The UI is the only thing that can
resolve a pending confirmation (approve/deny); the LLM cannot.
"""
from __future__ import annotations

import asyncio
import uuid

from pydantic import BaseModel

from app.core.exceptions import ConfirmationDeniedError, ConfirmationTimeoutError


class ConfirmationRequest(BaseModel):
    id: str
    tool_name: str
    arguments: dict
    reason: str


class ConfirmationManager:
    def __init__(self, default_timeout_seconds: float = 120.0) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._default_timeout = default_timeout_seconds

    def create_request(self, tool_name: str, arguments: dict, reason: str) -> ConfirmationRequest:
        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        self._pending[request_id] = loop.create_future()
        return ConfirmationRequest(id=request_id, tool_name=tool_name, arguments=arguments, reason=reason)

    async def wait_for_decision(self, request_id: str, timeout_seconds: float | None = None) -> bool:
        future = self._pending.get(request_id)
        if future is None:
            raise KeyError(f"No pending confirmation with id {request_id}")
        try:
            approved = await asyncio.wait_for(future, timeout=timeout_seconds or self._default_timeout)
        except TimeoutError as exc:
            raise ConfirmationTimeoutError(
                f"Confirmation {request_id} timed out waiting for the user."
            ) from exc
        finally:
            self._pending.pop(request_id, None)
        if not approved:
            raise ConfirmationDeniedError(f"User denied confirmation {request_id}.")
        return approved

    def resolve(self, request_id: str, approved: bool) -> None:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        future.set_result(approved)

    def cancel_all(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
