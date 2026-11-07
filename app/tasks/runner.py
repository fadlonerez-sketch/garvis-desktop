"""Executes one accepted protocol run, step by step, on AgentBridge's loop.

Tool steps go through Agent.run_tool -- the same validation, permission
decision and confirmation as a chat request -- so a protocol can never do
more than Garvis would do when asked in words. The runner works on a
snapshot taken when the run was accepted: editing the protocol meanwhile
doesn't change a run in progress.

Cancellation is honest: it takes effect between steps (and immediately
inside a pause or while waiting for your answer); a step that is already
acting on the desktop finishes first. A step that times out is reported as
such -- the action it started may still complete in the background.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from app.agent import presentation as pres
from app.agent.agent import CANCELLED_REPLY, CancelToken
from app.core.events import AgentEvent, EventType
from app.core.logging import get_logger
from app.tasks.protocols import ACTIONS, Protocol, Step, clean_params, describe_step

logger = get_logger("tasks.runner")

_MAX_OUTPUT = 600


def _clip(text: Any, limit: int = _MAX_OUTPUT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StepFailed(Exception):
    pass


class StepCancelled(Exception):
    pass


def _tool_message(step_type: str, data: Any) -> str:
    data = data if isinstance(data, dict) else {}
    if step_type in ("open_website", "open_websites"):
        window = data.get("window")
        return f"Opened {data.get('opened', '')}" + (f" — {window}" if window else "")
    if step_type == "open_application":
        window = data.get("window")
        return f"Opened {data.get('opened', '')}" + (f" — {window}" if window else " (window not confirmed)")
    if step_type == "open_path":
        return f"Opened {data.get('path', '')}"
    if step_type in ("focus_window", "wait_for_window"):
        return f"Window: {data.get('window') or data.get('title') or 'found'}"
    return "Done."


class ProtocolRun:
    def __init__(
        self,
        agent,
        events,
        protocol: Protocol,
        *,
        task_id: str,
        source: str | None,
        cancel: CancelToken,
        on_step: Callable[[list[dict]], None],
    ) -> None:
        self._agent = agent
        self._events = events
        self._protocol = protocol
        self._task_id = task_id
        self._source = source
        self._cancel = cancel
        self._on_step = on_step
        self.steps: list[dict] = [
            {
                "index": i,
                "id": s.id,
                "type": s.type,
                "label": describe_step(s),
                "status": "pending",
                "message": "",
                "started_at": None,
                "finished_at": None,
            }
            for i, s in enumerate(s for s in protocol.steps if s.enabled)
        ]

    async def run(self) -> tuple[str, str]:
        """("completed" | "failed" | "cancelled", summary)."""
        enabled = [s for s in self._protocol.steps if s.enabled]
        failures = 0
        for index, step in enumerate(enabled):
            record = self.steps[index]
            if self._cancel.requested:
                return self._stop_cancelled(index)
            record.update(status="running", started_at=time.time())
            self._on_step(self.steps)
            action = ACTIONS[step.type]
            try:
                message = await asyncio.wait_for(self._run_step(step), timeout=action.timeout)
                record.update(status="ok", message=_clip(message))
            except StepCancelled as exc:
                record.update(status="cancelled", message=_clip(exc) or "Stopped.", finished_at=time.time())
                return self._stop_cancelled(index + 1, already_marked=True)
            except TimeoutError:
                record.update(
                    status="timeout",
                    message=f"Timed out after {action.timeout:g} s — anything it started may still finish.",
                )
            except StepFailed as exc:
                record.update(status="failed", message=_clip(exc))
            except Exception as exc:  # noqa: BLE001 -- a bug in one step must not kill the run silently
                logger.exception("protocol.step_crashed")
                record.update(status="failed", message=f"Unexpected error: {_clip(exc, 200)}")
            record["finished_at"] = time.time()
            self._on_step(self.steps)
            if record["status"] in ("failed", "timeout"):
                failures += 1
                if step.on_failure == "stop":
                    for later in self.steps[index + 1 :]:
                        later.update(status="skipped", message="Not run: an earlier step failed.")
                    self._on_step(self.steps)
                    return "failed", f"Stopped at step {index + 1} ({record['label']}): {record['message']}"
        done = len(enabled)
        if failures:
            return "failed", f"Finished with {failures} failed step{'s' if failures != 1 else ''} of {done}."
        return "completed", f"All {done} step{'s' if done != 1 else ''} done."

    def _stop_cancelled(self, from_index: int, already_marked: bool = False) -> tuple[str, str]:
        for later in self.steps[from_index:]:
            if later["status"] == "pending":
                later.update(status="skipped", message="Not run: the protocol was cancelled.")
        self._cancel.honored = True
        self._on_step(self.steps)
        done = sum(1 for s in self.steps if s["status"] == "ok")
        return "cancelled", f"Cancelled after {done} completed step{'s' if done != 1 else ''}."

    async def _tool(self, step: Step, name: str, arguments: dict) -> str:
        result = await self._agent.run_tool(
            name, arguments, source=self._source, presentation_override=step.presentation(), task_id=self._task_id
        )
        if not result.success:
            raise StepFailed(result.error or "The action failed.")
        return _tool_message(step.type, result.data)

    async def _run_step(self, step: Step) -> str:
        params = clean_params(step)  # validated again at run time: the files may have changed
        kind = step.type
        if kind == "open_application":
            return await self._tool(
                step, "open_application", {"application": params["application"], "target_desktop": params["where"]}
            )
        if kind == "open_website":
            return await self._tool(step, "open_website", {"url": params["url"], "target_desktop": "user_desktop"})
        if kind == "open_websites":
            opened = []
            for url in params["urls"]:
                if self._cancel.requested:
                    raise StepCancelled(f"Stopped after opening {len(opened)} of {len(params['urls'])} pages.")
                await self._tool(step, "open_website", {"url": url, "target_desktop": "user_desktop"})
                opened.append(url)
            return f"Opened {len(opened)} page{'s' if len(opened) != 1 else ''}."
        if kind == "open_path":
            return await self._tool(step, "open_path", {"path": params["path"]})
        if kind == "focus_window":
            return await self._tool(step, "focus_window", {"window": params["window"]})
        if kind == "wait_for_window":
            return await self._tool(
                step,
                "wait_for_window",
                {"window": params["window"], "timeout_seconds": float(params["timeout_seconds"])},
            )
        if kind == "wait":
            deadline = time.monotonic() + float(params["seconds"])
            while time.monotonic() < deadline:
                if self._cancel.requested:
                    raise StepCancelled("Stopped during the pause.")
                await asyncio.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            return f"Paused {float(params['seconds']):g} s."
        if kind == "notify":
            level = (
                pres.Presentation(step.presentation()) if step.presentation() != "none" else pres.Presentation.SUBTLE
            )
            await self._events.publish(
                AgentEvent(
                    type=EventType.PRESENTATION,
                    payload=pres.notice(
                        params["title"], params.get("message", ""), level, source=self._source, task_id=self._task_id
                    ),
                )
            )
            return f"Showed “{params['title']}”."
        if kind == "confirm":
            outcome = await self._agent.ask_confirmation(
                f"Protocol: {self._protocol.name}", params["message"], {"question": params["message"]}
            )
            if self._cancel.requested:
                raise StepCancelled("Stopped while waiting for your answer.")
            if outcome != "approved":
                raise StepCancelled(
                    "You declined." if outcome == "denied" else "No answer in time, so the protocol stopped."
                )
            return "You approved."
        if kind == "ask_garvis":
            reply = await self._agent.handle_user_message(
                params["instruction"], source=self._source, cancel=self._cancel
            )
            if self._cancel.honored or reply.startswith(CANCELLED_REPLY):
                raise StepCancelled("Stopped while Garvis was working on it.")
            return reply
        raise StepFailed(f"Unknown action '{kind}'.")
