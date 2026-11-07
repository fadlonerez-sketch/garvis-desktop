"""The core agent loop.

USER -> AGENT CORE -> LLM -> TOOL SELECTION -> PERMISSION CHECK ->
CONFIRMATION (if needed) -> TOOL EXECUTION -> RESULT -> LLM -> FINAL ANSWER

The agent never executes anything itself: it asks the LLM what it wants to
do, validates and permission-checks that request, and only then calls into
a registered tool. This module has no Qt/voice dependencies -- it
communicates status purely through the EventBus, so it can run headlessly
(tests, a future API layer) or drive the desktop UI unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from app.agent import presentation
from app.agent.context import ConversationContext
from app.agent.memory import MemoryStore
from app.agent.policies import MAX_TOOL_ITERATIONS, build_system_prompt_parts
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import (
    ConfirmationDeniedError,
    ConfirmationTimeoutError,
    LLMBudgetExceededError,
    LLMError,
    ToolNotFoundError,
)
from app.core.logging import get_logger, log_extra
from app.llm.base import LLMProvider, Message, Role, ToolCall
from app.llm.usage import task_meter
from app.security.confirmations import ConfirmationManager
from app.security.permissions import PermissionChecker, PermissionDecision
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from app.tools.sources import extract_sources

if TYPE_CHECKING:
    # Type-only: the agent core never imports Playwright at runtime. This
    # reference exists purely so the UI layer has one place (`agent.browser_manager`)
    # to find the browser resource and shut it down cleanly on app exit --
    # the agent itself never calls anything on it.
    from app.browser.manager import BrowserManager

logger = get_logger("agent")

_MAX_IDENTICAL_CALLS = 3
"""The same tool with the same arguments runs at most this often per turn."""
_MAX_BLOCKED_REPEATS = 3
"""After this many refused repeats in one turn the loop stops: the model is
stuck, and every further round-trip is paid for nothing."""


def _call_key(tool_call: ToolCall) -> str:
    return tool_call.name + "\x00" + json.dumps(tool_call.arguments, sort_keys=True, default=str)


def _tool_message_content(result: ToolResult) -> str:
    """Compact JSON for the model: no null fields, no padding, and non-ASCII
    text (e.g. Hebrew OCR) kept as-is -- escaped \\uXXXX sequences cost
    several tokens per character."""
    payload = {"success": result.success}
    if result.data is not None:
        payload["data"] = result.data
    if result.error is not None:
        payload["error"] = result.error
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


class CancelToken:
    """Cooperative cancellation for one Agent call (a turn or a protocol).

    `cancel()` only *asks*: the Agent stops at its next safe point -- before
    the next model request or tool step -- never in the middle of a tool
    (a half-finished desktop action can't be undone). `honored` becomes True
    only once the work has actually stopped, so a caller can tell a
    cancelled task from one that finished before the request landed."""

    def __init__(self) -> None:
        self.requested = False
        self.honored = False

    def cancel(self) -> None:
        self.requested = True


class _TurnCancelled(Exception):
    pass


CANCELLED_REPLY = "Stopped: the task was cancelled. Anything already done stays done."


class _TurnState:
    """Per-turn bookkeeping for the loop guard and partial-progress reports."""

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {}
        self.failures: dict[str, str] = {}
        self.blocked_repeats = 0
        self.completed: list[str] = []
        self.cancel: CancelToken | None = None
        self.source: str | None = None


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: MemoryStore,
        events: EventBus,
        permissions: PermissionChecker | None = None,
        confirmations: ConfirmationManager | None = None,
        max_context_messages: int = 30,
        browser_manager: "BrowserManager | None" = None,
        max_context_tokens: int | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        context_provider: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        """`context_provider`, when given, returns a short live description
        of the user's desktop (active window, open apps) that is appended to
        the system prompt at the start of every turn -- so the model knows
        what the user is looking at without spending a tool call on it."""
        self._llm = llm
        self._registry = registry
        self._memory = memory
        self._events = events
        self._permissions = permissions or PermissionChecker()
        self._confirmations = confirmations or ConfirmationManager()
        context_kwargs = {"max_tokens": max_context_tokens} if max_context_tokens is not None else {}
        self._context = ConversationContext(max_messages=max_context_messages, **context_kwargs)
        self._browser_manager = browser_manager
        self.desktop = None
        """The DesktopController the tools use (set by build_agent), so the UI
        can bring a window a tool opened back into view. Never used here."""
        self._max_tool_iterations = max(1, max_tool_iterations)
        self._context_provider = context_provider
        self._turn_lock = asyncio.Lock()
        """Serializes turns from every caller (typed text, push-to-talk, a
        voice session, the mobile interface): two turns interleaving on the
        one shared ConversationContext would corrupt it (e.g. a user message
        landing between an assistant tool call and its tool result)."""

    @property
    def llm(self) -> LLMProvider:
        """The one model router this agent uses (shared by helpers such as
        reminder phrasing, so every request is metered and budgeted alike)."""
        return self._llm

    def register_tools(self, tools) -> None:
        """Add tools after construction -- the phone companion's reminder,
        image and phone-action tools, built once the mobile interface is on.
        They go through the same registry, permission and confirmation path
        as every other tool."""
        for tool in tools:
            self._registry.register(tool)

    @property
    def is_busy(self) -> bool:
        """True while a turn is running (or another caller is waiting for one)."""
        return self._turn_lock.locked()

    @property
    def confirmations(self) -> ConfirmationManager:
        return self._confirmations

    @property
    def browser_manager(self) -> "BrowserManager | None":
        """Exposed only so callers (e.g. the UI) can shut the browser down
        cleanly on exit -- the agent loop itself never touches it."""
        return self._browser_manager

    async def _publish(self, event_type: EventType, **payload: object) -> None:
        await self._events.publish(AgentEvent(type=event_type, payload=payload))

    def reset_conversation(self) -> None:
        """Start a fresh conversation: drop the message history (user,
        assistant and tool messages) and keep the system prompt. Long-term
        memory is not touched -- it is re-read into the system prompt on
        every turn. Called at each new session boundary (see
        AgentBridge._start_session_async) so a new session never re-sends the
        previous session's turns to the LLM."""
        dropped = sum(1 for m in self._context.messages if m.role != Role.SYSTEM)
        self._context.clear()
        log_extra(logger, 20, "agent.conversation_reset", dropped_messages=dropped)

    async def reset_conversation_between_turns(self) -> None:
        """reset_conversation(), but only once no turn is in flight -- so a
        reset requested by one interface (a new voice session) can never land
        in the middle of a turn started by another (the mobile interface)."""
        async with self._turn_lock:
            self.reset_conversation()

    async def _refresh_system_prompt(self) -> None:
        memory_items = await self._memory.recall(limit=20)
        desktop_context = None
        if self._context_provider is not None:
            try:
                desktop_context = await self._context_provider()
            except Exception:  # context is a convenience; never fail a turn over it
                log_extra(logger, 30, "agent.desktop_context_failed")
        stable, per_turn = build_system_prompt_parts(
            memory_items, desktop_context, self._context.earlier_turns_digest()
        )
        content = "\n\n".join([stable, per_turn]) if per_turn else stable
        self._context.set_system_prompt(content, cacheable_prefix_chars=len(stable))

    async def handle_user_message(
        self, text: str, *, source: str | None = None, cancel: CancelToken | None = None
    ) -> str:
        """Run one full user turn through the agent loop, returning the final reply.

        Turns are serialized (see `_turn_lock`): a call made while another
        turn runs waits for it. `source` optionally names the interface the
        turn came from (e.g. "mobile"); it is only echoed in the USER_MESSAGE
        and ASSISTANT_MESSAGE payloads so listeners can tell turns apart --
        the agent itself treats every turn identically."""
        async with self._turn_lock:
            with task_meter() as meter:
                try:
                    return await self._handle_turn(text, source, cancel)
                finally:
                    if meter.calls:
                        log_extra(logger, 20, "llm.task_summary", **meter.summary())

    async def _handle_turn(self, text: str, source: str | None, cancel: CancelToken | None = None) -> str:
        origin = {"source": source} if source is not None else {}
        # The user's message goes in first, so this turn's trimming/digest is
        # settled before the system prompt (which carries the digest) is built.
        self._context.add(Message(role=Role.USER, content=text))
        await self._refresh_system_prompt()
        await self._publish(EventType.USER_MESSAGE, text=text, **origin)
        await self._publish(EventType.STATUS, status="Thinking...")
        turn = _TurnState()
        turn.cancel = cancel
        turn.source = source

        try:
            reply = await self._run_loop(turn)
        except _TurnCancelled:
            cancel.honored = True
            log_extra(logger, 20, "agent.turn_cancelled", completed_tools=turn.completed)
            reply = CANCELLED_REPLY
            if turn.completed:
                reply += " Completed before stopping: " + ", ".join(turn.completed) + "."
            await self._publish(EventType.ASSISTANT_MESSAGE, text=reply, **origin)
            await self._publish(EventType.STATUS, status="Idle.")
            return reply
        except LLMBudgetExceededError as exc:
            # A local spending limit stopped the turn before another request.
            # Say what already happened -- tools may have acted.
            message = str(exc)
            if turn.completed:
                message += " Completed before stopping: " + ", ".join(turn.completed) + "."
            log_extra(logger, 30, "agent.budget_stop", limit=exc.limit, completed_tools=turn.completed)
            await self._publish(EventType.ERROR, message=message)
            await self._publish(EventType.STATUS, status="Idle.")
            return message
        except LLMError as exc:
            # Any known LLM-backend failure (unreachable, rate-limited,
            # authentication rejected, ...) already carries a specific,
            # user-facing message -- surface it as-is rather than falling
            # through to the generic safety net below, even if a tool call
            # earlier in this turn already succeeded and took effect.
            await self._publish(EventType.ERROR, message=str(exc))
            await self._publish(EventType.STATUS, status="Idle.")
            return str(exc)
        except Exception:
            # Safety net: an unexpected bug anywhere in the loop (a tool,
            # context handling, etc.) must never crash the background
            # asyncio task silently -- callers (AgentBridge) fire this
            # coroutine without awaiting its result, so an uncaught
            # exception here would otherwise leave the UI stuck in
            # "Thinking..." forever with no way to send another message.
            # The real exception (with traceback) is logged server-side;
            # only a generic, non-sensitive message reaches the user/LLM.
            logger.exception("agent.unexpected_error")
            safe_message = "Something went wrong while processing that. Please try again."
            await self._publish(EventType.ERROR, message=safe_message)
            await self._publish(EventType.STATUS, status="Idle.")
            return safe_message

        await self._publish(EventType.ASSISTANT_MESSAGE, text=reply, **origin)
        await self._publish(EventType.STATUS, status="Done.")
        return reply

    async def _run_loop(self, turn: _TurnState | None = None) -> str:
        turn = turn or _TurnState()
        tool_specs = self._registry.tool_specs()

        for _ in range(self._max_tool_iterations):
            self._check_cancel(turn)
            response = await self._llm.chat(self._context.messages, tools=tool_specs)
            assistant_message = response.message
            self._context.add(assistant_message)

            if response.finish_reason != "tool_calls" or not assistant_message.tool_calls:
                if response.finish_reason == "length" and not assistant_message.content.strip():
                    log_extra(logger, 30, "agent.reply_truncated")
                    return "My reply hit the output limit before I could finish. Ask me to continue, or narrow it down."
                return assistant_message.content

            for index, tool_call in enumerate(assistant_message.tool_calls):
                if turn.cancel is not None and turn.cancel.requested:
                    # Every tool call needs a result for the history to stay valid.
                    for skipped in assistant_message.tool_calls[index:]:
                        self._context.add(
                            Message(
                                role=Role.TOOL,
                                content=_tool_message_content(ToolResult.fail("Not run: the user cancelled the task.")),
                                tool_call_id=skipped.id,
                                name=skipped.name,
                            )
                        )
                    raise _TurnCancelled
                await self._handle_tool_call(tool_call, turn)

            if turn.blocked_repeats >= _MAX_BLOCKED_REPEATS:
                log_extra(logger, 30, "agent.loop_stopped", blocked_repeats=turn.blocked_repeats)
                last_error = next(reversed(turn.failures.values()), "")
                return (
                    "I stopped: I kept repeating the same step without getting anywhere"
                    + (f" (last error: {last_error})" if last_error else "")
                    + ". Tell me how you'd like to proceed."
                )

        log_extra(logger, 30, "agent.max_iterations_reached")
        return "I wasn't able to finish that within my step limit. Could you narrow down the request?"

    @staticmethod
    def _check_cancel(turn: _TurnState) -> None:
        if turn.cancel is not None and turn.cancel.requested:
            raise _TurnCancelled

    async def _handle_tool_call(self, tool_call: ToolCall, turn: _TurnState | None = None) -> None:
        turn = turn or _TurnState()
        key = _call_key(tool_call)
        refusal = self._repeat_refusal(key, turn)
        if refusal is not None:
            # Not executed: it would fail the same way (or loop). Every answer
            # still gets a tool result, so the history stays valid.
            turn.blocked_repeats += 1
            log_extra(logger, 30, "agent.tool_repeat_blocked", tool=tool_call.name)
            result = ToolResult.fail(refusal)
        else:
            turn.call_counts[key] = turn.call_counts.get(key, 0) + 1
            result = await self._execute_and_publish(
                tool_call.name, tool_call.arguments, source=getattr(turn, "source", None), from_model=True
            )
            if result.success:
                turn.failures.pop(key, None)
                turn.completed.append(tool_call.name)
            else:
                turn.failures[key] = (result.error or "failed")[:200]
        self._context.add(
            Message(
                role=Role.TOOL,
                content=_tool_message_content(result),
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        )

    @staticmethod
    def _repeat_refusal(key: str, turn: _TurnState) -> str | None:
        if key in turn.failures:
            return (
                "Not run again: this exact call already failed in this task with: "
                f"{turn.failures[key]}. Change the approach or report the blocker to the user."
            )
        if turn.call_counts.get(key, 0) >= _MAX_IDENTICAL_CALLS:
            return (
                f"Not run again: this exact call already ran {_MAX_IDENTICAL_CALLS} times in this task. "
                "Use the earlier results, or change the approach."
            )
        return None

    async def _execute_and_publish(
        self,
        name: str,
        arguments: dict,
        *,
        source: str | None = None,
        presentation_override: str | None = None,
        task_id: str | None = None,
        from_model: bool = False,
    ) -> ToolResult:
        """TOOL_STARTED, the permission-checked call, TOOL_FINISHED (with any
        sources) and, for a result the user should see, one PRESENTATION.
        Shared by chat turns and protocol steps, so both go through exactly
        the same validation, permission and confirmation path. `from_model`
        marks a call the model asked for (vs. a user-authored protocol step),
        which also gets the egress taint check (app/security/egress.py)."""
        await self._publish(EventType.TOOL_STARTED, name=name, arguments=arguments)
        result = await self._execute_tool_call(
            ToolCall(id=f"call_{name}", name=name, arguments=arguments), from_model=from_model
        )
        # For the desktop Sources view only; the model's tool message is unchanged.
        sources = extract_sources(name, result.data) if result.success else []
        await self._publish(
            EventType.TOOL_FINISHED,
            name=name,
            success=result.success,
            error=result.error,
            **({"sources": sources} if sources else {}),
        )
        if result.success:
            try:
                tool_level = getattr(self._registry.get(name), "presentation", "none")
            except ToolNotFoundError:
                tool_level = "none"
            level = presentation.level_for(tool_level, result.data, presentation_override)
            if level != presentation.Presentation.NONE:
                await self._events.publish(
                    AgentEvent(
                        type=EventType.PRESENTATION,
                        payload=presentation.describe(name, result.data, level, source=source, task_id=task_id),
                    )
                )
        return result

    async def run_tool(
        self,
        name: str,
        arguments: dict,
        *,
        source: str | None = None,
        presentation_override: str | None = None,
        task_id: str | None = None,
    ) -> ToolResult:
        """Run one registered tool outside a chat turn (a protocol step):
        the same validation, PermissionChecker decision and confirmation as
        when the model asks for it, serialized with turns by the turn lock.
        Nothing is added to the conversation and no model is called."""
        async with self._turn_lock:
            return await self._execute_and_publish(
                name, arguments, source=source, presentation_override=presentation_override, task_id=task_id
            )

    def has_tool(self, name: str) -> bool:
        try:
            self._registry.get(name)
        except ToolNotFoundError:
            return False
        return True

    async def ask_confirmation(self, title: str, reason: str, arguments: dict | None = None) -> str:
        """Ask the user a yes/no question through the same ConfirmationManager
        (desktop dialog or phone card, first answer wins). Returns
        "approved", "denied", "timeout" or "cancelled"."""
        request = self._confirmations.create_request(tool_name=title, arguments=arguments or {}, reason=reason)
        await self._publish(
            EventType.CONFIRMATION_REQUESTED,
            id=request.id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            reason=request.reason,
        )
        outcome = "cancelled"
        try:
            await self._confirmations.wait_for_decision(request.id)
            outcome = "approved"
        except ConfirmationDeniedError:
            outcome = "denied"
        except ConfirmationTimeoutError:
            outcome = "timeout"
        finally:
            await self._publish(EventType.CONFIRMATION_RESOLVED, id=request.id, outcome=outcome)
        return outcome

    async def _execute_tool_call(self, tool_call: ToolCall, from_model: bool = False) -> ToolResult:
        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotFoundError as exc:
            log_extra(logger, 30, "agent.tool_not_found", tool=tool_call.name)
            return ToolResult.fail(str(exc))

        try:
            args = tool.validate_arguments(tool_call.arguments)
        except Exception as exc:  # pydantic ValidationError, etc.
            log_extra(logger, 30, "agent.tool_validation_failed", tool=tool_call.name, error=str(exc))
            return ToolResult.fail(f"Invalid arguments for '{tool.name}': {exc}")

        if from_model:
            permission = await self._permissions.evaluate_call(tool, args, history=self._context.messages)
        else:
            permission = await self._permissions.evaluate_call(tool, args)
        if permission.decision == PermissionDecision.DENY:
            log_extra(logger, 30, "agent.tool_denied", tool=tool_call.name, reason=permission.reason)
            return ToolResult.fail(f"Denied: {permission.reason}")

        if permission.decision == PermissionDecision.REQUIRE_CONFIRMATION:
            request = self._confirmations.create_request(
                tool_name=tool.name, arguments=tool_call.arguments, reason=permission.reason
            )
            await self._publish(
                EventType.CONFIRMATION_REQUESTED,
                id=request.id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                reason=request.reason,
            )
            outcome = "cancelled"
            try:
                await self._confirmations.wait_for_decision(request.id)
                outcome = "approved"
            except ConfirmationDeniedError as exc:
                outcome = "denied"
                log_extra(logger, 20, "agent.tool_confirmation_declined", tool=tool_call.name)
                return ToolResult.fail(str(exc))
            except ConfirmationTimeoutError as exc:
                outcome = "timeout"
                log_extra(logger, 20, "agent.tool_confirmation_declined", tool=tool_call.name)
                return ToolResult.fail(str(exc))
            finally:
                # Lets every interface showing this request (desktop dialog,
                # mobile card) close it, whichever one answered it.
                await self._publish(EventType.CONFIRMATION_RESOLVED, id=request.id, outcome=outcome)

        try:
            return await tool.execute(args)
        except Exception as exc:  # tools should not raise, but never fail silently if they do
            log_extra(logger, 40, "agent.tool_execution_error", tool=tool_call.name, error=str(exc))
            return ToolResult.fail(f"Tool '{tool.name}' raised an unexpected error: {exc}")
