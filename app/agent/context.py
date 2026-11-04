"""Short-term conversation context for the current session.

History is kept in whole *turns*: a turn is one USER message plus every
assistant/tool message that follows it, up to the next USER message.
Trimming only ever removes whole turns, oldest first, so a tool result is
never separated from the assistant message that requested it (an orphaned
tool result is rejected outright by Anthropic's API) and the current turn --
the user's latest request and everything done for it so far -- is never
dropped.

Cost controls, all deterministic and local (no summarization calls):
  - Large tool results from *completed* turns are replaced by a short stub
    with an excerpt (`_elide_tool_result`). Earlier turns are re-sent on
    every request, so a 20 KB OCR dump from three questions ago would
    otherwise be paid for again and again.
  - If the current turn alone grows past the budget, its oldest large tool
    results are elided the same way, keeping the newest ones intact.
  - Dropped turns leave a one-line digest (request -> reply) that the agent
    puts in the system prompt, so the conversation keeps its thread.
"""
from __future__ import annotations

import json
from collections import deque

from app.llm.base import Message, Role

# Rough chars-per-token ratio for English/JSON text. This is a deliberately
# cheap heuristic (no tokenizer dependency) -- it doesn't need to be exact,
# only good enough to keep history inside its budget. Actual token counts
# for accounting come from the provider's usage metadata (app/llm/usage.py).
_CHARS_PER_TOKEN = 4
# Flat per-message overhead (role, tool_call_id, JSON wrapper, etc.).
_MESSAGE_OVERHEAD_TOKENS = 4

# Default conversation-history token budget -- applies to the non-system
# tail ONLY; the system/personality message is always preserved in full
# and is not counted against this budget. Sized for Groq's 8000-TPM limit
# (legacy providers); 'tiered'/'gemini' use Settings.effective_max_context_tokens().
_DEFAULT_MAX_TOKENS = 1200

STALE_TOOL_RESULT_CHARS = 1500
"""Tool results longer than this are elided once their turn is over."""
_ELIDED_EXCERPT_CHARS = 300
_ELIDED_MARKER = "to save tokens; call the tool again"
_KEEP_RECENT_TOOL_RESULTS = 2
"""Within the current turn, the newest results are never elided."""
_MAX_DIGEST_TURNS = 5
_DIGEST_USER_CHARS = 160
_DIGEST_REPLY_CHARS = 200


def _approx_tokens(message: Message) -> int:
    """Cheap, deterministic token estimate -- no tokenizer required."""
    chars = len(message.content or "")
    for tool_call in message.tool_calls:
        chars += len(tool_call.name) + len(str(tool_call.arguments))
    return chars // _CHARS_PER_TOKEN + _MESSAGE_OVERHEAD_TOKENS


def _elide_tool_result(message: Message, reason: str) -> Message:
    """A compact stand-in for a large tool result: success/error kept, the
    data replaced by an excerpt and an explicit note that it was cut."""
    content = message.content or ""
    note = f"[{len(content):,} characters of output removed ({reason}) {_ELIDED_MARKER} if you need the full result]"
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "success" in parsed:
        stub: dict = {"success": parsed.get("success")}
        if parsed.get("error"):
            stub["error"] = str(parsed["error"])[:_ELIDED_EXCERPT_CHARS]
        data = parsed.get("data")
        excerpt = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        stub["data_excerpt"] = excerpt[:_ELIDED_EXCERPT_CHARS]
        stub["elided"] = note
        new_content = json.dumps(stub, ensure_ascii=False)
    else:
        new_content = content[:_ELIDED_EXCERPT_CHARS] + " ... " + note
    return message.model_copy(update={"content": new_content})


def _is_elided(message: Message) -> bool:
    return _ELIDED_MARKER in (message.content or "")


class ConversationContext:
    """Holds the running message history, trimmed to a max length and a
    max approximate token budget, in whole turns (see module docstring).

    The system message (position 0, if present) is always preserved and is
    not counted against either limit.
    """

    def __init__(self, max_messages: int = 30, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._messages: list[Message] = []
        self._digest: deque[str] = deque(maxlen=_MAX_DIGEST_TURNS)

    def set_system_prompt(self, content: str, cacheable_prefix_chars: int | None = None) -> None:
        """`cacheable_prefix_chars`: see Message.cacheable_prefix_chars."""
        system_message = Message(role=Role.SYSTEM, content=content, cacheable_prefix_chars=cacheable_prefix_chars)
        if self._messages and self._messages[0].role == Role.SYSTEM:
            self._messages[0] = system_message
        else:
            self._messages.insert(0, system_message)

    def add(self, message: Message) -> None:
        if message.role == Role.USER:
            self._elide_completed_turns()
        self._messages.append(message)
        self._trim()

    # ------------------------------------------------------------ structure
    def _split(self) -> tuple[list[Message], list[list[Message]]]:
        """(system head, turns). Messages before the first USER message (only
        possible after external manipulation) form a leading pseudo-turn."""
        has_system = bool(self._messages) and self._messages[0].role == Role.SYSTEM
        head = self._messages[:1] if has_system else []
        turns: list[list[Message]] = []
        for message in self._messages[len(head):]:
            if message.role == Role.USER or not turns:
                turns.append([message])
            else:
                turns[-1].append(message)
        return head, turns

    def _join(self, head: list[Message], turns: list[list[Message]]) -> None:
        self._messages = head + [m for turn in turns for m in turn]

    # ------------------------------------------------------------ trimming
    def _trim(self) -> None:
        head, turns = self._split()
        # A leading chunk that doesn't start with the user's words (orphaned
        # tool results / assistant calls) is invalid history once a real
        # turn exists after it.
        while len(turns) > 1 and turns[0][0].role != Role.USER:
            turns.pop(0)
        while len(turns) > 1 and (
            sum(len(t) for t in turns) + len(head) > self._max_messages
            or sum(_approx_tokens(m) for t in turns for m in t) > self._max_tokens
        ):
            self._remember_dropped(turns.pop(0))
        if turns and sum(_approx_tokens(m) for m in turns[-1]) > self._max_tokens:
            turns[-1] = self._shrink_current_turn(turns[-1])
        self._join(head, turns)

    def _shrink_current_turn(self, turn: list[Message]) -> list[Message]:
        """Elide the oldest large tool results of an over-budget current turn,
        keeping the newest few intact. Never removes a message."""
        tool_indexes = [i for i, m in enumerate(turn) if m.role == Role.TOOL]
        candidates = tool_indexes[:-_KEEP_RECENT_TOOL_RESULTS] if len(tool_indexes) > _KEEP_RECENT_TOOL_RESULTS else []
        turn = list(turn)
        for index in candidates:
            if sum(_approx_tokens(m) for m in turn) <= self._max_tokens:
                break
            message = turn[index]
            if len(message.content or "") > STALE_TOOL_RESULT_CHARS and not _is_elided(message):
                turn[index] = _elide_tool_result(message, "older step of this task")
        return turn

    def _elide_completed_turns(self) -> None:
        """Called when a new user turn starts: every turn so far is complete."""
        for index, message in enumerate(self._messages):
            if (message.role == Role.TOOL and len(message.content or "") > STALE_TOOL_RESULT_CHARS
                    and not _is_elided(message)):
                self._messages[index] = _elide_tool_result(message, "earlier request")

    def _remember_dropped(self, turn: list[Message]) -> None:
        if not turn or turn[0].role != Role.USER:
            return
        request = " ".join((turn[0].content or "").split())[:_DIGEST_USER_CHARS]
        reply = next((m.content for m in reversed(turn) if m.role == Role.ASSISTANT and m.content), "")
        reply = " ".join(reply.split())[:_DIGEST_REPLY_CHARS] or "(no reply)"
        self._digest.append(f"- User: {request} -> You: {reply}")

    # ------------------------------------------------------------ access
    def earlier_turns_digest(self) -> str | None:
        """One line per turn trimmed from history (oldest first), or None."""
        return "\n".join(self._digest) if self._digest else None

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        system = self._messages[0] if self._messages and self._messages[0].role == Role.SYSTEM else None
        self._messages = [system] if system else []
        self._digest.clear()
