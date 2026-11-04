"""Provider-agnostic LLM interface.

Garvis's agent core talks only to `LLMProvider`. Swapping the backend
(OmniRoute, Gemini, Groq, ... -- always an external API; Garvis never runs
an AI model locally) must never require touching agent code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from app.llm.usage import TokenUsage


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM. Not yet executed."""

    id: str
    name: str
    arguments: dict[str, Any]
    provider_metadata: dict[str, Any] | None = None
    """Opaque per-call data a provider needs echoed back verbatim on the
    next request -- e.g. Gemini 3's thought signature
    (`extra_content.google.thought_signature`), without which multi-turn
    function calling is rejected. Providers that don't use it ignore it."""


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    """Set on Role.TOOL messages: which ToolCall this result answers."""
    name: str | None = None
    """Set on Role.TOOL messages: the tool name, for readability."""
    cacheable_prefix_chars: int | None = None
    """Set on the SYSTEM message: content[:n] is identical on every turn
    (operating rules + personality); the rest changes per turn (memory notes,
    digest, live desktop). Providers with prompt caching put a cache
    breakpoint at this boundary; others ignore it."""


class ToolSpec(BaseModel):
    """JSON-schema description of a tool, as handed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class LLMResponse(BaseModel):
    message: Message
    finish_reason: Literal["stop", "tool_calls", "length", "error"] = "stop"
    usage: TokenUsage | None = None
    """Token counts the provider reported for this request (None = not reported)."""
    served_model: str | None = None
    """The model the provider says produced this response."""
    request_id: str | None = None


def served_model_matches(requested_model: str, served: str | None) -> bool:
    """True if `served` names the requested model (an optional '<provider>/'
    prefix or a '-YYYYMMDD' snapshot suffix is accepted)."""
    if not served:
        return False
    expected = requested_model.lower()
    name = served.lower()
    if "/" in name and "/" not in expected:
        name = name.rpartition("/")[2]
    suffix = name[len(expected) + 1:]
    return name == expected or (name.startswith(expected + "-") and suffix.isdigit())


class LLMProvider(ABC):
    """Abstract base for local (or remote) chat-completion backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send ``messages`` (+ optional tool specs) and return one response."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve requests."""
