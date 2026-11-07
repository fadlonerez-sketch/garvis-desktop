"""Groq-backed implementation of LLMProvider.

Groq exposes an OpenAI-compatible chat completions API: tool call `id`s are
threaded through and tool arguments are JSON-encoded strings on the wire.
The module is message/tool-spec conversion helpers plus a thin
`LLMProvider` subclass with `chat()` + `health_check()`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec

logger = get_logger("llm.groq")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        # The OpenAI-compatible API matches tool results
        # back to calls by id, so the id has to survive the round trip and
        # arguments are sent back as a JSON string, not a dict.
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in message.tool_calls
        ]
    if message.role == Role.TOOL:
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    return payload


def _tool_spec_to_dict(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _map_http_error(exc: httpx.HTTPStatusError) -> LLMError:
    status = exc.response.status_code
    if status in (401, 403):
        return LLMAuthenticationError("Groq rejected the request credentials. Check GARVIS_GROQ_API_KEY.")
    if status == 429:
        return LLMRateLimitError("Groq rate limit exceeded. Please try again shortly.")
    return LLMUnavailableError(f"Groq returned an error: {status} {exc.response.text[:300]}")


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("Groq API key is not configured. Set GARVIS_GROQ_API_KEY.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [_message_to_dict(m) for m in messages],
        }
        if tools:
            body["tools"] = [_tool_spec_to_dict(t) for t in tools]

        log_extra(logger, 20, "groq.chat.request", model=self._model, num_messages=len(messages))

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.post("/chat/completions", json=body, headers=self._headers())
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError("Garvis can't reach Groq. Check your network connection.") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Groq did not respond in time.") from exc
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(exc) from exc

        try:
            data = response.json()
            raw_message = data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Groq returned an unexpected response shape.") from exc

        tool_calls_raw = raw_message.get("tool_calls") or []
        try:
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or str(uuid.uuid4()),
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"].get("arguments") or "{}"),
                )
                for tc in tool_calls_raw
            ]
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMError("Groq returned a malformed tool call.") from exc

        message = Message(
            role=Role.ASSISTANT,
            content=raw_message.get("content") or "",
            tool_calls=tool_calls,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        log_extra(
            logger,
            20,
            "groq.chat.response",
            finish_reason=finish_reason,
            num_tool_calls=len(tool_calls),
            **content_fingerprint(message.content),
        )
        return LLMResponse(message=message, finish_reason=finish_reason)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=5.0) as client:
                response = await client.get("/models", headers=self._headers())
                return response.status_code == 200
        except httpx.HTTPError:
            return False
