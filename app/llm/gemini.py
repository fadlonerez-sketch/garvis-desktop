"""Google Gemini implementation of LLMProvider, via Gemini's
OpenAI-compatible endpoint (https://ai.google.dev/gemini-api/docs/openai).

Structure deliberately mirrors app/llm/openrouter.py / app/llm/groq.py (see
their docstrings for why providers stay separate modules rather than
sharing a base). Gemini-specific differences, all handled here:

  - Auth: an API key from Google AI Studio (https://aistudio.google.com/apikey),
    sent as a Bearer token. Read from Settings.gemini_api_key
    (GARVIS_GEMINI_API_KEY); never logged.
  - Thought signatures: Gemini 3 returns
    `tool_calls[].extra_content.google.thought_signature` and REJECTS the
    next request (HTTP 400) unless it is echoed back verbatim. It is kept
    in ToolCall.provider_metadata and replayed. Tool calls that came from a
    different model (a mid-turn failover) have no signature; those get
    Google's documented placeholder `skip_thought_signature_validator`.
  - reasoning_effort: per-model thinking level (minimal/low/medium/high),
    configured per registry entry -- low latency matters for voice.
  - Tool schemas are reduced to the JSON-schema subset Gemini's function
    declarations accept (no `title`/`default`; `anyOf[X, null]` becomes
    X + `nullable`).
  - Errors: an invalid API key comes back as HTTP 400 ("API key not
    valid"), not 401, so it's mapped to LLMAuthenticationError explicitly.
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

logger = get_logger("llm.gemini")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
SKIP_SIGNATURE = "skip_thought_signature_validator"

_SCHEMA_KEYS = {
    "type", "description", "properties", "required", "items", "enum",
    "minimum", "maximum", "minItems", "maxItems", "format", "nullable",
}
_ERROR_TEXT_MAX = 300


def sanitize_schema(schema: Any) -> Any:
    """Reduce a Pydantic-generated JSON schema to what Gemini accepts."""
    if not isinstance(schema, dict):
        return schema
    node = dict(schema)
    any_of = node.pop("anyOf", None)
    if isinstance(any_of, list):
        variants = [v for v in any_of if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(variants) == 1 and isinstance(variants[0], dict):
            merged = {**variants[0], **{k: v for k, v in node.items() if k != "default"}}
            if len(variants) < len(any_of):
                merged["nullable"] = True
            node = merged
        elif variants:
            node = {**variants[0], **node}  # unsupported union: keep the first shape
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: sanitize_schema(prop) for name, prop in value.items()}
        elif key == "items":
            out[key] = sanitize_schema(value)
        elif key in _SCHEMA_KEYS:
            out[key] = value
    return out


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }
    extra = (tc.provider_metadata or {}).get("extra_content")
    if not (isinstance(extra, dict) and extra.get("google", {}).get("thought_signature")):
        extra = {"google": {"thought_signature": SKIP_SIGNATURE}}
    payload["extra_content"] = extra
    return payload


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_to_dict(tc) for tc in message.tool_calls]
        if not message.content:
            payload["content"] = None
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
            "parameters": sanitize_schema(spec.parameters),
        },
    }


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, list) and data:
            data = data[0]
        message = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        if message:
            return str(message)[:_ERROR_TEXT_MAX]
    except ValueError:
        pass
    return response.text[:_ERROR_TEXT_MAX]


def _map_http_error(exc: httpx.HTTPStatusError, model: str) -> LLMError:
    status = exc.response.status_code
    detail = _error_text(exc.response)
    if status in (401, 403) or (status == 400 and "api key" in detail.lower()):
        return LLMAuthenticationError(
            f"Gemini rejected the API key ({status}). Check GARVIS_GEMINI_API_KEY. {detail}".strip()
        )
    if status == 429:
        return LLMRateLimitError(f"Gemini rate limit/quota exceeded for {model}. {detail}".strip())
    if status == 404:
        return LLMUnavailableError(f"Gemini model '{model}' was not found or isn't available to this key.")
    return LLMUnavailableError(f"Gemini returned an error for {model}: {status} {detail}".strip())


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        reasoning_effort: str | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "Gemini API key is not configured. Create one at https://aistudio.google.com/apikey "
                "and set GARVIS_GEMINI_API_KEY."
            )
        if not model:
            raise ConfigurationError("Gemini model is not configured.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort

    @property
    def model(self) -> str:
        return self._model

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
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort

        log_extra(logger, 20, "gemini.chat.request", model=self._model, num_messages=len(messages))

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.post("/chat/completions", json=body, headers=self._headers())
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError("Garvis can't reach Gemini. Check your network connection.") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Gemini did not respond in time.") from exc
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(exc, self._model) from exc

        try:
            data = response.json()
            choice = data["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Gemini returned an unexpected response shape.") from exc

        tool_calls_raw = raw_message.get("tool_calls") or []
        try:
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or str(uuid.uuid4()),
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"].get("arguments") or "{}"),
                    provider_metadata={"extra_content": tc["extra_content"]} if tc.get("extra_content") else None,
                )
                for tc in tool_calls_raw
            ]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("Gemini returned a malformed tool call.") from exc

        content = raw_message.get("content") or ""
        if isinstance(content, list):  # defensive: parts-style content
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        message = Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
        if tool_calls:
            finish_reason = "tool_calls"
        elif choice.get("finish_reason") == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"
        log_extra(
            logger,
            20,
            "gemini.chat.response",
            model=self._model,
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
