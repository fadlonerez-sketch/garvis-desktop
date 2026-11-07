"""OmniRoute-backed implementation of LLMProvider.

OmniRoute (npm `omniroute`) is a local gateway that exposes one
OpenAI-compatible API (http://127.0.0.1:20128/v1) in front of the upstream
providers connected in its dashboard. Garvis uses it strictly as a
*transport*: TieredLLMRouter still decides tier, model order and failover,
and every request names one explicit `<provider>/<model>` id (e.g.
`gemini/gemini-3.1-flash-lite`). OmniRoute's own routing -- `auto/*`
combos and bare combo names -- is refused here, so a request can never be
re-routed to a provider Garvis didn't choose.

Structure deliberately mirrors app/llm/openrouter.py / app/llm/gemini.py
(see their docstrings for why providers stay separate modules). OmniRoute-
specific differences, all handled here:

  - Localhost only: the base URL must be a loopback address. The client key
    is never sent anywhere else, even if GARVIS_OMNIROUTE_BASE_URL is
    mis-set.
  - Auth: OmniRoute's local client API key (created in its dashboard), sent
    as a Bearer token. Read from Settings.omniroute_api_key
    (GARVIS_OMNIROUTE_API_KEY); never logged.
  - Gemini 3 thought signatures: OmniRoute stores them itself (keyed by
    tool-call id) and turns signature-less historical calls (from a mid-turn
    failover) into inert context, so no placeholder is injected here. Any
    `tool_calls[].extra_content` it does return is kept in
    ToolCall.provider_metadata and echoed back verbatim.
  - Tool schemas for `gemini/...` models are reduced with the same
    sanitize_schema() GeminiProvider uses; other upstreams get them as-is.
  - Errors name the model and keep OmniRoute's own message, so the user
    sees *which* upstream failed and why (quota exhausted, provider not
    connected, ...). Nothing is retried or re-routed here (bounded retries
    live in TieredLLMRouter; OmniRoute itself retries a 429 twice and a
    502/503/504 once before answering).
  - Claude via the `anthropic` connection (verified against OmniRoute
    3.8.50's source): `reasoning_effort` would become a manual thinking
    budget, which Claude Haiku 5.5 rejects, and Claude thinking signatures
    don't survive the OpenAI format -- so Haiku entries send
    `thinking: {"type": "disabled"}` and no effort. OmniRoute raises
    `max_tokens` to 32000 whenever tools are sent.
  - Usage: `prompt_tokens` includes cache reads (`prompt_tokens_details.
    cached_tokens`); cache writes come separately as `prompt_tokens_details.
    cache_creation_tokens`, without a 5m/1h split. Parsed into TokenUsage;
    nothing is estimated when a field is missing.
  - Every request opts out of OmniRoute's semantic response cache (its key
    ignores tools and tool calls) and its memory/skills injection, so the
    prompt Garvis builds is the prompt the model sees.
  - With `verify_served_model`, a response whose `model` isn't the requested
    upstream model raises LLMModelMismatchError (usage attached, since the
    request was billed) instead of being used.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMError,
    LLMModelMismatchError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec, served_model_matches
from app.llm.gemini import sanitize_schema
from app.llm.usage import PriceCard, TokenUsage

logger = get_logger("llm.omniroute")

DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_ERROR_TEXT_MAX = 300
_OPT_OUT_HEADERS = {"x-omniroute-no-cache": "true", "x-omniroute-no-memory": "true"}


def validate_base_url(base_url: str) -> str:
    """Return the normalized URL, or raise if it isn't a local http(s) URL."""
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS:
        raise ConfigurationError(
            f"OmniRoute must stay localhost-only, but GARVIS_OMNIROUTE_BASE_URL is {base_url!r}. "
            f"Use {DEFAULT_BASE_URL}."
        )
    return base_url.rstrip("/")


def validate_model_id(model: str) -> str:
    """An explicit `<provider>/<model>` id -- never an OmniRoute auto/combo route."""
    provider, sep, rest = model.partition("/")
    if not sep or not provider or not rest:
        raise ConfigurationError(
            f"OmniRoute model {model!r} must be an explicit '<provider>/<model>' id "
            "(e.g. 'gemini/gemini-3.1-flash-lite'); bare names can resolve to OmniRoute combos."
        )
    if provider.lower() in ("auto", "combo"):
        raise ConfigurationError(
            f"OmniRoute model {model!r} is an automatic route. Garvis selects models itself -- "
            "use an explicit '<provider>/<model>' id."
        )
    return model


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }
    extra = (tc.provider_metadata or {}).get("extra_content")
    if isinstance(extra, dict):
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


def _tool_spec_to_dict(spec: ToolSpec, gemini_schema: bool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            # Gemini's function declarations accept only a JSON-schema subset;
            # other upstreams get the schema unchanged.
            "parameters": sanitize_schema(spec.parameters) if gemini_schema else spec.parameters,
        },
    }


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        error = data.get("error") if isinstance(data, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        if message:
            return str(message)[:_ERROR_TEXT_MAX]
    except ValueError:
        pass
    return response.text[:_ERROR_TEXT_MAX]


def _map_http_error(exc: httpx.HTTPStatusError, model: str) -> LLMError:
    status = exc.response.status_code
    detail = _error_text(exc.response)
    if "no active credentials for provider" in detail.lower():
        # OmniRoute's 401 for "that upstream isn't connected" -- Garvis's own key was accepted.
        return LLMUnavailableError(
            f"{model} isn't usable yet: its provider has no connection in OmniRoute -- add that "
            f"provider's free API key in the OmniRoute dashboard. ({detail})"
        )
    if status == 401:
        return LLMAuthenticationError(
            f"OmniRoute rejected Garvis's client key (401). Check GARVIS_OMNIROUTE_API_KEY. {detail}".strip()
        )
    if status == 403:
        return LLMAuthenticationError(f"OmniRoute refused {model} (403): {detail}".strip())
    if status == 429:
        return LLMRateLimitError(
            f"Free quota or rate limit reached for {model} -- the request was refused, not billed. {detail}".strip()
        )
    if status in (400, 413, 422):
        return LLMBadRequestError(f"{model} rejected the request ({status}): {detail}".strip())
    if status == 404:
        return LLMUnavailableError(
            f"OmniRoute can't serve {model} (404) -- is that provider connected in the OmniRoute dashboard? "
            f"{detail}".strip()
        )
    return LLMUnavailableError(f"OmniRoute returned an error for {model}: {status} {detail}".strip())


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_usage(data: dict) -> TokenUsage | None:
    """TokenUsage from an OpenAI-format `usage` object as OmniRoute returns it,
    or None when the response carries no usage at all."""
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _int_or_none(usage.get("prompt_tokens"))
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    cached = _int_or_none(details.get("cached_tokens")) or 0
    reasoning = _int_or_none(usage.get("reasoning_tokens"))
    completion_details = usage.get("completion_tokens_details")
    if reasoning is None and isinstance(completion_details, dict):
        reasoning = _int_or_none(completion_details.get("reasoning_tokens"))
    return TokenUsage(
        input_tokens=None if prompt is None else max(0, prompt - cached),
        output_tokens=_int_or_none(usage.get("completion_tokens")),
        cache_read_tokens=cached,
        cache_write_unknown_ttl_tokens=_int_or_none(details.get("cache_creation_tokens")) or 0,
        reasoning_tokens=reasoning,
    )


class OmniRouteProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
        max_output_tokens: int | None = None,
        verify_served_model: bool = False,
        pricing: PriceCard | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "OmniRoute client API key is not configured. Create one in the OmniRoute dashboard "
                "and save it with: python scripts/check_omniroute.py --set-key"
            )
        if not model:
            raise ConfigurationError("OmniRoute model is not configured.")
        self._api_key = api_key
        self._model = validate_model_id(model)
        self._base_url = validate_base_url(base_url)
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._thinking = thinking
        self._max_output_tokens = max_output_tokens
        self._verify_served_model = verify_served_model
        self.pricing = pricing
        """Read by TieredLLMRouter for cost accounting and budget checks."""

    @property
    def model(self) -> str:
        return self._model

    @property
    def upstream(self) -> str:
        return self._model.partition("/")[0].lower()

    @property
    def upstream_model(self) -> str:
        return self._model.partition("/")[2]

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:  # never include the key
        return f"OmniRouteProvider(model={self._model!r}, base_url={self._base_url!r})"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", **_OPT_OUT_HEADERS}

    def build_request_body(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [_message_to_dict(m) for m in messages],
            # LLMProvider.chat() returns one complete response; no SSE.
            "stream": False,
        }
        if tools:
            gemini_schema = self._model.partition("/")[0].lower() == "gemini"
            body["tools"] = [_tool_spec_to_dict(t, gemini_schema) for t in tools]
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        if self._thinking:
            body["thinking"] = {"type": self._thinking}
        if self._max_output_tokens:
            body["max_tokens"] = self._max_output_tokens
        return body

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        body = self.build_request_body(messages, tools)
        log_extra(logger, 20, "omniroute.chat.request", model=self._model, num_messages=len(messages))

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.post("/chat/completions", json=body, headers=self._headers())
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Garvis can't reach OmniRoute at {self._base_url}. Is it running? "
                "(OMNIROUTE_SERVER_HOST=127.0.0.1 omniroute serve)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"OmniRoute did not respond in time for {self._model}.") from exc
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(exc, self._model) from exc

        try:
            data = response.json()
            choice = data["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"OmniRoute returned an unexpected response shape for {self._model}.") from exc

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
            raise LLMError(f"OmniRoute returned a malformed tool call for {self._model}.") from exc

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
        reasoning = raw_message.get("reasoning_content") or raw_message.get("reasoning") or ""
        usage = parse_usage(data)
        served_model = data.get("model") if isinstance(data.get("model"), str) else None
        request_id = data.get("id") if isinstance(data.get("id"), str) else None
        served_provider = response.headers.get("X-OmniRoute-Provider")
        log_extra(
            logger,
            20,
            "omniroute.chat.response",
            model=self._model,
            served_model=served_model,
            served_provider=served_provider,
            fallback_attempts=response.headers.get("X-OmniRoute-Fallback-Attempts"),
            request_id=request_id,
            finish_reason=finish_reason,
            num_tool_calls=len(tool_calls),
            reasoning_chars=len(reasoning) if isinstance(reasoning, str) else 0,
            usage=usage.model_dump(exclude_none=True) if usage else None,
            **content_fingerprint(message.content),
        )
        if self._verify_served_model and not (
            served_model_matches(self.upstream_model, served_model)
            and (not served_provider or served_provider.lower() == self.upstream)
        ):
            raise LLMModelMismatchError(
                f"Requested {self._model}, but OmniRoute reports the reply came from "
                f"{served_provider or self.upstream}/{served_model or 'an unnamed model'}. The reply was discarded.",
                usage=usage,
                served_model=served_model,
                request_id=request_id,
            )
        return LLMResponse(
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            served_model=served_model,
            request_id=request_id,
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=5.0) as client:
                response = await client.get("/models", headers=self._headers())
                return response.status_code == 200
        except httpx.HTTPError:
            return False
