"""Direct Anthropic Messages API implementation of LLMProvider (official
`anthropic` Python SDK). The only model provider Garvis uses: every tier in
config/models.json is Claude Haiku 5.5 through this module, with no
fallback to any other model (app/llm/router.py, "Single-model policy").

Structure mirrors the other providers (app/llm/omniroute.py, gemini.py):
one module, one class, Garvis's provider-neutral Message/ToolSpec in,
LLMResponse out. Anthropic-specific handling, all contained here:

  - Messages: SYSTEM messages become the top-level `system` blocks; TOOL
    messages become `tool_result` blocks in a user turn (consecutive results
    share one turn, as the API requires); consecutive same-role turns are
    merged. A tool call left without a result (an interrupted turn) gets a
    synthetic error result, and a result whose call is gone becomes plain
    text, so one bad history entry can't make every later request a 400.
    Tool-call ids are kept as-is when valid and mapped deterministically
    when not (e.g. ids in a conversation recorded before this provider).
  - Thinking: sent as configured (`"disabled"` in the shipped registry) --
    Garvis's message model doesn't carry thinking blocks between requests,
    and with thinking off there is nothing that must be passed back.
  - Effort: the registry entry's `reasoning_effort` is sent as
    `output_config.effort` (Claude Haiku 5.5: low/medium/high with thinking
    off). No temperature/top_p/top_k is ever sent (Haiku 5.5 rejects them).
  - Prompt caching (`prompt_cache` "5m"/"1h"): one explicit breakpoint at the
    end of the stable part of the system prompt (Message.
    cacheable_prefix_chars) -- which also covers the tool definitions,
    rendered before it -- plus top-level automatic caching for the growing
    message tail inside a tool loop. Volatile per-turn context (desktop,
    memory notes) sits after the system breakpoint.
  - Usage: input/output tokens, cache reads, and cache writes split into
    5-minute / 1-hour when the API provides the split. Nothing is estimated.
  - The SDK's own retries are off (max_retries=0): retries (same model
    only) are decided in one place, TieredLLMRouter.
  - stop_reason "refusal" raises LLMRefusalError (not rerouted elsewhere);
    "max_tokens" returns finish_reason "length" with any half-written tool
    calls dropped, so a truncated call is never executed.
  - The reply's `model` must be the requested model (LLMModelMismatchError
    otherwise, with the billed usage attached).
  - The API key comes from Settings.anthropic_api_key (ANTHROPIC_API_KEY);
    it is never logged, and the base URL is pinned explicitly. Without a key
    the provider still builds (Garvis starts), but every request fails at
    once with LLMAuthenticationError -- nothing is sent.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import anthropic

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBillingError,
    LLMError,
    LLMModelMismatchError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec, served_model_matches
from app.llm.usage import PriceCard, TokenUsage

logger = get_logger("llm.anthropic")

DEFAULT_BASE_URL = "https://api.anthropic.com"
_ERROR_TEXT_MAX = 300
_VALID_TOOL_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_CACHE_TTLS = {"5m": None, "1h": "1h"}
_INTERRUPTED_RESULT = "No result: this step was interrupted before it finished."


def tool_use_id(raw: str) -> str:
    """Anthropic accepts tool ids matching ^[a-zA-Z0-9_-]+$. Valid ids pass
    through unchanged; others are mapped deterministically (same input ->
    same output, so a call and its result still match)."""
    if _VALID_TOOL_ID.match(raw or ""):
        return raw
    digest = hashlib.sha256((raw or "").encode("utf-8")).hexdigest()[:16]
    return f"toolu_garvis_{digest}"


def _tool_result_block(message: Message) -> dict[str, Any]:
    content = message.content or "(no output)"
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id(message.tool_call_id or ""),
                             "content": content}
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("success") is False:
        block["is_error"] = True
    return block


def _system_blocks(messages: list[Message], cache_control: dict | None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message in messages:
        if message.role != Role.SYSTEM or not message.content:
            continue
        split = message.cacheable_prefix_chars
        if cache_control and split and 0 < split <= len(message.content):
            stable, rest = message.content[:split], message.content[split:].lstrip("\n")
            blocks.append({"type": "text", "text": stable, "cache_control": cache_control})
            if rest:
                blocks.append({"type": "text", "text": rest})
        else:
            blocks.append({"type": "text", "text": message.content})
    return blocks


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Garvis history (minus SYSTEM) -> a valid Anthropic `messages` list."""
    turns: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.USER:
            append("user", [{"type": "text", "text": message.content or "(empty message)"}])
        elif message.role == Role.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if message.content.strip():
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append({"type": "tool_use", "id": tool_use_id(call.id), "name": call.name,
                               "input": call.arguments})
            append("assistant", blocks)
        elif message.role == Role.TOOL:
            append("user", [_tool_result_block(message)])
    # The conversation must start with the user's words.
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return _repair_tool_pairing(turns)


def _interrupted_results(ids: list[str]) -> list[dict[str, Any]]:
    return [{"type": "tool_result", "tool_use_id": i, "content": _INTERRUPTED_RESULT, "is_error": True}
            for i in ids]


def _repair_tool_pairing(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool_use needs a tool_result in the very next user turn, results
    first; a result needs its call in the turn before it. `turns` alternate
    roles and start with the user. History should already satisfy this --
    this keeps one interrupted step from breaking every later request."""
    repaired: list[dict[str, Any]] = []
    for turn in turns:
        if turn["role"] == "user":
            previous = repaired[-1] if repaired and repaired[-1]["role"] == "assistant" else None
            open_ids = {b["id"] for b in previous["content"] if b["type"] == "tool_use"} if previous else set()
            results = [b for b in turn["content"] if b["type"] == "tool_result" and b["tool_use_id"] in open_ids]
            others = []
            for block in turn["content"]:
                if block["type"] == "tool_result" and block["tool_use_id"] not in open_ids:
                    text = block["content"] if isinstance(block["content"], str) else json.dumps(block["content"])
                    others.append({"type": "text", "text": f"[Earlier tool output] {text}"})
                elif block["type"] != "tool_result":
                    others.append(block)
            answered = {b["tool_use_id"] for b in results}
            missing = _interrupted_results(sorted(open_ids - answered))
            repaired.append({"role": "user", "content": results + missing + others})
        else:
            repaired.append(turn)
    # The request must end with a user turn (Claude Haiku 5.5 rejects prefill).
    if repaired and repaired[-1]["role"] == "assistant":
        pending = [b["id"] for b in repaired[-1]["content"] if b["type"] == "tool_use"]
        repaired.append({"role": "user", "content": _interrupted_results(pending) if pending
                         else [{"type": "text", "text": "Continue."}]})
    return repaired


def parse_usage(usage: Any) -> TokenUsage | None:
    """TokenUsage from an SDK `Usage` object (None if absent)."""
    if usage is None:
        return None
    creation = getattr(usage, "cache_creation", None)
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    split_5m = getattr(creation, "ephemeral_5m_input_tokens", None) if creation is not None else None
    split_1h = getattr(creation, "ephemeral_1h_input_tokens", None) if creation is not None else None
    has_split = split_5m is not None or split_1h is not None
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_5m_tokens=(split_5m or 0) if has_split else 0,
        cache_write_1h_tokens=(split_1h or 0) if has_split else 0,
        cache_write_unknown_ttl_tokens=0 if has_split else written,
    )


def _error_detail(exc: anthropic.APIStatusError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return str(message)[:_ERROR_TEXT_MAX]


def _map_error(exc: Exception, model: str, timeout: float) -> LLMError:
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(f"Claude ({model}) didn't answer within {timeout:g}s.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMUnavailableError("Garvis can't reach the Anthropic API (api.anthropic.com) -- check the internet "
                                   "connection.")
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMAuthenticationError(
            "Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY in Garvis's .env "
            "(python scripts/check_anthropic.py --set-key)."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMAuthenticationError(f"Anthropic refused {model} for this API key (403): {_error_detail(exc)}")
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after") if getattr(exc, "response", None) else None
        hint = f" (retry after {retry_after}s)" if retry_after else ""
        return LLMRateLimitError(f"Anthropic rate limit reached for {model}{hint}: {_error_detail(exc)}")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMUnavailableError(f"Anthropic doesn't know the model {model!r} (404): {_error_detail(exc)}")
    if isinstance(exc, (anthropic.BadRequestError, anthropic.RequestTooLargeError,
                        anthropic.UnprocessableEntityError)):
        detail = _error_detail(exc)
        if "credit balance" in detail.lower() or "billing" in detail.lower():
            # Account-level: no request can succeed until the balance is topped up.
            return LLMBillingError(f"Anthropic API billing problem: {detail}")
        return LLMBadRequestError(f"Claude ({model}) rejected the request ({exc.status_code}): {detail}")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMUnavailableError(f"Anthropic API error for {model}: {exc.status_code} {_error_detail(exc)}")
    return LLMError(f"Unexpected Anthropic SDK error for {model}: {type(exc).__name__}")


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 45.0,
        reasoning_effort: str | None = None,
        thinking: str | None = "disabled",
        max_output_tokens: int | None = None,
        prompt_cache: str | None = "5m",
        verify_served_model: bool = True,
        pricing: PriceCard | None = None,
        client: Any = None,
    ) -> None:
        """`client`: an AsyncAnthropic-compatible object (tests inject one);
        by default one is created on first use with the SDK's retries off."""
        if not model:
            raise ConfigurationError("Anthropic model is not configured.")
        if prompt_cache not in (None, "off", *_CACHE_TTLS):
            raise ConfigurationError(f"prompt_cache must be 'off', '5m' or '1h', not {prompt_cache!r}.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._effort = reasoning_effort
        self._thinking = thinking
        self._max_output_tokens = max_output_tokens or 8192
        self._prompt_cache = None if prompt_cache in (None, "off") else prompt_cache
        self._verify_served_model = verify_served_model
        self.pricing = pricing
        """Read by TieredLLMRouter for cost accounting and budget checks."""
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:  # never include the key
        return f"AnthropicProvider(model={self._model!r}, base_url={self._base_url!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key, base_url=self._base_url, timeout=self._timeout, max_retries=0,
            )
        return self._client

    def _cache_control(self) -> dict | None:
        if self._prompt_cache is None:
            return None
        ttl = _CACHE_TTLS[self._prompt_cache]
        return {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}

    def build_request(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> dict[str, Any]:
        """Keyword arguments for `client.messages.create()`."""
        cache_control = self._cache_control()
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "messages": to_anthropic_messages(messages),
        }
        system = _system_blocks(messages, cache_control)
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        if self._thinking:
            request["thinking"] = {"type": self._thinking}
        if self._effort:
            request["output_config"] = {"effort": self._effort}
        if cache_control is not None:
            if tools and not any("cache_control" in block for block in system):
                # No stable system prefix marked: at least keep the tool definitions cached.
                request["tools"][-1] = {**request["tools"][-1], "cache_control": cache_control}
            request["cache_control"] = cache_control  # automatic breakpoint on the growing tail
        return request

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        if not self._api_key and self._client is None:
            raise LLMAuthenticationError(
                "Claude Haiku 5.5 can't answer: ANTHROPIC_API_KEY is not set. Add it to Garvis's .env "
                "(python scripts/check_anthropic.py --set-key) and restart Garvis. No other model is used."
            )
        request = self.build_request(messages, tools)
        log_extra(logger, 20, "anthropic.chat.request", model=self._model, num_messages=len(request["messages"]),
                  num_tools=len(request.get("tools", [])), effort=self._effort, prompt_cache=self._prompt_cache)
        started = time.monotonic()
        try:
            response = await self._get_client().messages.create(**request)
        except anthropic.AnthropicError as exc:
            raise _map_error(exc, self._model, self._timeout) from exc

        usage = parse_usage(getattr(response, "usage", None))
        served_model = getattr(response, "model", None)
        request_id = getattr(response, "id", None)
        stop_reason = getattr(response, "stop_reason", None)
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                texts.append(block.text)
            elif block_type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
        content = "".join(texts)

        if stop_reason == "max_tokens" and tool_calls:
            # A call cut off mid-arguments must never run.
            log_extra(logger, 30, "anthropic.chat.truncated_tool_calls_dropped", model=self._model,
                      dropped=[c.name for c in tool_calls])
            tool_calls = []
        if stop_reason == "tool_use" and tool_calls:
            finish_reason = "tool_calls"
        elif stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        log_extra(
            logger, 20, "anthropic.chat.response", model=self._model, served_model=served_model,
            request_id=request_id, sdk_request_id=getattr(response, "_request_id", None), stop_reason=stop_reason,
            num_tool_calls=len(tool_calls), latency_ms=int((time.monotonic() - started) * 1000),
            usage=usage.model_dump(exclude_none=True) if usage else None, **content_fingerprint(content),
        )
        if self._verify_served_model and not served_model_matches(self._model, served_model):
            raise LLMModelMismatchError(
                f"Requested {self._model}, but the Anthropic API reports {served_model or 'no model'}. "
                "The reply was discarded.",
                usage=usage, served_model=served_model, request_id=request_id,
            )
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMRefusalError(
                "Claude declined this request" + (f" (safety category: {category})" if category else "")
                + ". Try rephrasing it.",
                usage=usage, served_model=served_model, request_id=request_id,
            )
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls),
            finish_reason=finish_reason, usage=usage, served_model=served_model, request_id=request_id,
        )

    async def health_check(self) -> bool:
        if not self._api_key and self._client is None:
            return False
        try:
            await self._get_client().models.retrieve(self._model)
            return True
        except anthropic.AnthropicError:
            return False
