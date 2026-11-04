"""Request routing: `TieredLLMRouter` (GARVIS_LLM_PROVIDER=tiered).

Deterministic, local rules (`classify_tier()` -- no network call, no model
inference) put each request in a SMALL/MEDIUM/LARGE tier. A tier is not a
choice of model: every tier is served by the one configured model (Claude
Haiku 5.5, at a tier-specific effort level). See "Single-model policy"
below -- a failed request is reported, never sent to another model.

The former two-provider `LLMRouter` was removed: it routed requests between
two different models. Its keyword rules (`_classify()` and helpers) remain,
used by classify_tier().
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMBilledError,
    LLMBudgetExceededError,
    LLMError,
    LLMModelMismatchError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import get_logger, log_extra
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolSpec
from app.llm.registry import ModelTier
from app.llm.usage import BudgetGuard, TaskMeter, TokenUsage, UsageLedger, current_task, estimate_tokens

logger = get_logger("llm.router")

_MAX_SIMPLE_CONTEXT_MESSAGES = 3
"""Beyond a fresh question (or one already-answered exchange), the turn is
treated as needing real conversational memory -- not SMALL (low effort)."""

_MAX_SIMPLE_WORDS = 20
"""A "simple" request must also be answerable briefly. Long requests skip
past SMALL even if they otherwise look simple --
length alone never qualifies a request as simple, but it can disqualify
one."""

# Multi-word phrases (not bare words) wherever a bare word would be
# ambiguous, so a question *about* a word ("what does 'search' mean?")
# doesn't trip these. Kept short and skimmable on purpose -- this is a
# handful of unambiguous signals, not an exhaustive keyword list.
_TOOL_PHRASES = (
    "search for", "search the web", "web search", "look up", "look for",
    "browse to", "open the browser", "navigate to", "go to http",
    "check my email", "send an email", "draft an email", "read my email",
    "my emails", "book a", "find a restaurant", "find restaurants",
    "places near", "near me", "weather in", "weather for", "latest news",
    "remind me", "schedule a", "open chrome", "open notepad", "click on",
    "research", "screenshot", "double click", "double-click", "right click",
    "right-click", "scroll down", "scroll up", "switch to", "press enter",
    "open whatsapp",
)
_TOOL_WORDS = ("gmail", "inbox")

_COMPLEX_PHRASES = (
    "pros and cons", "difference between", "which is better",
    "help me decide", "step by step", "figure out", "write a function",
    "write some code", "write code", "fix this bug", "fix my code",
)
_COMPLEX_WORDS = (
    "why", "explain", "compare", "analyze", "analyse", "debug", "refactor",
    "algorithm", "architecture", "recommend", "traceback",
)

_TOOL_WORD_RE = re.compile(r"\b(?:" + "|".join(_TOOL_WORDS) + r")\b", re.IGNORECASE)
_COMPLEX_WORD_RE = re.compile(r"\b(?:" + "|".join(_COMPLEX_WORDS) + r")\b", re.IGNORECASE)

# A curated allowlist of unambiguously-simple request shapes -- see spec
# examples (capital-of, unit conversion, "stand for", short rewrite/
# translate, arithmetic, greetings). Anything not matching one of these
# falls through to MEDIUM by default.
_SIMPLE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^what(?:'s| is) the capital of\b",
        r"^how (?:many|much)\b.+\b(?:is|are|in)\b",
        r"^convert\b.+\bto\b",
        r"^what(?:'s| is| does)\b.+\bstand for\b",
        r"^(?:say|translate) this (?:in|to|into)\b",
        r"^translate\b",
        r"^define\b",
        r"^rewrite this\b",
        r"^spell\b",
        r"^\s*\d+(?:\.\d+)?\s*[+\-*/]\s*\d+",
        r"^(?:hi|hello|hey|thanks|thank you|good morning|good night)\b",
        r"^what time is it\b",
    )
)


def _has_tool_activity(messages: list[Message]) -> bool:
    """True once this turn (or an earlier one still in context) has already
    involved a tool -- a strong signal to keep the turn out of SMALL."""
    return any(m.role == Role.TOOL or (m.role == Role.ASSISTANT and m.tool_calls) for m in messages)


def _latest_user_text(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.role == Role.USER:
            return message.content
    return None


def _looks_tool_oriented(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _TOOL_PHRASES) or bool(_TOOL_WORD_RE.search(text))


def _looks_complex(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _COMPLEX_PHRASES) or bool(_COMPLEX_WORD_RE.search(text))


def _looks_simple(text: str) -> bool:
    if len(text.split()) > _MAX_SIMPLE_WORDS:
        return False
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _SIMPLE_PATTERNS)


def _classify(messages: list[Message]) -> str:
    """Return a short, log-friendly reason for the tier decision. Only
    "simple_request" qualifies for SMALL -- every other reason (including
    "uncertain", the default) is MEDIUM unless classify_tier() says LARGE."""
    if _has_tool_activity(messages):
        return "tool_activity"

    non_system = [m for m in messages if m.role != Role.SYSTEM]
    if len(non_system) > _MAX_SIMPLE_CONTEXT_MESSAGES:
        return "substantial_context"

    text = _latest_user_text(messages)
    if not text or not text.strip():
        return "no_user_message"

    if _looks_tool_oriented(text):
        return "tool_intent"
    if _looks_complex(text):
        return "complex_request"
    if _looks_simple(text):
        return "simple_request"
    return "uncertain"


# ---------------------------------------------------------------------------
# Tiered routing (GARVIS_LLM_PROVIDER=tiered)
# ---------------------------------------------------------------------------

_MIN_LARGE_WORDS = 60
"""A request this long is treated as LARGE regardless of wording -- long,
detailed asks get a higher effort level."""

# --- Single-model policy ---------------------------------------------------
# The one place it is defined. Every tier is served by exactly ONE model --
# in the shipped configuration Claude Haiku 5.5 (`required_model`, enforced
# in app/agent/build.py) -- and a failed request is never handed to any
# other model or provider. For each request:
#   1. a rate limit / overload / server error (_RETRY_SAME_MODEL) gets up to
#      `max_retries` more attempts on the SAME model, with backoff;
#   2. a timeout is never resent: it may already have been processed and
#      billed (_NEVER_RETRY);
#   3. anything else -- bad request, invalid key, billing problem, refusal,
#      a reply from another model, a local budget stop -- ends the request:
#      the error is raised to the agent, which reports it to the user.
# Tool calls run only after the model returns a complete reply, so a failed
# or timed-out request has executed nothing, and a retry sends the same
# history (including results of tools that already ran) -- nothing runs twice.
_RETRY_SAME_MODEL = (LLMRateLimitError, LLMUnavailableError)
_NEVER_RETRY = (LLMTimeoutError,)
_MAX_RETRY_DELAY_SECONDS = 30.0


def retry_same_model(exc: LLMError) -> bool:
    return isinstance(exc, _RETRY_SAME_MODEL) and not isinstance(exc, _NEVER_RETRY)


def estimate_prompt_tokens(messages: list[Message], tools: list[ToolSpec] | None) -> int:
    """Pessimistic pre-request size estimate (for budget checks only)."""
    chars = 0
    for message in messages:
        chars += len(message.content or "")
        for call in message.tool_calls:
            chars += len(call.name) + len(json.dumps(call.arguments, ensure_ascii=False))
    if tools:
        chars += len(json.dumps([tool.model_dump() for tool in tools], ensure_ascii=False))
    return estimate_tokens(chars)


def classify_tier(messages: list[Message]) -> tuple[ModelTier, str]:
    """Deterministic, local SMALL/MEDIUM/LARGE decision for one chat() call.

    The tier selects the effort level of the (single) model and is recorded
    as the request's cost category. Order:
      1. LARGE  -- the latest user message looks complex (why/explain/
         compare/debug/"step by step"/...) or is very long. Checked first so
         a complex request keeps its tier for every call of a tool loop.
      2. SMALL  -- `_classify()` says "simple_request" (fresh, short,
         allowlisted simple shape, no tools involved).
      3. MEDIUM -- everything else (tool intent/activity, conversation
         context, uncertain) -- the default execution tier.

    The latest *user* message is what's classified, so every call within
    one agent turn (including after tool results) lands on the same tier.
    """
    text = _latest_user_text(messages) or ""
    if text.strip():
        if _looks_complex(text):
            return ModelTier.LARGE, "complex_request"
        if len(text.split()) >= _MIN_LARGE_WORDS:
            return ModelTier.LARGE, "long_request"
    reason = _classify(messages)
    if reason == "simple_request":
        return ModelTier.SMALL, reason
    return ModelTier.MEDIUM, reason


PoolMember = tuple[str, LLMProvider]
"""(log label, e.g. "anthropic:claude-haiku-5-5@low", provider instance)."""


class TieredLLMRouter(LLMProvider):
    """Sends each chat() to its tier's one model -- never to any other.

    Construction refuses a tier with zero or several models, and, with
    `required_model`, any model whose id isn't that one. The same checks run
    again before every request, so even a pool edited at runtime can't send
    a request elsewhere. Failures follow the single-model policy above:
    bounded same-model retries, then the error is raised.

    Cost controls (all optional):
      - `budget`: checked before *every* request, retries included
        (app/llm/usage.py:BudgetGuard).
      - `ledger`: one row per request sent, with the provider's reported
        token counts and the estimated cost (provider.pricing).
    """

    def __init__(
        self,
        pools: dict[ModelTier, list[PoolMember]],
        *,
        required_model: str | None = None,
        max_retries: int = 0,
        retry_backoff_seconds: float = 2.0,
        budget: BudgetGuard | None = None,
        ledger: UsageLedger | None = None,
        output_reserve_tokens: int = 2000,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        missing = [t.value for t in ModelTier if not pools.get(t)]
        if missing:
            raise ValueError(f"TieredLLMRouter needs one model per tier; empty: {missing}")
        self._pools = pools
        self._required_model = required_model
        for tier in ModelTier:
            self._only_member(tier)  # raises on a pool that breaks the single-model policy
        self._max_retries = max(0, max_retries)
        self._retry_backoff = max(0.0, retry_backoff_seconds)
        self._budget = budget
        self._ledger = ledger
        self._output_reserve = output_reserve_tokens
        self._sleep = sleep
        self.last_selection: dict | None = None
        """Diagnostics: tier/reason/model of the most recent successful call."""

    @property
    def pools(self) -> dict[ModelTier, list[PoolMember]]:
        return self._pools

    @property
    def required_model(self) -> str | None:
        return self._required_model

    def _only_member(self, tier: ModelTier) -> PoolMember:
        pool = self._pools.get(tier) or []
        if len(pool) != 1:
            raise LLMError(
                f"Blocked: the {tier.value} tier has {len(pool)} models configured; Garvis sends every request "
                "to exactly one model and never falls back to another."
            )
        label, provider = pool[0]
        model = getattr(provider, "model", None)
        if self._required_model is not None and model != self._required_model:
            raise LLMError(
                f"Blocked: only {self._required_model} may answer, but the {tier.value} tier is set to "
                f"{model or label}. No request was sent."
            )
        return label, provider

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        tier, reason = classify_tier(messages)
        label, provider = self._only_member(tier)
        task = current_task() or TaskMeter()
        if task.category is None:
            task.category = f"{tier.value}:{reason}"
        prompt_estimate = estimate_prompt_tokens(messages, tools)
        attempt = 0
        while True:
            self._check_budget(provider, task, prompt_estimate)
            started = time.monotonic()
            try:
                response = await provider.chat(messages, tools=tools)
            except LLMBilledError as exc:
                status = "model_mismatch" if isinstance(exc, LLMModelMismatchError) else "refusal"
                self._record(provider, tier, task, started, status=status, error=exc, label=label,
                             usage=exc.usage, served_model=exc.served_model, request_id=exc.request_id)
                log_extra(logger, 40 if status == "model_mismatch" else 30, f"llm.router.{status}",
                          tier=tier.value, model=label, served_model=exc.served_model)
                raise
            except LLMError as exc:
                self._record(provider, tier, task, started, status="error", error=exc)
                if retry_same_model(exc) and attempt < self._max_retries:
                    delay = min(self._retry_backoff * (2 ** attempt), _MAX_RETRY_DELAY_SECONDS)
                    attempt += 1
                    log_extra(logger, 30, "llm.router.retry", tier=tier.value, model=label,
                              error_type=type(exc).__name__, error=str(exc)[:300], attempt=attempt,
                              max_retries=self._max_retries, delay_seconds=delay)
                    await self._sleep(delay)
                    continue
                log_extra(
                    logger, 40 if isinstance(exc, LLMAuthenticationError) else 30, "llm.router.failed",
                    tier=tier.value, model=label, error_type=type(exc).__name__, error=str(exc)[:300],
                    attempts=attempt + 1, note="no other model is used",
                )
                raise
            self._record(provider, tier, task, started, status="ok", usage=response.usage, label=label,
                         served_model=response.served_model, request_id=response.request_id,
                         tools_requested=[call.name for call in response.message.tool_calls])
            self.last_selection = {"tier": tier.value, "reason": reason, "model": label, "attempt": attempt + 1}
            log_extra(logger, 20, "llm.router.tier_selection", tier=tier.value, reason=reason, model=label,
                      attempt=attempt + 1)
            return response

    def _check_budget(self, provider: LLMProvider, task: TaskMeter, prompt_estimate: int) -> None:
        if self._budget is None:
            return
        pricing = getattr(provider, "pricing", None)
        projected = pricing.projected_cost(prompt_estimate, self._output_reserve) if pricing else None
        try:
            self._budget.check(task, prompt_estimate, projected)
        except LLMBudgetExceededError as exc:
            log_extra(logger, 30, "llm.budget.stopped", limit=exc.limit, task_id=task.task_id,
                      task_calls=task.calls, task_cost_usd=round(task.cost_usd, 6),
                      prompt_tokens_estimate=prompt_estimate)
            raise

    def _record(
        self,
        provider: LLMProvider,
        tier: ModelTier,
        task: TaskMeter,
        started: float,
        *,
        status: str,
        error: LLMError | None = None,
        usage: TokenUsage | None = None,
        served_model: str | None = None,
        request_id: str | None = None,
        tools_requested: list[str] | None = None,
        label: str | None = None,
    ) -> None:
        pricing = getattr(provider, "pricing", None)
        cost = pricing.cost(usage) if pricing is not None and usage is not None else None
        if usage is None and status == "error":
            task.add_failed()
        else:
            task.add(usage, cost)
        if label and status == "ok" and label not in task.models:
            task.models.append(label)
        if self._ledger is None:
            return
        model = getattr(provider, "model", type(provider).__name__)
        try:
            self._ledger.record(
                status=status, model_requested=model, provider=type(provider).__name__, tier=tier.value,
                task=task, model_served=served_model, request_id=request_id, usage=usage, cost_usd=cost,
                error_type=type(error).__name__ if error else None,
                latency_ms=int((time.monotonic() - started) * 1000), tools_requested=tools_requested,
            )
        except Exception as exc:  # accounting must never break a turn
            log_extra(logger, 40, "llm.usage.record_failed", error=str(exc)[:300])

    async def health_check(self) -> bool:
        # MEDIUM is the default execution tier.
        _, provider = self._only_member(ModelTier.MEDIUM)
        return await provider.health_check()
