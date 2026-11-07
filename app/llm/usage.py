"""LLM usage accounting and local spending controls.

Three pieces, all local and all optional for callers that don't need them:

  - `TokenUsage` / `PriceCard`: the token counts a provider actually
    reported for one request, and the per-model prices (config/models.json
    "pricing") used to turn them into an *estimated* cost. Missing usage
    fields stay None -- nothing is guessed from character counts.
  - `UsageLedger`: one SQLite row per model request (data/usage.db): time,
    task id, category, requested/served model, request id, token counts,
    estimated cost. Never prompts, replies, screen text or keys.
  - `TaskMeter` + `BudgetGuard`: per-task call/cost counters (a "task" is one
    user turn; Agent opens one per turn via `task_meter()`) and the limits
    checked *before* every request: calls and cost per task, daily and
    monthly spend.

These are application-level limits on Garvis's own estimates. They stop
Garvis from sending more requests; they are not a billing cap. The only hard
cap is the provider's own (e.g. the Anthropic Console's monthly spend limit
or a prepaid balance with auto-reload off).
"""

from __future__ import annotations

import contextvars
import sqlite3
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.exceptions import LLMBudgetExceededError

CHARS_PER_TOKEN_ESTIMATE = 3.0
"""Pre-request size estimate only (budget checks happen before the provider
has counted anything). Deliberately pessimistic for English/JSON on Claude's
tokenizer so the estimate errs high. Never used for accounting."""


class TokenUsage(BaseModel):
    """Token counts as reported by the provider for one request. `input_tokens`
    is the uncached input only (Anthropic's convention); cache reads/writes
    are separate. None = the provider didn't report it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None
    cache_write_unknown_ttl_tokens: int | None = None
    """Cache writes reported without a 5m/1h split (OmniRoute's OpenAI-format
    responses). Priced at the 1-hour write rate, so the estimate errs high."""
    reasoning_tokens: int | None = None
    """Part of output_tokens (already billed there); informational."""

    @property
    def prompt_tokens(self) -> int | None:
        """Everything the prompt cost: uncached + cache reads + cache writes."""
        if self.input_tokens is None:
            return None
        return self.input_tokens + (self.cache_read_tokens or 0) + self.cache_write_tokens

    @property
    def cache_write_tokens(self) -> int:
        return (
            (self.cache_write_5m_tokens or 0)
            + (self.cache_write_1h_tokens or 0)
            + (self.cache_write_unknown_ttl_tokens or 0)
        )


class PriceCard(BaseModel):
    """USD per million tokens. Long-context prices apply to the whole request
    once its prompt (input + cache reads + cache writes) exceeds the threshold
    (Claude Haiku 5.5: 100,000 tokens)."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write_5m: float = Field(ge=0)
    cache_write_1h: float = Field(ge=0)
    long_context_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None
    long_cache_read: float | None = None
    long_cache_write_5m: float | None = None
    long_cache_write_1h: float | None = None
    source: str = ""

    def _rates(self, prompt_tokens: int) -> tuple[float, float, float, float, float]:
        if self.long_context_threshold is not None and prompt_tokens > self.long_context_threshold:
            return (
                self.long_input if self.long_input is not None else self.input,
                self.long_output if self.long_output is not None else self.output,
                self.long_cache_read if self.long_cache_read is not None else self.cache_read,
                self.long_cache_write_5m if self.long_cache_write_5m is not None else self.cache_write_5m,
                self.long_cache_write_1h if self.long_cache_write_1h is not None else self.cache_write_1h,
            )
        return self.input, self.output, self.cache_read, self.cache_write_5m, self.cache_write_1h

    def cost(self, usage: TokenUsage) -> float | None:
        """Estimated USD for one request, or None when input/output counts are missing."""
        if usage.input_tokens is None or usage.output_tokens is None:
            return None
        rate_in, rate_out, rate_read, rate_w5, rate_w1 = self._rates(usage.prompt_tokens or 0)
        return (
            usage.input_tokens * rate_in
            + usage.output_tokens * rate_out
            + (usage.cache_read_tokens or 0) * rate_read
            + (usage.cache_write_5m_tokens or 0) * rate_w5
            + ((usage.cache_write_1h_tokens or 0) + (usage.cache_write_unknown_ttl_tokens or 0)) * rate_w1
        ) / 1_000_000

    def projected_cost(self, prompt_tokens: int, output_tokens: int) -> float:
        """Pessimistic cost of a request before it is sent: the whole prompt
        billed at the most expensive input class (a 1-hour cache write) plus
        `output_tokens` of reply."""
        rate_in, rate_out, _, rate_w5, rate_w1 = self._rates(prompt_tokens)
        return (prompt_tokens * max(rate_in, rate_w5, rate_w1) + output_tokens * rate_out) / 1_000_000


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN_ESTIMATE) + 1


# ---------------------------------------------------------------------------
# Per-task metering
# ---------------------------------------------------------------------------


@dataclass
class TaskMeter:
    """Counters for one task (one user turn), shared by every request in it."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str | None = None
    calls: int = 0
    cost_usd: float = 0.0
    unknown_cost_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    models: list[str] = field(default_factory=list)
    """Labels of the model entries that answered (always Claude Haiku 5.5)."""

    def add(self, usage: TokenUsage | None, cost: float | None) -> None:
        self.calls += 1
        if cost is None:
            self.unknown_cost_calls += 1
        else:
            self.cost_usd += cost
        if usage is not None:
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.cache_read_tokens += usage.cache_read_tokens or 0
            self.cache_write_tokens += usage.cache_write_tokens

    def add_failed(self) -> None:
        """A request that was sent but failed (not billed by the provider)."""
        self.calls += 1

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
            "unknown_cost_calls": self.unknown_cost_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "models": list(self.models),
        }


_current_task: contextvars.ContextVar[TaskMeter | None] = contextvars.ContextVar("garvis_task_meter", default=None)


@contextmanager
def task_meter():
    """Open a TaskMeter for the requests made inside this block (one user turn)."""
    meter = TaskMeter()
    token = _current_task.set(meter)
    try:
        yield meter
    finally:
        _current_task.reset(token)


def current_task() -> TaskMeter | None:
    return _current_task.get()


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    task_id TEXT,
    category TEXT,
    tier TEXT,
    provider TEXT,
    model_requested TEXT,
    model_served TEXT,
    request_id TEXT,
    status TEXT NOT NULL,
    error_type TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_5m_tokens INTEGER,
    cache_write_1h_tokens INTEGER,
    cache_write_unknown_ttl_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    tools_requested TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_ts ON llm_calls(ts);
CREATE INDEX IF NOT EXISTS llm_calls_task ON llm_calls(task_id);
"""


class UsageLedger:
    """Local SQLite ledger of model requests. Thread-safe; synchronous writes
    are tiny (one row), so callers don't need to offload them."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(
        self,
        *,
        status: str,
        model_requested: str,
        provider: str | None = None,
        tier: str | None = None,
        task: TaskMeter | None = None,
        model_served: str | None = None,
        request_id: str | None = None,
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
        error_type: str | None = None,
        latency_ms: int | None = None,
        tools_requested: list[str] | None = None,
        now: datetime | None = None,
    ) -> None:
        u = usage or TokenUsage()
        row = (
            (now or datetime.now()).isoformat(timespec="seconds"),
            task.task_id if task else None,
            task.category if task else None,
            tier,
            provider,
            model_requested,
            model_served,
            request_id,
            status,
            error_type,
            u.input_tokens,
            u.output_tokens,
            u.cache_read_tokens,
            u.cache_write_5m_tokens,
            u.cache_write_1h_tokens,
            u.cache_write_unknown_ttl_tokens,
            u.reasoning_tokens,
            cost_usd,
            latency_ms,
            ",".join(tools_requested) if tools_requested else None,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO llm_calls (ts, task_id, category, tier, provider, model_requested, model_served, "
                "request_id, status, error_type, input_tokens, output_tokens, cache_read_tokens, "
                "cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens, reasoning_tokens, "
                "cost_usd, latency_ms, tools_requested) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )

    def spent_since(self, since: datetime) -> float:
        """Estimated USD spent on requests since `since` (local time)."""
        with self._lock, self._connect() as conn:
            (total,) = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE ts >= ?",
                (since.isoformat(timespec="seconds"),),
            ).fetchone()
        return float(total)

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Read-only helper for scripts/usage_report.py."""
        with self._lock, self._connect() as conn:
            return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetLimits(BaseModel):
    max_calls_per_task: int = Field(default=15, ge=1)
    max_cost_per_task_usd: float = Field(default=0.05, ge=0)
    daily_budget_usd: float = Field(default=0.50, ge=0)
    monthly_budget_usd: float = Field(default=5.00, ge=0)
    max_prompt_tokens: int = Field(
        default=60_000,
        ge=1000,
        description="Refuse a request whose estimated prompt is larger (keeps Haiku 5.5 below its "
        "100K-token long-context price tier with a wide margin for estimate error).",
    )


def _month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


class BudgetGuard:
    """Checks the local limits before a request is sent. A zero limit means
    "no requests"; limits are only as accurate as the ledger's estimates, and
    the projection for the next request reserves a fixed amount of reply
    (Settings.llm_output_reserve_tokens) -- one unusually long reply can
    overshoot a limit by the difference, after which nothing more is sent."""

    def __init__(
        self, limits: BudgetLimits, ledger: UsageLedger | None, clock: Callable[[], datetime] = datetime.now
    ) -> None:
        self._limits = limits
        self._ledger = ledger
        self._clock = clock

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    def check(self, task: TaskMeter | None, prompt_tokens_estimate: int, projected_cost: float | None) -> None:
        """Raise LLMBudgetExceededError if sending this request could break a
        limit. `projected_cost` None = the model has no price card: only the
        call-count and prompt-size limits can be enforced."""
        lim = self._limits
        if prompt_tokens_estimate > lim.max_prompt_tokens:
            raise LLMBudgetExceededError(
                f"Stopped before sending: this request would be about {prompt_tokens_estimate:,} tokens, over the "
                f"{lim.max_prompt_tokens:,}-token limit (GARVIS_LLM_MAX_PROMPT_TOKENS).",
                limit="max_prompt_tokens",
            )
        if task is not None and task.calls >= lim.max_calls_per_task:
            raise LLMBudgetExceededError(
                f"Stopped: this task reached its limit of {lim.max_calls_per_task} model calls "
                "(GARVIS_LLM_MAX_CALLS_PER_TASK).",
                limit="max_calls_per_task",
            )
        if projected_cost is None:
            return
        if task is not None and task.cost_usd + projected_cost > lim.max_cost_per_task_usd:
            raise LLMBudgetExceededError(
                f"Stopped: this task has used about ${task.cost_usd:.4f}; another request could pass its "
                f"${lim.max_cost_per_task_usd:.2f} limit (GARVIS_LLM_MAX_COST_PER_TASK_USD).",
                limit="max_cost_per_task_usd",
            )
        if self._ledger is None:
            return
        now = self._clock()
        day = self._ledger.spent_since(_day_start(now))
        if day + projected_cost > lim.daily_budget_usd:
            raise LLMBudgetExceededError(
                f"Stopped: today's estimated model spend is ${day:.4f} of the ${lim.daily_budget_usd:.2f} daily "
                "budget (GARVIS_LLM_DAILY_BUDGET_USD). It resets at midnight.",
                limit="daily_budget_usd",
            )
        month = self._ledger.spent_since(_month_start(now))
        if month + projected_cost > lim.monthly_budget_usd:
            next_month = (_month_start(now) + timedelta(days=32)).replace(day=1)
            raise LLMBudgetExceededError(
                f"Stopped: this month's estimated model spend is ${month:.4f} of the ${lim.monthly_budget_usd:.2f} "
                f"monthly budget (GARVIS_LLM_MONTHLY_BUDGET_USD). It resets on {next_month:%Y-%m-%d}.",
                limit="monthly_budget_usd",
            )
