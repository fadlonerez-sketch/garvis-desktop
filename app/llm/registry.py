"""Model registry: which models serve which Garvis tier.

Configuration-driven (config/models.json) so model IDs are explicit and
changed without touching code. Tiers are logical Garvis tiers, kept
distinct -- never flattened into one pool:

  SMALL   simple/fast requests. Any provider except Groq, and no
          OpenAI/ChatGPT model (enforced here: such an entry is a
          configuration error, so they can't creep in). For "omniroute"
          entries the rule applies to the *upstream* provider/model named
          in the id, e.g. "groq/..." is rejected.
  MEDIUM  the default execution tier (tool use, normal conversation).
  LARGE   complex reasoning (explain/compare/analyze/debug, long requests).

Each tier is an ORDERED list; TieredLLMRouter tries entries in order and
fails over (logged, and shown to the user) on errors. Entries whose
provider has no credentials configured are skipped at startup with a
warning; a tier left with no usable model is a startup error -- tiers
never borrow each other's models. See app/agent/build.py:_build_tiered_router.

Single-model policy: every tier has exactly one entry, Claude Haiku 5.5 via
the direct Anthropic API (REQUIRED_PROVIDER/REQUIRED_MODEL below, checked by
enforce_single_model() when Garvis builds its router -- any other model,
enabled or not, is a startup error). The loader itself stays generic.
`pricing` (USD per million tokens) feeds the usage ledger and budgets in
app/llm/usage.py.

Zero-cost policy: with `"zero_cost": true` at the top of models.json,
every ENABLED entry must be `"cost": "free"` AND be served by one of
ZERO_COST_UPSTREAMS -- the providers whose $0 tier is documented, needs
no card, and refuses (HTTP 429) instead of billing once the free quota is
used up. The cost label is set by hand after checking the provider's own
pricing page; model names ("...-free") and OmniRoute's catalog are never
treated as proof. OmniRoute auto/combo routes are rejected outright.

Extension point: Code Mode (Claude) is intentionally NOT a tier yet. It
will be a new ModelTier member plus a pool here, and a classification rule
in app/llm/router.py:classify_tier -- nothing else needs to change.
"""
from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.core.exceptions import ConfigurationError
from app.llm.usage import PriceCard

ProviderId = Literal["anthropic", "gemini", "openrouter", "groq", "omniroute"]
CostClass = Literal["free", "paid", "unknown"]

ZERO_COST_UPSTREAMS = frozenset({"gemini", "groq", "cloudflare-ai", "mistral"})
"""Providers approved for zero-cost operation (research 2026-09-29): Google
AI Studio free tier (billing never linked), Groq free plan, Cloudflare
Workers AI on the Workers Free plan, Mistral Free mode. For "omniroute"
entries this is the provider id before the first "/" of the model id."""

_OMNIROUTE_AUTO_PROVIDERS = {"auto", "combo"}


class ModelTier(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ModelEntry(BaseModel):
    provider: ProviderId
    model: str = Field(min_length=1)
    reasoning_effort: str | None = Field(
        default=None,
        description="Passed to providers that support it (Gemini: minimal/low/medium/high).",
    )
    cost: CostClass = Field(
        default="unknown",
        description="'free' only when the upstream's documented free tier covers this exact model.",
    )
    enabled: bool = True
    note: str = ""
    thinking: Literal["disabled"] | None = Field(
        default=None,
        description="'disabled' sends thinking={type:'disabled'} (Claude). OmniRoute 3.8.50 can't "
        "round-trip Claude thinking signatures through its OpenAI API, so tool loops need it off.",
    )
    max_output_tokens: int | None = Field(
        default=None, ge=1, description="Sent as max_tokens (OmniRoute raises it to 32000 when tools are sent)."
    )
    verify_served_model: bool = Field(
        default=False,
        description="Reject a response whose reported model isn't this entry's model (no silent switch).",
    )
    pricing: PriceCard | None = Field(
        default=None, description="USD per million tokens; needed for cost accounting and budgets."
    )
    prompt_cache: Literal["off", "5m", "1h"] = Field(
        default="off",
        description="'anthropic' entries: cache the tools + stable system prompt and the tool-loop tail "
        "(5m or 1h TTL). Other providers ignore it.",
    )

    @model_validator(mode="after")
    def _anthropic_effort_is_valid(self) -> ModelEntry:
        if self.provider != "anthropic" or self.reasoning_effort is None:
            return self
        allowed = ("low", "medium", "high") if self.thinking == "disabled" else ("low", "medium", "high", "xhigh", "max")
        if self.reasoning_effort not in allowed:
            raise ValueError(
                f"{self.model}: reasoning_effort must be one of {allowed}"
                + (" with thinking disabled" if self.thinking == "disabled" else "")
            )
        return self

    @model_validator(mode="after")
    def _no_effort_for_claude_haiku_via_omniroute(self) -> ModelEntry:
        # OmniRoute 3.8.50 turns any reasoning_effort for a Claude Haiku model into
        # thinking={type:'enabled', budget_tokens} (openai-to-claude.ts +
        # claudeHaikuConstraints.ts), which Claude Haiku 5.5 rejects with HTTP 400.
        if (self.provider == "omniroute" and self.upstream == "anthropic"
                and "haiku" in self.upstream_model.lower() and self.reasoning_effort):
            raise ValueError(
                f"{self.model}: remove 'reasoning_effort' -- OmniRoute converts it to a thinking budget, "
                "which Claude Haiku 5.5 rejects (HTTP 400)"
            )
        return self

    @model_validator(mode="after")
    def _omniroute_ids_are_explicit(self) -> ModelEntry:
        if self.provider == "omniroute":
            upstream, sep, rest = self.model.partition("/")
            if not sep or not upstream or not rest:
                raise ValueError(f"omniroute model {self.model!r} must be an explicit '<provider>/<model>' id")
            if upstream.lower() in _OMNIROUTE_AUTO_PROVIDERS:
                raise ValueError(f"omniroute model {self.model!r} is an automatic route; Garvis selects models itself")
        return self

    @property
    def upstream(self) -> str:
        """The provider that actually serves the model."""
        if self.provider == "omniroute":
            return self.model.partition("/")[0].lower()
        return self.provider

    @property
    def upstream_model(self) -> str:
        if self.provider == "omniroute":
            return self.model.partition("/")[2]
        return self.model

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" + (f"@{self.reasoning_effort}" if self.reasoning_effort else "")


_SMALL_EXCLUDED_PROVIDERS = {"groq"}
_OPENAI_MODEL_PREFIXES = ("openai/", "gpt-", "chatgpt")
"""OpenAI/ChatGPT model IDs: the "openai/" vendor prefix OpenRouter (and
Groq) use, or a bare GPT/ChatGPT name."""


def _allowed_in_small(entry: ModelEntry) -> bool:
    return entry.upstream not in _SMALL_EXCLUDED_PROVIDERS and not entry.upstream_model.lower().startswith(
        _OPENAI_MODEL_PREFIXES
    )


def zero_cost_violation(entry: ModelEntry) -> str | None:
    """Why `entry` may not run in zero-cost mode, or None if it may."""
    if entry.cost != "free":
        return f"cost is {entry.cost!r}, not 'free'"
    if entry.upstream not in ZERO_COST_UPSTREAMS:
        return f"upstream {entry.upstream!r} is not an approved zero-cost provider"
    return None


class ModelRegistry(BaseModel):
    small: list[ModelEntry]
    medium: list[ModelEntry]
    large: list[ModelEntry]
    zero_cost: bool = False

    @field_validator("small")
    @classmethod
    def _small_excludes_groq_and_openai(cls, entries: list[ModelEntry]) -> list[ModelEntry]:
        offenders = [e.label for e in entries if not _allowed_in_small(e)]
        if offenders:
            raise ValueError(f"the SMALL tier may not use Groq or OpenAI/ChatGPT models; remove {offenders}")
        return entries

    @model_validator(mode="after")
    def _enforce_zero_cost(self) -> ModelRegistry:
        if not self.zero_cost:
            return self
        offenders = [
            f"{tier.value}: {e.label} ({reason})"
            for tier in ModelTier
            for e in getattr(self, tier.value)
            if e.enabled and (reason := zero_cost_violation(e))
        ]
        if offenders:
            raise ValueError(
                f"zero_cost is on, so every enabled model must be an approved free model; offending: {offenders}"
            )
        return self

    def tier(self, tier: ModelTier) -> list[ModelEntry]:
        return [e for e in getattr(self, tier.value) if e.enabled]


REQUIRED_PROVIDER = "anthropic"
REQUIRED_MODEL = "claude-haiku-5-5"
"""The only model Garvis may call: Claude Haiku 5.5 via the direct Anthropic API."""


def enforce_single_model(registry: ModelRegistry) -> None:
    """Raise ConfigurationError unless every tier has exactly one entry and it
    is REQUIRED_PROVIDER/REQUIRED_MODEL, enabled. Other models are refused
    even when disabled, so none can be switched on as a fallback."""
    problems: list[str] = []
    for tier in ModelTier:
        entries = getattr(registry, tier.value)
        others = [e.label for e in entries if (e.provider, e.model) != (REQUIRED_PROVIDER, REQUIRED_MODEL)]
        if others:
            problems.append(f"{tier.value} lists other models {others}")
        if len(entries) != 1 or not entries[0].enabled:
            problems.append(f"{tier.value} must have exactly one enabled entry (has {len(entries)}, "
                            f"{len(registry.tier(tier))} enabled)")
    if problems:
        raise ConfigurationError(
            f"Garvis may only use {REQUIRED_PROVIDER}/{REQUIRED_MODEL}, one entry per tier with no fallback: "
            + "; ".join(problems)
        )


def load_model_registry(path: Path) -> ModelRegistry:
    if not path.exists():
        raise ConfigurationError(f"Model registry not found at {path} (GARVIS_LLM_MODEL_REGISTRY_PATH).")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigurationError(f"Model registry {path} is not valid JSON: {exc}") from exc
    tiers = raw.get("tiers") if isinstance(raw, dict) else None
    if not isinstance(tiers, dict):
        raise ConfigurationError(f"Model registry {path} must contain a 'tiers' object.")
    unknown = sorted(set(tiers) - {t.value for t in ModelTier})
    if unknown:
        raise ConfigurationError(f"Model registry {path} has unknown tiers {unknown}.")
    zero_cost = raw.get("zero_cost", False)
    if not isinstance(zero_cost, bool):
        raise ConfigurationError(f"Model registry {path}: 'zero_cost' must be true or false.")
    try:
        return ModelRegistry.model_validate({**tiers, "zero_cost": zero_cost})
    except ValidationError as exc:
        raise ConfigurationError(f"Model registry {path} is invalid: {exc}") from exc
