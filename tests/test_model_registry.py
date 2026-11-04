"""Tests for the model registry (app/llm/registry.py + config/models.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import ConfigurationError
from app.llm.registry import (
    ZERO_COST_UPSTREAMS,
    ModelTier,
    enforce_single_model,
    load_model_registry,
    zero_cost_violation,
)

_REPO_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "models.json"


def _write(tmp_path, tiers, **top) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"tiers": tiers, **top}), encoding="utf-8")
    return path


def _entry(provider="gemini", model="m", **kw):
    return {"provider": provider, "model": model, **kw}


_EFFORT = {ModelTier.SMALL: "low", ModelTier.MEDIUM: "medium", ModelTier.LARGE: "high"}


def test_shipped_registry_is_claude_haiku_only_one_entry_per_tier():
    """Single-model policy: every tier lists exactly one model, Claude Haiku 5.5
    via the direct Anthropic API -- no other model, not even disabled."""
    registry = load_model_registry(_REPO_REGISTRY)
    enforce_single_model(registry)
    for tier in ModelTier:
        (entry,) = getattr(registry, tier.value)
        assert entry.label == f"anthropic:claude-haiku-5-5@{_EFFORT[tier]}" and entry.enabled, tier
        assert entry.provider == "anthropic" and entry.thinking == "disabled"
        assert entry.prompt_cache == "5m" and entry.verify_served_model and entry.cost == "paid"


def test_shipped_small_tier_is_exactly_the_configured_small_models():
    registry = load_model_registry(_REPO_REGISTRY)
    assert [e.label for e in registry.tier(ModelTier.SMALL)] == ["anthropic:claude-haiku-5-5@low"]


def test_shipped_registry_names_no_other_model_anywhere():
    raw = _REPO_REGISTRY.read_text(encoding="utf-8").lower()
    for other in ("gemini", "groq", "mistral", "cloudflare", "openrouter", "omniroute", "gpt", "sonnet", "opus"):
        assert other not in raw, other


@pytest.mark.parametrize("entries", [
    [_entry("anthropic", "claude-haiku-5-5"), _entry("gemini", "gemini-3.8-flash", enabled=False)],
    [_entry("omniroute", "anthropic/claude-haiku-5-5")],
    [_entry("anthropic", "claude-haiku-4-5")],
    [_entry("anthropic", "claude-haiku-5-5"), _entry("anthropic", "claude-haiku-5-5")],
    [_entry("anthropic", "claude-haiku-5-5", enabled=False)],
])
def test_enforce_single_model_refuses_anything_else(tmp_path, entries):
    ok = [_entry("anthropic", "claude-haiku-5-5")]
    registry = load_model_registry(_write(tmp_path, {"small": ok, "medium": ok, "large": entries}))
    with pytest.raises(ConfigurationError, match="may only use anthropic/claude-haiku-5-5"):
        enforce_single_model(registry)


def test_shipped_haiku_prices_match_the_official_rate_card():
    """platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-22."""
    registry = load_model_registry(_REPO_REGISTRY)
    price = registry.tier(ModelTier.MEDIUM)[0].pricing
    assert (price.input, price.output, price.cache_read, price.cache_write_5m, price.cache_write_1h) == (
        0.10, 0.50, 0.01, 0.125, 0.20)
    assert price.long_context_threshold == 100_000
    assert (price.long_input, price.long_output) == (0.50, 2.50)


def test_shipped_registry_has_no_openai_or_chatgpt_service_in_any_tier():
    """No tier -- enabled or disabled -- routes to an OpenAI/ChatGPT service.
    The only 'openai/...' id is gpt-oss, OpenAI's open-weight model, served
    by Groq in MEDIUM (never SMALL)."""
    registry = load_model_registry(_REPO_REGISTRY)
    entries = [(t, e) for t in ModelTier for e in getattr(registry, t.value)]
    for tier, e in entries:
        assert e.upstream not in ("openai", "chatgpt-web", "codex", "codex-app-server", "openrouter"), e.label
        if "gpt" in e.model.lower() or "openai" in e.model.lower():
            assert tier != ModelTier.SMALL and e.upstream == "groq" and "gpt-oss" in e.model, e.label


def test_small_tier_accepts_allowed_non_gemini_providers_in_order(tmp_path):
    small = [_entry("openrouter", "x/small-a"), _entry("gemini", "small-b"), _entry("openrouter", "y/small-c")]
    registry = load_model_registry(_write(tmp_path, {"small": small, "medium": [_entry()], "large": [_entry()]}))
    assert [e.label for e in registry.tier(ModelTier.SMALL)] == [
        "openrouter:x/small-a", "gemini:small-b", "openrouter:y/small-c",
    ]


@pytest.mark.parametrize("model", ["x/any-model", "llama-3.1-8b-instant"])
def test_small_tier_rejects_groq(tmp_path, model):
    path = _write(tmp_path, {"small": [_entry("groq", model)], "medium": [_entry()], "large": [_entry()]})
    with pytest.raises(ConfigurationError, match="Groq or OpenAI"):
        load_model_registry(path)


@pytest.mark.parametrize("model", ["openai/gpt-5-mini", "OpenAI/o4-mini", "gpt-4o-mini", "chatgpt-4o-latest"])
def test_small_tier_rejects_openai_models(tmp_path, model):
    path = _write(tmp_path, {"small": [_entry("openrouter", model)], "medium": [_entry()], "large": [_entry()]})
    with pytest.raises(ConfigurationError, match="Groq or OpenAI"):
        load_model_registry(path)


def test_small_exclusion_applies_to_small_only(tmp_path):
    """The Groq/OpenAI rule is a SMALL policy; MEDIUM/LARGE validation is unchanged."""
    path = _write(tmp_path, {"small": [_entry()], "medium": [_entry("groq", "g")], "large": [_entry("openrouter", "openai/x")]})
    registry = load_model_registry(path)
    assert [e.label for e in registry.tier(ModelTier.SMALL)] == ["gemini:m"]


def test_unknown_provider_and_tier_are_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="invalid"):
        load_model_registry(_write(tmp_path, {"small": [_entry()], "medium": [_entry("mystery")], "large": [_entry()]}))
    with pytest.raises(ConfigurationError, match="unknown tiers"):
        load_model_registry(_write(tmp_path, {"small": [], "medium": [], "large": [], "huge": []}))


def test_missing_or_malformed_file(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_model_registry(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        load_model_registry(bad)
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="'tiers'"):
        load_model_registry(bad)


def test_labels_include_reasoning_effort(tmp_path):
    path = _write(tmp_path, {"small": [_entry(model="a", reasoning_effort="low")], "medium": [_entry("openrouter", "x/y")], "large": [_entry()]})
    registry = load_model_registry(path)
    assert registry.tier(ModelTier.SMALL)[0].label == "gemini:a@low"
    assert registry.tier(ModelTier.MEDIUM)[0].label == "openrouter:x/y"


# ------------------------------------------------------------ omniroute entries
def _omni(model, **kw):
    return _entry("omniroute", model, **kw)


def test_omniroute_entry_exposes_its_upstream(tmp_path):
    path = _write(tmp_path, {
        "small": [_omni("gemini/gemini-3.1-flash-lite")], "medium": [_omni("groq/openai/gpt-oss-120b")], "large": [_entry()],
    })
    registry = load_model_registry(path)
    [small], [medium] = registry.tier(ModelTier.SMALL), registry.tier(ModelTier.MEDIUM)
    assert (small.upstream, small.upstream_model) == ("gemini", "gemini-3.1-flash-lite")
    assert (medium.upstream, medium.upstream_model) == ("groq", "openai/gpt-oss-120b")
    assert medium.label == "omniroute:groq/openai/gpt-oss-120b"


@pytest.mark.parametrize("model", ["groq/openai/gpt-oss-20b", "GROQ/qwen/qwen3.8-27b", "mistral/openai/gpt-x", "gemini/gpt-5-mini"])
def test_small_tier_rejects_groq_and_openai_behind_omniroute(tmp_path, model):
    path = _write(tmp_path, {"small": [_omni(model)], "medium": [_entry()], "large": [_entry()]})
    with pytest.raises(ConfigurationError, match="Groq or OpenAI"):
        load_model_registry(path)


def test_groq_is_allowed_in_medium_behind_omniroute(tmp_path):
    path = _write(tmp_path, {"small": [_omni("gemini/a")], "medium": [_omni("groq/openai/gpt-oss-120b")], "large": [_entry()]})
    assert load_model_registry(path).tier(ModelTier.MEDIUM)[0].upstream == "groq"


@pytest.mark.parametrize("model", ["auto", "auto/best-free", "auto/coding:free", "combo/x", "gemini/", "/x", "my-combo"])
def test_omniroute_auto_or_implicit_routes_are_rejected_in_every_tier(tmp_path, model):
    for tier in ("small", "medium", "large"):
        tiers = {"small": [_entry()], "medium": [_entry()], "large": [_entry()]}
        tiers[tier] = [_omni(model, enabled=False)]  # rejected even when disabled
        with pytest.raises(ConfigurationError, match="invalid"):
            load_model_registry(_write(tmp_path, tiers))


# ------------------------------------------------------------ zero-cost policy
def _free(model):
    return _omni(model, cost="free")


def _zero_cost(tmp_path, small=None, medium=None, large=None):
    return _write(tmp_path, {
        "small": small or [_free("gemini/s")],
        "medium": medium or [_free("groq/m")],
        "large": large or [_free("mistral/l")],
    }, zero_cost=True)


def test_zero_cost_accepts_approved_free_models(tmp_path):
    registry = load_model_registry(_zero_cost(tmp_path, medium=[_free("cloudflare-ai/@cf/zai-org/glm-4.7-flash")]))
    assert registry.zero_cost and registry.tier(ModelTier.MEDIUM)[0].upstream == "cloudflare-ai"


@pytest.mark.parametrize("cost", ["paid", "unknown", None])
def test_zero_cost_rejects_enabled_models_not_marked_free(tmp_path, cost):
    entry = _omni("gemini/gemini-3.1-pro-preview", **({"cost": cost} if cost else {}))
    with pytest.raises(ConfigurationError, match="zero_cost is on.*not 'free'"):
        load_model_registry(_zero_cost(tmp_path, large=[entry]))


@pytest.mark.parametrize(
    "entry",
    [
        _free("duckduckgo-web/gpt-5.4-mini"),
        _free("opencode/kimi-free"),
        _free("openrouter/deepseek/deepseek-v4-flash:free"),
        _free("cloudflare-playground/glm-5.2"),
        _entry("openrouter", "x/model:free", cost="free"),
    ],
)
def test_zero_cost_rejects_unapproved_upstreams_even_if_labelled_free(tmp_path, entry):
    with pytest.raises(ConfigurationError, match="not an approved zero-cost provider"):
        load_model_registry(_zero_cost(tmp_path, medium=[entry]))


def test_zero_cost_ignores_disabled_paid_entries(tmp_path):
    paid = _omni("gemini/gemini-3.1-pro-preview", cost="paid", enabled=False)
    registry = load_model_registry(_zero_cost(tmp_path, large=[_free("gemini/l"), paid]))
    assert [e.model for e in registry.tier(ModelTier.LARGE)] == ["gemini/l"]


def test_zero_cost_is_off_unless_requested(tmp_path):
    path = _write(tmp_path, {"small": [_entry()], "medium": [_entry("openrouter", "x/paid", cost="paid")], "large": [_entry()]})
    assert load_model_registry(path).zero_cost is False


def test_zero_cost_must_be_a_boolean(tmp_path):
    path = _write(tmp_path, {"small": [_entry()], "medium": [_entry()], "large": [_entry()]}, zero_cost="yes")
    with pytest.raises(ConfigurationError, match="'zero_cost' must be true or false"):
        load_model_registry(path)


def test_invalid_cost_label_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="invalid"):
        load_model_registry(_write(tmp_path, {"small": [_entry(cost="cheap")], "medium": [_entry()], "large": [_entry()]}))
