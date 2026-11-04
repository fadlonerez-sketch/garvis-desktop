"""Tests for provider selection in app/agent/build.py:build_llm_provider():
only 'tiered' (Claude Haiku 5.5) is accepted -- the former groq / openrouter /
gemini modes and the local 'ollama' / 'auto' modes are all rejected.
Mirrors the Settings(_env_file=None, **overrides) pattern used in
tests/test_voice_build.py so these are hermetic and never touch the real
.env or make network calls.
"""
from __future__ import annotations

import pytest

from app.agent.build import build_llm_provider
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.llm.router import TieredLLMRouter


def _settings(**overrides) -> Settings:
    overrides.setdefault("groq_api_key", "fake-test-key")
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize("mode", ["groq", "openrouter", "gemini", "omniroute"])
def test_other_model_providers_cannot_be_selected(mode):
    """Single-model policy: no setting leads to a model other than Claude Haiku 5.5."""
    with pytest.raises(ConfigurationError, match="only Claude Haiku 5.5"):
        build_llm_provider(_settings(llm_provider=mode, openrouter_api_key="fake-openrouter-key",
                                     openrouter_model="anthropic/claude-haiku-4.5", gemini_api_key="g",
                                     omniroute_api_key="k"))


def test_default_provider_is_tiered():
    """GARVIS_LLM_PROVIDER missing -> 'tiered' (Claude Haiku 5.5 via the Anthropic API)."""
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "tiered"
    router = build_llm_provider(_settings())
    assert isinstance(router, TieredLLMRouter) and router.required_model == "claude-haiku-5-5"


@pytest.mark.parametrize("provider", ["ollama", "auto"])
def test_removed_local_model_modes_are_rejected(provider):
    """Garvis never runs an AI model locally: the former Ollama-backed modes
    fail loudly instead of silently starting a local model."""
    with pytest.raises(ConfigurationError, match="never runs an AI model locally"):
        build_llm_provider(_settings(llm_provider=provider))


def test_no_local_model_runtime_ships_with_garvis():
    """The Ollama provider module is gone and no settings for it remain."""
    import importlib.util

    assert importlib.util.find_spec("app.llm.ollama") is None
    assert not any(name.startswith("ollama") for name in Settings.model_fields)


def test_unknown_provider_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="Unknown llm_provider"):
        build_llm_provider(_settings(llm_provider="mystery_llm"))
