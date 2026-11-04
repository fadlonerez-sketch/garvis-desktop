"""Tests for the voice composition root (app/voice/build.py)."""
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.voice.build import build_stt_provider, build_tts_provider, build_voice_pipeline
from app.voice.pipeline import VoicePipeline
from app.voice.stt.faster_whisper_provider import FasterWhisperProvider
from app.voice.tts.piper_provider import PiperProvider


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_build_stt_provider_faster_whisper():
    provider = build_stt_provider(_settings(stt_provider="faster_whisper", stt_model="tiny"))
    assert isinstance(provider, FasterWhisperProvider)


def test_build_stt_provider_wires_language_and_min_duration_from_settings():
    """Regression test: the STT anti-hallucination settings (language pin,
    minimum utterance duration) must actually reach the provider, not just
    exist in config."""
    provider = build_stt_provider(
        _settings(stt_provider="faster_whisper", stt_language="fr", stt_min_duration_seconds=0.5)
    )
    assert provider._language == "fr"
    assert provider._min_duration_seconds == 0.5


def test_build_stt_provider_unknown_raises():
    with pytest.raises(ConfigurationError, match="Unknown STT provider"):
        build_stt_provider(_settings(stt_provider="mystery_stt"))


def test_build_tts_provider_piper():
    provider = build_tts_provider(_settings(tts_provider="piper"))
    assert isinstance(provider, PiperProvider)


def test_build_tts_provider_uses_the_configured_voice_unmodified():
    """Regression test: the installed 'Jarvis-High' voice was previously
    left unused at runtime because .env still pointed at the default
    'en_US-lessac-medium' -- not a code bug, but this guards against the
    code ever silently overriding or ignoring settings.tts_voice again."""
    provider = build_tts_provider(_settings(tts_provider="piper", tts_voice="Jarvis-High"))
    assert provider.voice == "Jarvis-High"


def test_build_tts_provider_default_voice_is_not_silently_swapped():
    provider = build_tts_provider(_settings(tts_provider="piper", tts_voice="en_US-lessac-medium"))
    assert provider.voice == "en_US-lessac-medium"


def test_build_tts_provider_unknown_raises():
    with pytest.raises(ConfigurationError, match="Unknown TTS provider"):
        build_tts_provider(_settings(tts_provider="mystery_tts"))


def test_build_voice_pipeline_assembles_both_sides():
    pipeline = build_voice_pipeline(_settings())
    assert isinstance(pipeline, VoicePipeline)
