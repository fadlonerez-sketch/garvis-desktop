"""Composition root for the voice pipeline, mirroring app/agent/build.py.

Providers are chosen by name from Settings (never hardcoded); adding a new
STT/TTS backend means adding a branch here, not touching the pipeline or
the UI.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.voice.echo import EchoGuard
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.faster_whisper_provider import FasterWhisperProvider
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.piper_provider import PiperProvider


def build_stt_provider(settings: Settings) -> SpeechToTextProvider:
    if settings.stt_provider == "faster_whisper":
        return FasterWhisperProvider(
            model_size=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            language=settings.stt_language,
            min_duration_seconds=settings.stt_min_duration_seconds,
        )
    raise ConfigurationError(f"Unknown STT provider: '{settings.stt_provider}'")


def build_tts_provider(settings: Settings) -> TextToSpeechProvider:
    if settings.tts_provider == "piper":
        return PiperProvider(
            voice=settings.tts_voice,
            voice_dir=settings.tts_voice_dir,
            piper_executable=settings.tts_executable,
        )
    raise ConfigurationError(f"Unknown TTS provider: '{settings.tts_provider}'")


def build_voice_pipeline(settings: Settings | None = None) -> VoicePipeline:
    settings = settings or get_settings()
    from app.voice.tts.player import AudioPlayer, StreamingAudioPlayer

    player = StreamingAudioPlayer() if settings.tts_playback_backend == "sounddevice" else AudioPlayer()
    echo = (
        EchoGuard(sample_rate=settings.stt_sample_rate, doubt_fraction=settings.session_echo_margin)
        if settings.session_echo_guard_enabled
        else None
    )
    return VoicePipeline(
        stt=build_stt_provider(settings),
        tts=build_tts_provider(settings),
        player=player,
        echo_guard=echo,
    )
