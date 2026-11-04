"""faster-whisper implementation of SpeechToTextProvider.

Scaffolded for Phase 5. The model name/device/compute type come from
Settings (stt_model / stt_device / stt_compute_type) -- never hardcoded --
so switching model size is a config change, not a code change.

Language is pinned by default (Settings.stt_language, "en") rather than
left to faster-whisper's own per-utterance auto-detection: on short or
noisy clips -- exactly what a VAD-segmented utterance often is -- language
ID has very little signal to work with and can confidently misdetect
(observed in real use: English speech transcribed as garbled German/Dutch
text). A handful of other faster-whisper defaults are also overridden here
(vad_filter, no_speech_threshold, compression_ratio_threshold,
condition_on_previous_text) specifically to suppress hallucinated output on
low-quality input, since this project's own upstream VAD (app/voice/vad.py)
is a simple RMS-threshold detector with no cross-check against what
Whisper's own internal (Silero-based) VAD would consider speech.
"""
from __future__ import annotations

import asyncio
import wave

from app.core.exceptions import SpeechToTextError
from app.voice.stt.base import SpeechToTextProvider, TranscriptionResult


class FasterWhisperProvider(SpeechToTextProvider):
    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
        min_duration_seconds: float = 0.3,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language or None
        self._min_duration_seconds = min_duration_seconds
        self._model = None  # lazily loaded

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechToTextError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc
        return WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)

    def _audio_duration_seconds(self, audio_path: str) -> float | None:
        """Best-effort WAV duration; None if it can't be determined (in
        which case the caller should not skip transcription)."""
        try:
            with wave.open(audio_path, "rb") as wav_file:
                rate = wav_file.getframerate()
                if rate <= 0:
                    return None
                return wav_file.getnframes() / rate
        except (wave.Error, OSError):
            return None

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        def _run() -> TranscriptionResult:
            duration = self._audio_duration_seconds(audio_path)
            if (
                duration is not None
                and self._min_duration_seconds > 0
                and duration < self._min_duration_seconds
            ):
                return TranscriptionResult(text="", language=self._language)

            if self._model is None:
                self._model = self._load_model()
            segments, info = self._model.transcribe(
                audio_path,
                language=self._language,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            text = " ".join(segment.text.strip() for segment in segments)
            return TranscriptionResult(text=text.strip(), language=info.language)

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            raise SpeechToTextError(f"Transcription failed: {exc}") from exc
