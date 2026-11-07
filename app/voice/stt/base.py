"""Speech-to-text provider abstraction (implemented in Phase 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None


class SpeechToTextProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe a recorded audio file to text."""
