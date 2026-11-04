"""Text-to-speech provider abstraction (implemented in Phase 6)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TextToSpeechProvider(ABC):
    @abstractmethod
    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Render `text` to speech, writing a playable audio file, and return its path."""

    @abstractmethod
    def stop(self) -> None:
        """Stop any in-progress playback."""
