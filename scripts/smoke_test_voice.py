"""Manual smoke test: exercises the REAL voice pipeline end-to-end --
actual microphone recording, actual faster-whisper transcription, actual
Piper synthesis, and actual audio playback. Not part of pytest (hardware
and model dependent). Run with: python scripts/smoke_test_voice.py
"""
from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.voice.build import build_voice_pipeline


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    pipeline = build_voice_pipeline(settings)

    print("=== Recording 2 seconds from the default microphone ===")
    pipeline.start_listening()
    await asyncio.sleep(2)
    text = await pipeline.stop_listening_and_transcribe()
    print(f"Transcribed text: {text!r}")

    print("\n=== Synthesizing and playing a short reply ===")
    await pipeline.speak("Voice pipeline test successful.")
    await asyncio.sleep(3)
    pipeline.stop_speaking()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
