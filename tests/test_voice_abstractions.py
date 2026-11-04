import pytest

from app.voice.stt.base import SpeechToTextProvider, TranscriptionResult
from app.voice.tts.base import TextToSpeechProvider


def test_stt_provider_is_abstract():
    with pytest.raises(TypeError):
        SpeechToTextProvider()


def test_tts_provider_is_abstract():
    with pytest.raises(TypeError):
        TextToSpeechProvider()


def test_transcription_result_schema():
    result = TranscriptionResult(text="hello world", language="en")
    assert result.text == "hello world"
    assert result.language == "en"


def test_transcription_result_language_optional():
    result = TranscriptionResult(text="hello")
    assert result.language is None
