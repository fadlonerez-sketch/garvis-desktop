"""MobileService: the transport-agnostic logic behind the iPhone interface,
driven through a real Agent (scripted LLM) and a real VoicePipeline whose
STT/TTS providers are fakes -- no API credits, no audio hardware."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.llm.base import Role
from app.mobile.service import (
    MOBILE_SOURCE,
    AudioValidationError,
    MobileBusyError,
    MobileService,
    NoReplyError,
    validate_wav,
)
from tests.conftest import final_response, tool_call_response
from tests.mobile_fakes import FakeSTT, FakeTTS, GatedLLM, HighRiskTool, make_agent, make_voice, make_wav, wait_until


def _service(tmp_path, llm=None, *, stt=None, tts=None, tools=None, session_active=lambda: False, **kwargs):
    events = EventBus()
    agent = make_agent(tmp_path, llm or GatedLLM(gated=False), events=events, tools=tools)
    service = MobileService(agent, events, make_voice(stt, tts), is_desktop_session_active=session_active, **kwargs)
    service.start()
    return service, agent, events


def _kinds(service, kind):
    return [e for e in service.snapshot()["events"] if e["kind"] == kind]


# ------------------------------------------------------------------ text turns
async def test_text_turn_runs_through_the_shared_agent(tmp_path):
    service, agent, _ = _service(tmp_path)
    turn = service.submit_text("  hello garvis  ")
    await wait_until(lambda: turn.done)

    assert turn.reply == "reply to: hello garvis"
    users = _kinds(service, "user")
    assert users[-1]["text"] == "hello garvis" and users[-1]["source"] == MOBILE_SOURCE
    assistant = _kinds(service, "assistant")[-1]
    assert assistant["turn_id"] == turn.id and assistant["source"] == MOBILE_SOURCE
    finished = _kinds(service, "turn_finished")[-1]
    assert finished == {**finished, "turn_id": turn.id, "turn_kind": "text", "has_reply": True}
    # The same Agent context the desktop uses -- not a private copy.
    assert [m.content for m in agent._context.messages if m.role == Role.USER] == ["hello garvis"]


@pytest.mark.parametrize("text", ["", "   ", "x" * 101])
async def test_invalid_text_is_rejected(tmp_path, text):
    service, _, _ = _service(tmp_path, max_message_chars=100)
    with pytest.raises(ValueError):
        service.submit_text(text)


async def test_desktop_turns_appear_in_the_shared_conversation(tmp_path):
    service, agent, _ = _service(tmp_path)
    await agent.handle_user_message("typed on the PC")
    users = _kinds(service, "user")
    assert users[-1] == {**users[-1], "text": "typed on the PC", "source": "desktop"}
    assert _kinds(service, "assistant")[-1]["turn_id"] is None


async def test_second_mobile_turn_is_refused_while_one_runs(tmp_path):
    llm = GatedLLM()
    service, _, _ = _service(tmp_path, llm)
    first = service.submit_text("first")
    with pytest.raises(MobileBusyError):
        service.submit_text("second")
    llm.release()
    await wait_until(lambda: first.done)
    service.submit_text("third")  # fine once the first finished


async def test_mobile_turn_refused_while_a_desktop_turn_holds_the_agent(tmp_path):
    llm = GatedLLM()
    service, agent, _ = _service(tmp_path, llm)
    desktop = asyncio.create_task(agent.handle_user_message("desktop"))
    await wait_until(lambda: llm.in_flight == 1)
    with pytest.raises(MobileBusyError, match="busy"):
        service.submit_text("phone")
    assert service.snapshot()["busy"] is True
    llm.release()
    await desktop


async def test_mobile_turn_refused_during_a_desktop_voice_session(tmp_path):
    service, _, _ = _service(tmp_path, session_active=lambda: True)
    with pytest.raises(MobileBusyError, match="voice session"):
        service.submit_text("hello")


async def test_desktop_turn_submitted_during_a_mobile_turn_waits_its_turn(tmp_path):
    """Concurrency/context safety: a desktop turn arriving mid mobile turn is
    queued by the Agent's lock, never interleaved into its messages."""
    llm = GatedLLM()
    service, agent, _ = _service(tmp_path, llm)
    mobile = service.submit_text("phone question")
    await wait_until(lambda: llm.in_flight == 1)
    desktop = asyncio.create_task(agent.handle_user_message("desk question"))
    await asyncio.sleep(0.05)
    assert llm.in_flight == 1
    llm.release()
    await wait_until(lambda: mobile.done)
    await desktop
    contents = [m.content for m in agent._context.messages if m.role in (Role.USER, Role.ASSISTANT)]
    assert contents == ["phone question", "reply to: phone question", "desk question", "reply to: desk question"]
    assert llm.max_in_flight == 1


async def test_failed_turn_has_no_reply_and_still_finishes(tmp_path):
    from app.core.exceptions import LLMUnavailableError

    class _DownLLM(GatedLLM):
        async def chat(self, messages, tools=None):
            raise LLMUnavailableError("The model service is unreachable.")

    service, _, _ = _service(tmp_path, _DownLLM(gated=False))
    turn = service.submit_text("hi")
    await wait_until(lambda: turn.done)
    assert turn.reply is None
    assert _kinds(service, "error")[-1]["message"] == "The model service is unreachable."
    assert _kinds(service, "turn_finished")[-1]["has_reply"] is False
    with pytest.raises(NoReplyError):
        await service.synthesize_reply(turn.id)


# ---------------------------------------------------------------- long-poll
async def test_long_poll_wakes_on_new_events_and_times_out_quietly(tmp_path):
    service, _, _ = _service(tmp_path)
    start = service.snapshot()["last_seq"]
    quiet = await service.wait_for_updates(start, 0.05)
    assert quiet["events"] == []

    waiter = asyncio.create_task(service.wait_for_updates(start, 5))
    await asyncio.sleep(0.02)
    service.submit_text("ping")
    result = await asyncio.wait_for(waiter, 2)
    assert result["busy"] is True or result["events"]


async def test_snapshot_flags_a_cursor_from_a_previous_server_run(tmp_path):
    service, _, _ = _service(tmp_path)
    assert service.snapshot(after=999)["reset"] is True
    assert service.snapshot(after=0)["reset"] is False


async def test_stop_releases_long_polls_and_unsubscribes(tmp_path):
    service, agent, _ = _service(tmp_path)
    waiter = asyncio.create_task(service.wait_for_updates(service.snapshot()["last_seq"], 10))
    await asyncio.sleep(0.02)
    await service.stop()
    await asyncio.wait_for(waiter, 1)
    before = service.snapshot()["last_seq"]
    await agent.handle_user_message("after stop")
    assert service.snapshot()["last_seq"] == before


async def test_tool_arguments_are_truncated_and_secret_keys_masked(tmp_path):
    service, _, events = _service(tmp_path)
    await events.publish(
        AgentEvent(type=EventType.TOOL_STARTED, payload={"name": "t", "arguments": {"text": "a" * 900, "password": "hunter2"}})
    )
    arguments = _kinds(service, "tool_started")[-1]["arguments"]
    assert len(arguments["text"]) < 600 and arguments["text"].endswith("...")
    assert "hunter2" not in arguments["password"]


# -------------------------------------------------------------------- voice
def test_validate_wav_accepts_pcm16_mono_and_stereo():
    assert validate_wav(make_wav(1.0), max_seconds=60).duration_seconds == pytest.approx(1.0)
    assert validate_wav(make_wav(1.0, 48000, channels=2), max_seconds=60).channels == 2


@pytest.mark.parametrize(
    "data, kind",
    [
        (b"", "empty"),
        (b"OggS" + b"\x00" * 100, "unsupported"),
        (b"RIFF\x00\x00\x00\x00WAVEjunk", "invalid"),
        (make_wav(1.0, sample_width=1), "unsupported"),
        (make_wav(1.0, sample_rate=96000), "unsupported"),
        (make_wav(0.1), "empty"),
        (make_wav(3.0), "too_long"),
        (make_wav(1.0)[:-2000], "invalid"),
    ],
    ids=["empty", "ogg", "broken-header", "8-bit", "96khz", "too-short", "too-long", "truncated"],
)
def test_validate_wav_rejects_bad_recordings(data, kind):
    with pytest.raises(AudioValidationError) as info:
        validate_wav(data, max_seconds=2.0)
    assert info.value.kind == kind


async def test_voice_recording_is_transcribed_from_a_temp_file_that_is_then_deleted(tmp_path):
    stt = FakeSTT(text=" open my calendar ")
    service, _, _ = _service(tmp_path, stt=stt)
    text = await service.transcribe_recording(make_wav(1.0))
    assert text == "open my calendar"
    assert stt.existed_during_call == [True]
    path = Path(stt.paths[0])
    assert not path.exists()
    assert path.parent == Path(tempfile.gettempdir())


async def test_invalid_recording_never_reaches_stt(tmp_path):
    stt = FakeSTT()
    service, _, _ = _service(tmp_path, stt=stt)
    with pytest.raises(AudioValidationError):
        await service.transcribe_recording(b"not audio")
    assert stt.paths == []


async def test_stt_failure_is_raised_and_temp_file_removed(tmp_path):
    stt = FakeSTT(error="Transcription failed: model missing")
    service, _, _ = _service(tmp_path, stt=stt)
    with pytest.raises(SpeechToTextError, match="model missing"):
        await service.transcribe_recording(make_wav(1.0))
    assert not Path(stt.paths[0]).exists()


async def test_stt_timeout_cleans_up_once_the_worker_finishes(tmp_path):
    stt = FakeSTT(delay=0.4)
    service, _, _ = _service(tmp_path, stt=stt, stt_timeout_seconds=0.05)
    with pytest.raises(SpeechToTextError, match="too long"):
        await service.transcribe_recording(make_wav(1.0))
    await asyncio.sleep(0.6)
    assert not Path(stt.paths[0]).exists()


async def test_voice_is_refused_before_transcribing_while_busy(tmp_path):
    llm = GatedLLM()
    stt = FakeSTT()
    service, _, _ = _service(tmp_path, llm, stt=stt)
    service.submit_text("busy")
    with pytest.raises(MobileBusyError):
        await service.transcribe_recording(make_wav(1.0))
    assert stt.paths == []
    llm.release()


async def test_reply_speech_comes_from_the_tts_provider_as_bytes(tmp_path):
    tts = FakeTTS()
    llm = GatedLLM([final_response("Here is **the** answer: https://example.com/x")], gated=False)
    service, _, _ = _service(tmp_path, llm, tts=tts)
    turn = service.submit_text("question", kind="voice")
    await wait_until(lambda: turn.done)
    audio = await service.synthesize_reply(turn.id)
    assert audio[:4] == b"RIFF"
    assert "**" not in tts.texts[0] and "https://" not in tts.texts[0]  # spoken copy, like the desktop
    assert not Path(tts.paths[0]).exists()


async def test_long_replies_are_truncated_for_speech_only(tmp_path):
    tts = FakeTTS()
    long_reply = "word " * 400
    llm = GatedLLM([final_response(long_reply)], gated=False)
    service, _, _ = _service(tmp_path, llm, tts=tts, max_tts_chars=100)
    turn = service.submit_text("question")
    await wait_until(lambda: turn.done)
    await service.synthesize_reply(turn.id)
    assert len(tts.texts[0]) < 150 and tts.texts[0].endswith("The rest is on screen.")
    assert _kinds(service, "assistant")[-1]["text"] == long_reply


async def test_tts_failure_is_raised_and_temp_file_removed(tmp_path):
    tts = FakeTTS(error="Voice model not found")
    service, _, _ = _service(tmp_path, tts=tts)
    turn = service.submit_text("question")
    await wait_until(lambda: turn.done)
    with pytest.raises(TextToSpeechError, match="Voice model"):
        await service.synthesize_reply(turn.id)
    assert not Path(tts.paths[0]).exists()


async def test_unknown_turn_has_no_speech(tmp_path):
    service, _, _ = _service(tmp_path)
    with pytest.raises(NoReplyError):
        await service.synthesize_reply("nope")


# ------------------------------------------------------------ confirmations
async def _pending_confirmation(tmp_path):
    tool = HighRiskTool()
    llm = GatedLLM([tool_call_response("high_risk_echo", {}), final_response("finished")], gated=False)
    service, agent, _ = _service(tmp_path, llm, tools=[tool])
    turn = service.submit_text("do the risky thing")
    await wait_until(lambda: service.snapshot()["pending_confirmations"])
    request = service.snapshot()["pending_confirmations"][0]
    return service, agent, tool, turn, request


async def test_mobile_approval_resolves_the_existing_confirmation(tmp_path):
    service, _, tool, turn, request = await _pending_confirmation(tmp_path)
    assert request["tool_name"] == "high_risk_echo"
    assert tool.executions == 0  # nothing runs before the answer

    assert service.resolve_confirmation(request["id"], True) is True
    await wait_until(lambda: turn.done)
    assert tool.executions == 1
    assert service.snapshot()["pending_confirmations"] == []
    assert _kinds(service, "confirmation_resolved")[-1]["outcome"] == "approved"


async def test_mobile_rejection_prevents_the_tool_from_running(tmp_path):
    service, _, tool, turn, request = await _pending_confirmation(tmp_path)
    assert service.resolve_confirmation(request["id"], False) is True
    await wait_until(lambda: turn.done)
    assert tool.executions == 0
    assert _kinds(service, "confirmation_resolved")[-1]["outcome"] == "denied"


async def test_unknown_or_already_answered_confirmation_is_refused(tmp_path):
    service, agent, tool, turn, request = await _pending_confirmation(tmp_path)
    assert service.resolve_confirmation("00000000-0000-0000-0000-000000000000", True) is False
    agent.confirmations.resolve(request["id"], False)  # answered on the desktop first
    await wait_until(lambda: turn.done)
    assert service.resolve_confirmation(request["id"], True) is False
    assert tool.executions == 0
