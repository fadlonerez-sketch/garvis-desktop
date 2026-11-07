"""Manual smoke test for the iPhone interface, without an iPhone.

Runs the REAL mobile HTTPS server (local CA + TLS) with the REAL configured
STT/TTS providers (faster-whisper, Piper), and plays the phone's part over
HTTPS on 127.0.0.1: pair, send a text message, upload a spoken WAV (made by
Piper, so STT hears real speech), wait for the reply and fetch its speech.

The LLM is a local echo stub, so no API credits are used and no prompt
leaves the PC; the PC microphone and speakers are never touched. Uses a
throwaway certificate directory, not data/mobile. Not part of pytest
(model dependent). Run with: python scripts/smoke_test_mobile.py
"""

from __future__ import annotations

import asyncio
import ssl
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp  # noqa: E402

from app.agent.agent import Agent  # noqa: E402
from app.agent.memory import SQLiteMemoryStore  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.events import EventBus  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolSpec  # noqa: E402
from app.mobile.network import parse_networks  # noqa: E402
from app.mobile.pairing import PairingManager  # noqa: E402
from app.mobile.server import COOKIE_NAME, MobileServer  # noqa: E402
from app.mobile.service import MobileService  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402
from app.voice.build import build_voice_pipeline  # noqa: E402

SPOKEN_QUESTION = "What is the weather like in London today?"


class _EchoLLM(LLMProvider):
    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        return LLMResponse(message=Message(role=Role.ASSISTANT, content=f"You said: {messages[-1].content}"))

    async def health_check(self) -> bool:
        return True


async def _wait_for_turn(session, base, headers, turn_id, after=0, timeout=60.0) -> list[dict]:
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with session.get(f"{base}/api/events?after={after}&wait=10", headers=headers) as response:
            data = await response.json()
        seen += data["events"]
        after = data["last_seq"]
        if any(e["kind"] == "turn_finished" and e["turn_id"] == turn_id for e in seen):
            return seen
    raise TimeoutError("the turn did not finish")


async def main() -> None:
    settings = get_settings()
    voice = build_voice_pipeline(settings)
    events = EventBus()
    # ignore_cleanup_errors: the memory store keeps its SQLite file open on Windows.
    with tempfile.TemporaryDirectory(prefix="garvis_mobile_smoke_", ignore_cleanup_errors=True) as workdir:
        work = Path(workdir)
        agent = Agent(llm=_EchoLLM(), registry=ToolRegistry(), memory=SQLiteMemoryStore(work / "m.db"), events=events)
        service = MobileService(agent, events, voice)
        pairing = PairingManager(work / "devices.json")
        server = MobileServer(
            service, pairing, host="127.0.0.1", port=0, allowed_networks=parse_networks(""), data_dir=work / "tls"
        )
        await server.start()
        base = f"https://127.0.0.1:{server.bound_port}"
        trust = ssl.create_default_context(cafile=str(work / "tls" / "ca.crt"))
        print(f"Server up at {base} (CA SHA-256 {server.ca_fingerprint[:23]}...)")
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=trust)) as session:
                async with session.get(f"{base}/api/events") as response:
                    print(f"Unpaired API call -> HTTP {response.status} (expect 401)")

                code = pairing.new_pairing_code()
                async with session.post(
                    f"{base}/api/pair", json={"code": code.code}, headers={"X-Garvis-Client": "1"}
                ) as response:
                    token = response.cookies[COOKIE_NAME].value
                    print(f"Pairing -> HTTP {response.status}")
                headers = {"X-Garvis-Client": "1", "Cookie": f"{COOKIE_NAME}={token}"}

                async with session.post(
                    f"{base}/api/messages", json={"text": "hello from the smoke test"}, headers=headers
                ) as response:
                    turn_id = (await response.json())["turn_id"]
                replies = [
                    e["text"] for e in await _wait_for_turn(session, base, headers, turn_id) if e["kind"] == "assistant"
                ]
                print(f"Text turn reply: {replies[-1]!r}")

                print(f"\nSynthesizing the question with Piper: {SPOKEN_QUESTION!r}")
                question_wav = await voice.synthesize_to_bytes(SPOKEN_QUESTION)
                started = time.monotonic()
                async with session.post(
                    f"{base}/api/voice", data=question_wav, headers={**headers, "Content-Type": "audio/wav"}
                ) as response:
                    body = await response.json()
                    print(f"Voice upload -> HTTP {response.status} in {time.monotonic() - started:.1f}s")
                if "error" in body:
                    raise SystemExit(f"Voice upload failed: {body['error']}")
                print(f"Transcript (faster-whisper): {body['transcript']!r}")
                await _wait_for_turn(session, base, headers, body["turn_id"])

                started = time.monotonic()
                async with session.get(f"{base}/api/turns/{body['turn_id']}/speech", headers=headers) as response:
                    audio = await response.read()
                    print(
                        f"Reply speech (Piper) -> HTTP {response.status}, {response.headers['Content-Type']}, "
                        f"{len(audio)} bytes in {time.monotonic() - started:.1f}s"
                    )
                assert audio[:4] == b"RIFF", "speech is not a WAV file"
        finally:
            await server.stop()
    print("\nMobile smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
