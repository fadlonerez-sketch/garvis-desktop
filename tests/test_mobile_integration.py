"""The mobile server for real: TLS on 127.0.0.1 verified against the
generated CA, and its lifecycle inside AgentBridge (shared Agent, shutdown)."""
from __future__ import annotations

import asyncio
import json
import socket
import ssl
import time
import urllib.error
import urllib.request

import aiohttp
import pytest

from app.core.config import Settings
from app.core.events import AgentEvent, EventType
from app.mobile.network import parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.server import MobileServer
from app.mobile.service import MOBILE_SOURCE, MobileService
from app.ui.agent_bridge import AgentBridge
from tests.mobile_fakes import GatedLLM, make_agent, make_voice


def _server(tmp_path, service) -> MobileServer:
    return MobileServer(
        service,
        PairingManager(tmp_path / "devices.json"),
        host="127.0.0.1",
        port=0,
        allowed_networks=parse_networks(""),
        data_dir=tmp_path / "mobile",
    )


async def test_https_server_is_trusted_via_the_local_ca_only(tmp_path):
    from app.core.events import EventBus

    events = EventBus()
    service = MobileService(make_agent(tmp_path, GatedLLM(gated=False), events=events), events, make_voice())
    server = _server(tmp_path, service)
    await server.start()
    try:
        port = server.bound_port
        url = f"https://127.0.0.1:{port}/"
        trusting = ssl.create_default_context(cafile=str(tmp_path / "mobile" / "ca.crt"))
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=trusting) as response:
                assert response.status == 200
                assert "Content-Security-Policy" in response.headers
            # Without the CA (like Safari before the profile is trusted) the
            # certificate is rejected -- nothing falls back to plain HTTP.
            with pytest.raises(aiohttp.ClientConnectorCertificateError):
                await session.get(url, ssl=ssl.create_default_context())
        assert len(server.ca_fingerprint) == 95
        assert server.urls  # shown in the desktop pairing dialog
    finally:
        await server.stop()
    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()


# ----------------------------------------------------------------- bridge
def _bridge_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        memory_db_path=tmp_path / "test.db",
        llm_provider="tiered",  # providers constructed lazily, no network call at build time
        omniroute_api_key="fake-test-key",
        **overrides,
    )


def _wait(predicate, timeout=10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.02)


def test_mobile_interface_is_off_by_default(tmp_path):
    assert Settings(_env_file=None).mobile_enabled is False
    bridge = AgentBridge(settings=_bridge_settings(tmp_path))
    try:
        assert bridge.mobile is None
    finally:
        bridge.shutdown()


def test_bridge_serves_the_mobile_interface_from_its_own_agent_and_shuts_it_down(tmp_path):
    bridge = AgentBridge(
        settings=_bridge_settings(
            tmp_path, mobile_enabled=True, mobile_host="127.0.0.1", mobile_port=0, mobile_data_dir=tmp_path / "mobile"
        )
    )
    try:
        _wait(lambda: bridge.mobile.is_running)
        service = bridge.mobile._service
        # One Agent, one EventBus, one VoicePipeline -- never a second stack.
        assert service._agent is bridge._agent
        assert service._events is bridge._events
        assert service._voice is bridge._voice

        port = bridge.mobile.bound_port
        context = ssl.create_default_context(cafile=str(tmp_path / "mobile" / "ca.crt"))
        with urllib.request.urlopen(f"https://127.0.0.1:{port}/api/session", context=context, timeout=5) as response:
            assert json.loads(response.read()) == {"paired": False}

        # Desktop typed turns land in the phone's view of the conversation.
        asyncio.run_coroutine_threadsafe(
            bridge._events.publish(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "from the PC"})),
            bridge._loop,
        ).result(5)
        assert service.snapshot()["events"][-1]["text"] == "from the PC"
    finally:
        bridge.shutdown()
    assert bridge.mobile.is_running is False
    with pytest.raises((urllib.error.URLError, OSError)):
        urllib.request.urlopen(f"https://127.0.0.1:{port}/", context=context, timeout=2)


def test_bridge_reports_a_mobile_start_failure_without_breaking_the_desktop(tmp_path, caplog):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    bridge = AgentBridge(
        settings=_bridge_settings(
            tmp_path, mobile_enabled=True, mobile_host="127.0.0.1", mobile_port=port, mobile_data_dir=tmp_path / "m"
        )
    )
    try:
        # (bridge.error is a queued Qt signal from the loop thread; this test
        # has no Qt event loop, so it checks the logged failure instead.)
        _wait(lambda: any(r.getMessage() == "agent_bridge.mobile_start_failed" for r in caplog.records))
        assert bridge.mobile.is_running is False
        # The desktop side is untouched: the same Agent still takes turns.
        assert bridge._agent.is_busy is False
        assert bridge._thread.is_alive()
    finally:
        bridge.shutdown()
        blocker.close()


def test_mobile_replies_are_not_spoken_on_the_pc_speakers(tmp_path):
    bridge = AgentBridge(settings=_bridge_settings(tmp_path))
    spoken: list[str] = []
    real_loop = bridge._loop

    class _RecordingLoop:
        def create_task(self, value):
            spoken.append(value)

    bridge._safe_speak = lambda text: text
    bridge._loop = _RecordingLoop()
    try:
        bridge.set_voice_replies_enabled(True)
        bridge._on_event(AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "for the phone", "source": MOBILE_SOURCE}))
        bridge._on_event(AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": "for the desk"}))
        assert spoken == ["for the desk"]
    finally:
        bridge._loop = real_loop
        bridge.shutdown()


def test_bridge_relays_remote_messages_and_confirmation_resolution(tmp_path):
    bridge = AgentBridge(settings=_bridge_settings(tmp_path))
    remote: list[tuple[str, str]] = []
    resolved: list[tuple[str, str]] = []
    bridge.remote_user_message.connect(lambda text, source: remote.append((text, source)))
    bridge.confirmation_resolved.connect(lambda rid, outcome: resolved.append((rid, outcome)))
    try:
        bridge._on_event(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "typed here"}))
        bridge._on_event(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "from phone", "source": MOBILE_SOURCE}))
        bridge._on_event(AgentEvent(type=EventType.CONFIRMATION_RESOLVED, payload={"id": "r1", "outcome": "approved"}))
        assert remote == [("from phone", MOBILE_SOURCE)]
        assert resolved == [("r1", "approved")]
    finally:
        bridge.shutdown()
