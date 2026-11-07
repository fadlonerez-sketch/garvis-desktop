"""Tests for FetchWebpageTool's SSRF guard and redirect handling.

DNS/network are faked (via monkeypatched `socket.getaddrinfo` and a
patched `httpx.AsyncClient` transport) so these tests are hermetic and
don't depend on real internet access or real internal hosts.
"""

from __future__ import annotations

import socket

import httpcore
import httpx
import pytest

import app.tools.webpage as webpage_module
from app.tools.webpage import FetchWebpageTool


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    original_client = webpage_module.httpx.AsyncClient

    class _PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(webpage_module.httpx, "AsyncClient", _PatchedClient)


def _patch_getaddrinfo(monkeypatch: pytest.MonkeyPatch, ip_by_host: dict[str, str]) -> None:
    def fake_getaddrinfo(host, *args, **kwargs):
        ip = ip_by_host.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(webpage_module.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.asyncio
async def test_blocks_loopback_ip_literal():
    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://127.0.0.1/secret"})
    result = await tool.execute(args)
    assert not result.success
    assert "non-public" in result.error.lower()


@pytest.mark.asyncio
async def test_blocks_private_ip_literal():
    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://192.168.1.5/admin"})
    result = await tool.execute(args)
    assert not result.success
    assert "non-public" in result.error.lower()


@pytest.mark.asyncio
async def test_blocks_link_local_metadata_endpoint():
    """169.254.169.254 is the cloud-provider instance-metadata address --
    a classic real-world SSRF target."""
    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://169.254.169.254/latest/meta-data/"})
    result = await tool.execute(args)
    assert not result.success
    assert "non-public" in result.error.lower()


@pytest.mark.asyncio
async def test_blocks_unsupported_scheme():
    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "file:///etc/passwd"})
    result = await tool.execute(args)
    assert not result.success
    assert "only http" in result.error.lower()


@pytest.mark.asyncio
async def test_allows_public_host_and_extracts_text(monkeypatch):
    _patch_getaddrinfo(monkeypatch, {"public.example.com": "93.184.216.34"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body><p>Hello world</p></body></html>")

    _patch_transport(monkeypatch, handler)

    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://public.example.com/page"})
    result = await tool.execute(args)

    assert result.success
    assert result.data["text"] == "Hello world"
    assert result.data["url"] == "http://public.example.com/page"


@pytest.mark.asyncio
async def test_follows_safe_redirect_to_public_host(monkeypatch):
    _patch_getaddrinfo(
        monkeypatch,
        {"public.example.com": "93.184.216.34", "public-2.example.com": "93.184.216.35"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example.com":
            return httpx.Response(302, headers={"location": "http://public-2.example.com/final"})
        return httpx.Response(200, html="<html><body>Final page</body></html>")

    _patch_transport(monkeypatch, handler)

    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://public.example.com/start"})
    result = await tool.execute(args)

    assert result.success
    assert result.data["text"] == "Final page"
    assert result.data["url"] == "http://public-2.example.com/final"


@pytest.mark.asyncio
async def test_blocks_redirect_to_private_target(monkeypatch):
    """The initial URL is public, but it redirects to a private-network
    hostname -- the redirect target must be validated too, not just the
    URL the user originally supplied."""
    _patch_getaddrinfo(
        monkeypatch,
        {"public.example.com": "93.184.216.34", "internal.example.com": "10.0.0.5"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example.com":
            return httpx.Response(302, headers={"location": "http://internal.example.com/secret"})
        raise AssertionError("must not follow the redirect to the internal host")

    _patch_transport(monkeypatch, handler)

    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://public.example.com/start"})
    result = await tool.execute(args)

    assert not result.success
    assert "non-public" in result.error.lower()


@pytest.mark.asyncio
async def test_too_many_redirects_fails_gracefully(monkeypatch):
    _patch_getaddrinfo(monkeypatch, {})  # everything resolves "public" by default

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": str(request.url) + "x"})

    _patch_transport(monkeypatch, handler)

    tool = FetchWebpageTool()
    args = tool.validate_arguments({"url": "http://public.example.com/"})
    result = await tool.execute(args)

    assert not result.success
    assert "too many redirects" in result.error.lower()


# --- DNS-rebinding: the connection must go to the validated IP ------------


class _RecordingStream(httpcore.AsyncNetworkStream):
    def __init__(self, log: list) -> None:
        self._log = log

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._log.append(("start_tls", server_hostname))
        raise httpcore.ConnectError("stop after TLS hostname check")

    async def read(self, max_bytes, timeout=None):  # pragma: no cover - never reached
        return b""

    async def write(self, buffer, timeout=None):  # pragma: no cover - never reached
        return None

    async def aclose(self):
        return None


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.log: list = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.log.append(("connect", host, port))
        return _RecordingStream(self.log)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):  # pragma: no cover
        raise AssertionError("no unix sockets")

    async def sleep(self, seconds):  # pragma: no cover
        return None


def _rebinding_getaddrinfo(monkeypatch, first_ip: str, later_ip: str) -> list:
    """First lookup answers `first_ip` (passes validation); any later lookup
    -- i.e. a second resolution by the HTTP client -- answers `later_ip`."""
    calls: list = []

    def fake_getaddrinfo(host, *args, **kwargs):
        calls.append(host)
        ip = first_ip if len(calls) == 1 else later_ip
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(webpage_module.socket, "getaddrinfo", fake_getaddrinfo)
    return calls


@pytest.mark.asyncio
async def test_connection_is_pinned_to_validated_ip_with_hostname_tls(monkeypatch):
    calls = _rebinding_getaddrinfo(monkeypatch, "93.184.216.34", "127.0.0.1")
    backend = _RecordingBackend()
    real_backend_cls = webpage_module._PinnedBackend

    def pinned_with_recorder(host, addresses, inner=None):
        return real_backend_cls(host, addresses, inner=backend)

    monkeypatch.setattr(webpage_module, "_PinnedBackend", pinned_with_recorder)

    tool = FetchWebpageTool()
    result = await tool.execute(tool.validate_arguments({"url": "https://rebind.example.com/x"}))

    assert not result.success  # the recording stream aborts the handshake
    assert calls == ["rebind.example.com"]  # resolved exactly once, by the validator
    assert backend.log[0] == ("connect", "93.184.216.34", 443)
    # TLS still verifies (and sends SNI for) the real hostname, not the IP.
    assert backend.log[1] == ("start_tls", "rebind.example.com")


@pytest.mark.asyncio
async def test_pinned_backend_refuses_unvalidated_host():
    backend = _RecordingBackend()
    pinned = webpage_module._PinnedBackend("good.example.com", ["93.184.216.34"], inner=backend)
    with pytest.raises(httpcore.ConnectError):
        await pinned.connect_tcp("evil.example.com", 80)
    assert backend.log == []


@pytest.mark.asyncio
async def test_pinned_backend_tries_each_validated_address(monkeypatch):
    attempts: list = []

    class _FlakyBackend(_RecordingBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            attempts.append(host)
            if host == "2001:db8::1":
                raise httpcore.ConnectError("unreachable")
            return _RecordingStream(self.log)

    pinned = webpage_module._PinnedBackend("dual.example.com", ["2001:db8::1", "93.184.216.34"], inner=_FlakyBackend())
    await pinned.connect_tcp("dual.example.com", 443)
    assert attempts == ["2001:db8::1", "93.184.216.34"]


def test_pinned_transport_ignores_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    transport = webpage_module._pinned_transport("example.com", ["93.184.216.34"])
    assert isinstance(transport._pool, httpcore.AsyncConnectionPool)
    assert isinstance(transport._pool._network_backend, webpage_module._PinnedBackend)


def test_validator_rejects_any_non_public_answer(monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.101.102.103", 0)),
        ]

    monkeypatch.setattr(webpage_module.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(webpage_module.UnsafeURLError):
        webpage_module._resolve_public_addresses("http://mixed.example.com/")


@pytest.mark.asyncio
async def test_each_redirect_hop_is_revalidated_and_repinned(monkeypatch):
    _patch_getaddrinfo(monkeypatch, {"public.example.com": "93.184.216.34", "next.example.com": "8.8.8.8"})
    pinned: list = []
    real = webpage_module._pinned_transport

    def recording_pinned_transport(host, addresses):
        pinned.append((host, tuple(addresses)))
        return real(host, addresses)

    monkeypatch.setattr(webpage_module, "_pinned_transport", recording_pinned_transport)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example.com":
            return httpx.Response(302, headers={"location": "http://next.example.com/"})
        return httpx.Response(200, text="<p>done</p>")

    _patch_transport(monkeypatch, handler)
    tool = FetchWebpageTool()
    result = await tool.execute(tool.validate_arguments({"url": "http://public.example.com/"}))
    assert result.success
    assert pinned == [("public.example.com", ("93.184.216.34",)), ("next.example.com", ("8.8.8.8",))]
