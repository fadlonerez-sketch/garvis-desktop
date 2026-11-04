"""Tests for FetchWebpageTool's SSRF guard and redirect handling.

DNS/network are faked (via monkeypatched `socket.getaddrinfo` and a
patched `httpx.AsyncClient` transport) so these tests are hermetic and
don't depend on real internet access or real internal hosts.
"""
from __future__ import annotations

import socket

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
            return httpx.Response(
                302, headers={"location": "http://public-2.example.com/final"}
            )
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
            return httpx.Response(
                302, headers={"location": "http://internal.example.com/secret"}
            )
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
