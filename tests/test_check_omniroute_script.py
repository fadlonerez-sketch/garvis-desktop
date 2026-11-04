"""Tests for scripts/check_omniroute.py -- httpx MockTransport, no network, no prompts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx

from app.core.config import Settings

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # the script imports its sibling env_file
_spec = importlib.util.spec_from_file_location("check_omniroute", _SCRIPTS / "check_omniroute.py")
check_omniroute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_omniroute)

_SECRET = "omr-script-test-secret"
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def recording(request):
        seen.append(request)
        return handler(request)

    class _Patched(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(recording)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(check_omniroute.httpx, "AsyncClient", _Patched)
    return seen


def _gateway(auth_enforced=True, allow_origin=None, extra_ids=()):
    def handler(request: httpx.Request) -> httpx.Response:
        cors = {"access-control-allow-origin": allow_origin} if allow_origin else {}
        authed = request.headers.get("authorization") == f"Bearer {_SECRET}"
        if request.method == "OPTIONS":
            return httpx.Response(204, headers=cors)
        if request.url.path.endswith("/chat/completions"):
            # Only the credential-less cross-origin probe may reach chat by default.
            assert not authed and request.headers.get("origin") == "https://example.com"
            assert request.content and b"garvis-probe-nonexistent" in request.content
            if auth_enforced:
                return httpx.Response(401, json={"error": {"message": "Authentication required"}}, headers=cors)
            return httpx.Response(404, json={"error": {"message": "unknown model"}}, headers=cors)
        if request.url.path.endswith("/models"):
            if auth_enforced and not authed:
                return httpx.Response(401, json={"error": {"message": "unauthorized"}})
            return httpx.Response(200, json={"data": [{"id": "gemini/gemini-3.1-flash-lite"}, *({"id": i} for i in extra_ids)]})
        raise AssertionError(f"unexpected request {request.method} {request.url}")
    return handler


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, omniroute_api_key=_SECRET, **kw)


async def test_secure_gateway_passes_without_sending_prompts(monkeypatch, capsys):
    """Echoed preflight origin (OmniRoute's /v1 design) passes while a key is required."""
    seen = _patch(monkeypatch, _gateway(allow_origin="https://example.com"))
    assert await check_omniroute.verify(_settings())
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "(enforced)" in out and "Keyless providers: none listed." in out
    assert "can't call without a key" in out
    assert _SECRET not in out
    chats = [r for r in seen if r.url.path.endswith("/chat/completions") and r.method == "POST"]
    assert len(chats) == 1 and "authorization" not in chats[0].headers  # only the unauthenticated probe


async def test_open_api_wildcard_cors_and_keyless_providers_are_flagged(monkeypatch, capsys):
    _patch(monkeypatch, _gateway(auth_enforced=False, allow_origin="*", extra_ids=["ddgw/gpt-5.4-mini"]))
    assert not await check_omniroute.verify(_settings())
    out = capsys.readouterr().out
    assert "API is open without a key" in out and "CORS: WARNING" in out and "ddgw/gpt-5.4-mini" in out
    assert _SECRET not in out


async def test_missing_key_and_remote_url_are_refused_before_any_request(monkeypatch, capsys):
    seen = _patch(monkeypatch, _gateway())
    assert not await check_omniroute.verify(Settings(_env_file=None), )
    assert not await check_omniroute.verify(_settings(omniroute_base_url="http://192.168.1.2:20128/v1"), )
    assert seen == []
    out = capsys.readouterr().out
    assert "NOT CONFIGURED" in out and "REFUSED" in out


async def test_the_check_never_sends_a_prompt_to_any_model(monkeypatch, capsys):
    """Single-model policy: the only chat request is the keyless CORS probe for a
    model that doesn't exist -- refused before any model could run."""
    seen = _patch(monkeypatch, _gateway())
    assert await check_omniroute.verify(_settings())
    chats = [r for r in seen if r.method == "POST"]
    assert len(chats) == 1 and b"garvis-probe-nonexistent" in chats[0].content
    assert "authorization" not in chats[0].headers
    assert not hasattr(check_omniroute, "tool_round_trip")
    assert "no prompt was sent" in capsys.readouterr().out
