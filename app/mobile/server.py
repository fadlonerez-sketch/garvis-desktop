"""HTTPS transport for the mobile interface.

An aiohttp server running on AgentBridge's existing asyncio loop (no second
loop, no extra thread), so its handlers call MobileService -- and through it
the shared Agent and ConfirmationManager -- directly.

Request pipeline (the `_guard` middleware, applied to every route):
1. Client address must be inside the allowed networks (private LAN ranges
   by default) -> otherwise 403 before anything else runs.
2. Host header must be one of the names in the server certificate.
3. Handler: /api/* needs a paired-device cookie; every POST also needs the
   `X-Garvis-Client` header and (if sent) a same-origin `Origin`, so another
   website can't drive the API from the phone's browser (CSRF).
4. Security headers (strict CSP, no framing, no caching, no referrer) on
   every response.

The API exposes exactly the mobile features -- chat, voice, confirmations,
pairing, tasks (submit an instruction, read status/results, cancel) and
saved protocols (list, request a run). There is no route that runs commands
or code, reads files, edits protocols or changes settings; tool use only
ever happens inside an Agent turn or a desktop-validated protocol run,
behind the existing permission and confirmation checks. Submissions carry
a client `request_id`, so a retried request returns the original task
instead of running it twice.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web

from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.core.logging import get_logger, log_extra
from app.mobile.network import (
    IPNetwork,
    detect_lan_addresses,
    detect_tailscale_addresses,
    is_client_allowed,
    is_tailscale_connection,
    local_hostname,
)
from app.mobile.pairing import DeviceRecord, PairingError, PairingManager
from app.mobile.service import (
    AudioValidationError,
    MobileBusyError,
    MobileService,
    NoReplyError,
)
from app.mobile.tls import TlsMaterial, ensure_tls_material

logger = get_logger("mobile.server")

COOKIE_NAME = "garvis_device"
CLIENT_HEADER = "X-Garvis-Client"
STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_FILES = {
    "app.js": "text/javascript",
    "style.css": "text/css",
    "recorder-worklet.js": "text/javascript",
    "icon-180.png": "image/png",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "icon-maskable-512.png": "image/png",
}
_BINARY_STATIC = {"icon-180.png", "icon-192.png", "icon-512.png", "icon-maskable-512.png"}
_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
_AUDIO_ERROR_STATUS = {"empty": 422, "invalid": 400, "unsupported": 415, "too_long": 413}
_TURN_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_CONFIRMATION_ID = re.compile(r"^[0-9a-fA-F-]{36}$")
_PROTOCOL_ID = re.compile(r"^[a-f0-9]{12,32}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_MAX_JSON_BYTES = 64 * 1024
LONG_POLL_MAX_SECONDS = 25.0

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "media-src 'self' blob:; connect-src 'self'; manifest-src 'self'; worker-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
}


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _host_only(host_header: str | None) -> str:
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _device_label(user_agent: str) -> str:
    for name in ("iPhone", "iPad", "Android", "Windows", "Macintosh"):
        if name in user_agent:
            return name
    return "browser"


def create_app(
    service: MobileService,
    pairing: PairingManager,
    *,
    allowed_networks: list[IPNetwork],
    allowed_hosts: set[str] | None,
    ca_cert_der: bytes | None,
    max_audio_bytes: int,
    companion=None,
    remote_access: bool = False,
    remote_hosts: set[str] | None = None,
) -> web.Application:
    """Build the aiohttp application. `allowed_hosts=None` skips the Host
    check (tests only); `ca_cert_der=None` disables the CA download.
    `companion` (app/mobile/companion.py) adds the native app's /api/v1.

    `remote_access` (GARVIS_MOBILE_REMOTE_ACCESS=tailscale) additionally
    admits a tailnet peer that reached this PC on its own Tailscale address
    -- for /api/v1 only (the native app); it still needs a device token.
    Its Host header may be that address or one of `remote_hosts`."""
    static = {name: (STATIC_DIR / name).read_bytes() for name in _STATIC_FILES if (STATIC_DIR / name).exists()}
    service_worker = (STATIC_DIR / "sw.js").read_bytes()
    manifest = (STATIC_DIR / "manifest.webmanifest").read_bytes()
    index_html = (STATIC_DIR / "index.html").read_bytes()
    hosts = {host.lower() for host in allowed_hosts} if allowed_hosts is not None else None
    extra_remote_hosts = {host.lower() for host in remote_hosts or ()}

    def _is_remote(request: web.Request) -> bool:
        if not remote_access or is_client_allowed(request.remote, allowed_networks):
            return False
        sockname = request.transport.get_extra_info("sockname") if request.transport is not None else None
        local = sockname[0] if isinstance(sockname, tuple) and sockname else None
        return is_tailscale_connection(request.remote, local)

    tailnet = {"at": float("-inf"), "addresses": []}

    async def _remote_urls(request: web.Request) -> list[str]:
        """Where a paired phone can reach this PC away from home: its
        Tailscale IPv4 addresses (re-read at most once a minute) and the
        configured remote host names. Empty unless remote access is on."""
        if not remote_access:
            return []
        if time.monotonic() - tailnet["at"] > 60:
            tailnet["addresses"] = [a for a in await asyncio.to_thread(detect_tailscale_addresses) if "." in a]
            tailnet["at"] = time.monotonic()
        sockname = request.transport.get_extra_info("sockname") if request.transport is not None else None
        port = int(sockname[1]) if isinstance(sockname, tuple) and len(sockname) > 1 else 8443
        return [f"https://{host}:{port}" for host in [*tailnet["addresses"], *sorted(extra_remote_hosts)]]

    def _remote_host_ok(request: web.Request) -> bool:
        sockname = request.transport.get_extra_info("sockname") if request.transport is not None else None
        local = str(sockname[0]).split("%", 1)[0].lower() if isinstance(sockname, tuple) and sockname else ""
        host = _host_only(request.host)
        return host == local or host in extra_remote_hosts or (hosts is not None and host in hosts)

    @web.middleware
    async def _guard(request: web.Request, handler) -> web.StreamResponse:
        remote = _is_remote(request)
        if not remote and not is_client_allowed(request.remote, allowed_networks):
            log_extra(logger, 30, "mobile.client_refused", remote=request.remote)
            response: web.StreamResponse = web.Response(status=403, text="Forbidden")
        elif remote and not (companion is not None and request.path.startswith("/api/v1/")):
            response = web.Response(status=403, text="Forbidden")
        elif remote and not _remote_host_ok(request):
            response = web.Response(status=421, text="Unknown host name")
        elif not remote and hosts is not None and _host_only(request.host) not in hosts:
            response = web.Response(status=421, text="Unknown host name")
        else:
            try:
                response = await handler(request)
            except web.HTTPException as exc:
                response = _error(exc.status, exc.reason)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def _device(request: web.Request) -> DeviceRecord | None:
        # The web app is for phones: a station's token (presence/voice only on
        # the native API) must not get chat, tasks or confirmations here.
        record = pairing.verify(request.cookies.get(COOKIE_NAME))
        return record if record is not None and record.kind == "phone" else None

    def _same_origin_post(request: web.Request) -> bool:
        if request.headers.get(CLIENT_HEADER) != "1":
            return False
        origin = request.headers.get("Origin")
        return origin is None or urlsplit(origin).netloc.lower() == (request.host or "").lower()

    def _authorize(request: web.Request, *, post: bool) -> web.Response | None:
        if post and not _same_origin_post(request):
            return _error(403, "Cross-site request refused.")
        if _device(request) is None:
            return _error(401, "This device is not paired.")
        return None

    async def _read_json(request: web.Request) -> dict[str, Any] | None:
        if request.content_type != "application/json":
            return None
        if (request.content_length or 0) > _MAX_JSON_BYTES:
            raise web.HTTPRequestEntityTooLarge(max_size=_MAX_JSON_BYTES, actual_size=request.content_length)
        try:
            data = await request.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------ pages
    async def index(request: web.Request) -> web.Response:
        return web.Response(body=index_html, content_type="text/html", charset="utf-8")

    async def static_file(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name not in static:
            return web.Response(status=404, text="Not found")
        if name in _BINARY_STATIC:
            return web.Response(body=static[name], content_type=_STATIC_FILES[name])
        return web.Response(body=static[name], content_type=_STATIC_FILES[name], charset="utf-8")

    async def sw(request: web.Request) -> web.Response:
        # Served from the root so it may control the whole app; it only
        # caches the static shell -- never API responses.
        return web.Response(body=service_worker, content_type="text/javascript", charset="utf-8",
                            headers={"Service-Worker-Allowed": "/"})

    async def web_manifest(request: web.Request) -> web.Response:
        return web.Response(body=manifest, content_type="application/manifest+json", charset="utf-8")

    async def ca_certificate(request: web.Request) -> web.Response:
        # Public by design: it's the certificate the phone has to trust
        # before HTTPS works. The CA private key is never served.
        if ca_cert_der is None:
            return web.Response(status=404, text="Not found")
        return web.Response(
            body=ca_cert_der,
            content_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": 'attachment; filename="garvis-local-ca.crt"'},
        )

    # ---------------------------------------------------------- pairing
    async def session(request: web.Request) -> web.Response:
        return web.json_response({"paired": _device(request) is not None})

    async def pair(request: web.Request) -> web.Response:
        if not _same_origin_post(request):
            return _error(403, "Cross-site request refused.")
        data = await _read_json(request)
        if data is None or not isinstance(data.get("code"), str):
            return _error(400, "Send the pairing code shown on the PC.")
        try:
            token = pairing.redeem(data["code"], label=_device_label(request.headers.get("User-Agent", "")))
        except PairingError as exc:
            return _error(403, str(exc))
        response = web.json_response({"paired": True})
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=int(pairing.token_ttl_seconds),
            path="/",
            secure=True,
            httponly=True,
            samesite="Strict",
        )
        return response

    async def unpair(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        pairing.revoke(request.cookies.get(COOKIE_NAME))
        response = web.json_response({"paired": False})
        response.del_cookie(COOKIE_NAME, path="/")
        return response

    # ------------------------------------------------------ conversation
    async def events(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=False)) is not None:
            return denied
        try:
            after = max(0, int(request.query.get("after", "0")))
            wait = min(LONG_POLL_MAX_SECONDS, max(0.0, float(request.query.get("wait", "0"))))
        except ValueError:
            return _error(400, "Invalid 'after' or 'wait'.")
        return web.json_response(await service.wait_for_updates(after, wait))

    async def messages(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        data = await _read_json(request)
        if data is None or not isinstance(data.get("text"), str):
            return _error(400, "Expected JSON with a 'text' field.")
        request_id = _request_id(data)
        if request_id is False:
            return _error(400, "Invalid request id.")
        device = _device(request)
        try:
            turn = service.submit_text(data["text"], request_id=request_id, device=device.label if device else "")
        except ValueError as exc:
            return _error(400, str(exc))
        except MobileBusyError as exc:
            return _error(409, str(exc))
        return web.json_response({"turn_id": turn.id}, status=202)

    async def voice(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        if request.content_type not in _AUDIO_TYPES:
            return _error(415, "Unsupported audio format: send audio/wav.")
        if (request.content_length or 0) > max_audio_bytes:
            return _error(413, "The recording is too large.")
        data = await request.read()
        if len(data) > max_audio_bytes:
            return _error(413, "The recording is too large.")
        try:
            transcript = await service.transcribe_recording(data)
        except AudioValidationError as exc:
            return _error(_AUDIO_ERROR_STATUS.get(exc.kind, 400), str(exc))
        except MobileBusyError as exc:
            return _error(409, str(exc))
        except SpeechToTextError as exc:
            log_extra(logger, 30, "mobile.transcription_failed", error=str(exc))
            return _error(502, str(exc)[:300])
        if not transcript:
            return _error(422, "No speech was recognized. Try again a little closer to the microphone.")
        try:
            device = _device(request)
            turn = service.submit_text(transcript, kind="voice", device=device.label if device else "")
        except ValueError as exc:
            return _error(400, str(exc))
        except MobileBusyError as exc:
            return _error(409, str(exc))
        return web.json_response({"turn_id": turn.id, "transcript": transcript}, status=202)

    async def speech(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=False)) is not None:
            return denied
        turn_id = request.match_info["turn_id"]
        if not _TURN_ID.match(turn_id):
            return _error(404, "Unknown message.")
        try:
            audio = await service.synthesize_reply(turn_id)
        except NoReplyError as exc:
            return _error(404, str(exc))
        except TextToSpeechError as exc:
            log_extra(logger, 30, "mobile.speech_failed", error=str(exc))
            return _error(502, str(exc)[:300])
        return web.Response(body=audio, content_type="audio/wav")

    async def confirm(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        request_id = request.match_info["request_id"]
        data = await _read_json(request)
        if not _CONFIRMATION_ID.match(request_id) or data is None or not isinstance(data.get("approved"), bool):
            return _error(400, "Expected JSON with a boolean 'approved'.")
        if not service.resolve_confirmation(request_id, data["approved"]):
            return _error(404, "That request is no longer waiting for an answer.")
        return web.json_response({"ok": True})

    # ----------------------------------------------------------- tasks
    def _request_id(data: dict) -> str | None | bool:
        value = data.get("request_id")
        if value is None:
            return None
        return value if isinstance(value, str) and _REQUEST_ID.match(value) else False

    async def tasks_list(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=False)) is not None:
            return denied
        return web.json_response(service.list_tasks())

    async def task_detail(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=False)) is not None:
            return denied
        task_id = request.match_info["task_id"]
        detail = service.task_detail(task_id) if _TURN_ID.match(task_id) else None
        if detail is None:
            return _error(404, "Unknown task.")
        return web.json_response(detail)

    async def task_cancel(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        task_id = request.match_info["task_id"]
        if not _TURN_ID.match(task_id):
            return _error(404, "Unknown task.")
        try:
            return web.json_response(service.cancel_task(task_id), status=202)
        except LookupError as exc:
            return _error(404, str(exc))
        except ValueError as exc:
            return _error(409, str(exc))

    async def protocols_list(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=False)) is not None:
            return denied
        return web.json_response({"protocols": service.list_protocols()})

    async def protocol_run(request: web.Request) -> web.Response:
        if (denied := _authorize(request, post=True)) is not None:
            return denied
        protocol_id = request.match_info["protocol_id"]
        data = await _read_json(request)
        if not _PROTOCOL_ID.match(protocol_id) or data is None:
            return _error(400, "Expected JSON for a known protocol.")
        request_id = _request_id(data)
        if request_id is False:
            return _error(400, "Invalid request id.")
        device = _device(request)
        try:
            return web.json_response(service.run_protocol(protocol_id, request_id, device.label if device else ""),
                                     status=202)
        except LookupError as exc:
            return _error(404, str(exc))
        except ValueError as exc:
            return _error(409, str(exc))

    app = web.Application(middlewares=[_guard], client_max_size=max(max_audio_bytes, _MAX_JSON_BYTES) + 1024)
    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static_file)
    app.router.add_get("/garvis-ca.crt", ca_certificate)
    app.router.add_get("/api/session", session)
    app.router.add_post("/api/pair", pair)
    app.router.add_post("/api/unpair", unpair)
    app.router.add_get("/api/events", events)
    app.router.add_post("/api/messages", messages)
    app.router.add_post("/api/voice", voice)
    app.router.add_get("/api/turns/{turn_id}/speech", speech)
    app.router.add_post("/api/confirmations/{request_id}", confirm)
    app.router.add_get("/sw.js", sw)
    app.router.add_get("/manifest.webmanifest", web_manifest)
    app.router.add_get("/api/tasks", tasks_list)
    app.router.add_get("/api/tasks/{task_id}", task_detail)
    app.router.add_post("/api/tasks/{task_id}/cancel", task_cancel)
    app.router.add_get("/api/protocols", protocols_list)
    app.router.add_post("/api/protocols/{protocol_id}/run", protocol_run)
    if companion is not None:
        from app.mobile.native_api import register_native_routes

        register_native_routes(app, service=service, pairing=pairing, companion=companion,
                               max_audio_bytes=max_audio_bytes, is_remote=_is_remote, remote_urls=_remote_urls)
    return app


class MobileServer:
    """Owns the HTTPS listener's lifecycle. start()/stop() must run on the
    loop that hosts the Agent (AgentBridge's background loop)."""

    def __init__(
        self,
        service: MobileService,
        pairing: PairingManager,
        *,
        host: str,
        port: int,
        allowed_networks: list[IPNetwork],
        data_dir: Path,
        extra_hosts: list[str] | None = None,
        name_constraints: bool = True,
        max_audio_bytes: int = 8 * 1024 * 1024,
        companion=None,
        remote_access: str = "off",
        remote_hosts: list[str] | None = None,
    ) -> None:
        self._service = service
        self._pairing = pairing
        self._host = host
        self._port = port
        self._allowed_networks = allowed_networks
        self._data_dir = data_dir
        self._extra_hosts = extra_hosts or []
        self._name_constraints = name_constraints
        self._max_audio_bytes = max_audio_bytes
        self._companion = companion
        self._remote_access = remote_access == "tailscale"
        self._remote_hosts = remote_hosts or []
        self._remote_urls: list[str] = []
        self._runner: web.AppRunner | None = None
        self._tls: TlsMaterial | None = None
        self._urls: list[str] = []

    @property
    def pairing(self) -> PairingManager:
        return self._pairing

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    @property
    def remote_urls(self) -> list[str]:
        """Tailscale addresses the native app can use away from home (empty
        unless remote access is on and Tailscale is running)."""
        return list(self._remote_urls)

    @property
    def companion(self):
        return self._companion

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def ca_cert_der(self) -> bytes:
        """The public CA certificate (DER) -- e.g. to copy to an Android phone
        over USB, since Android installs CA certificates only from Settings."""
        return self._tls.ca_cert_der() if self._tls is not None else b""

    @property
    def ca_fingerprint(self) -> str:
        return self._tls.ca_fingerprint_sha256 if self._tls is not None else ""

    @property
    def bound_port(self) -> int | None:
        if self._runner is None:
            return None
        for address in self._runner.addresses:
            if isinstance(address, tuple) and len(address) >= 2:
                return int(address[1])
        return None

    def _reachable_hosts(self) -> list[str]:
        if self._host not in ("0.0.0.0", "::", ""):
            return [self._host]
        return detect_lan_addresses()

    async def start(self) -> None:
        if self._runner is not None:
            return
        lan_hosts = await asyncio.to_thread(self._reachable_hosts)
        cert_hosts = [*lan_hosts, local_hostname(), "localhost", "127.0.0.1", *self._extra_hosts]
        self._tls = await asyncio.to_thread(
            ensure_tls_material, self._data_dir, cert_hosts, name_constraints=self._name_constraints
        )
        app = create_app(
            self._service,
            self._pairing,
            allowed_networks=self._allowed_networks,
            allowed_hosts=set(self._tls.hosts),
            ca_cert_der=self._tls.ca_cert_der(),
            max_audio_bytes=self._max_audio_bytes,
            companion=self._companion,
            remote_access=self._remote_access,
            remote_hosts=set(self._remote_hosts),
        )
        runner = web.AppRunner(app, access_log=None, keepalive_timeout=30, shutdown_timeout=3.0)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self._host, self._port, ssl_context=self._tls.server_ssl_context())
            await site.start()
        except Exception:
            await runner.cleanup()
            raise
        self._runner = runner
        self._service.start()
        port = self.bound_port or self._port
        self._urls = [f"https://{host}:{port}/" for host in [*lan_hosts, local_hostname()]]
        if self._remote_access:
            tailnet = await asyncio.to_thread(detect_tailscale_addresses)
            self._remote_urls = [f"https://{host if ':' not in host else '[' + host + ']'}:{port}/"
                                 for host in [*tailnet, *self._remote_hosts]]
        log_extra(logger, 20, "mobile.server_started", host=self._host, port=port, urls=self._urls)

    async def stop(self) -> None:
        runner = self._runner
        self._runner = None
        await self._service.stop()  # wakes pending long-polls so shutdown isn't held up
        if runner is not None:
            await runner.cleanup()
            log_extra(logger, 20, "mobile.server_stopped")
