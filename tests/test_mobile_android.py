"""Android (Chrome, Samsung Internet) support for the phone interface,
alongside the existing iPhone support -- every check here that has an iPhone
counterpart asserts both.

Simulated only: in-process aiohttp server, offscreen Qt, and certificates
validated with `cryptography`'s strict RFC 5280 path validator (name
constraints, EKU, validity -- the checks a modern browser verifier makes).
None of this runs on a real Android device.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from aiohttp.test_utils import TestClient, TestServer
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store, VerificationError
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from app.core.events import EventBus
from app.mobile.network import parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.server import COOKIE_NAME, STATIC_DIR, create_app
from app.mobile.service import MobileService
from app.mobile.tls import ensure_tls_material
from app.tasks.manager import TaskManager
from app.tasks.protocols import Protocol, ProtocolStore, Step
from tests.mobile_fakes import GatedLLM, make_agent, make_voice, wait_until
from tests.test_tasks_and_protocols import FakeWebsiteTool

GALAXY_S24_CHROME = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Mobile Safari/537.36"
)
"""Chrome's reduced user agent, as an Android phone sends it (model and real Android version are hidden)."""
GALAXY_SAMSUNG_INTERNET = (
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) "
    "SamsungBrowser/27.0 Chrome/125.0.0.0 Mobile Safari/537.36"
)
IPHONE_SAFARI = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/18.5 Mobile/15E148 Safari/604.1"
)
POST = {"X-Garvis-Client": "1"}


# ------------------------------------------------------------ certificates
def _load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _verify(material, host: str) -> None:
    try:
        subject = x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        subject = x509.DNSName(host)
    verifier = (
        PolicyBuilder()
        .store(Store([_load(material.ca_cert_path)]))
        .time(dt.datetime.now(dt.UTC))
        .build_server_verifier(subject)
    )
    verifier.verify(_load(material.server_cert_path), [])


def test_server_certificate_passes_strict_validation_for_every_lan_name(tmp_path):
    material = ensure_tls_material(tmp_path, ["192.168.10.156", "desktop-example.local", "localhost", "127.0.0.1"])
    for host in ("192.168.10.156", "desktop-example.local", "127.0.0.1", "localhost"):
        _verify(material, host)  # raises if a browser-grade verifier would reject it
    with pytest.raises(VerificationError):
        _verify(material, "192.168.10.157")  # and it is really checking


def test_ca_has_what_android_requires_of_a_user_ca(tmp_path):
    ca = _load(ensure_tls_material(tmp_path, ["192.168.1.20"]).ca_cert_path)
    basic = ca.extensions.get_extension_for_class(x509.BasicConstraints)
    assert basic.critical and basic.value.ca
    assert ca.extensions.get_extension_for_class(x509.KeyUsage).value.key_cert_sign
    constraints = ca.extensions.get_extension_for_class(x509.NameConstraints)
    assert constraints.critical  # the CA can't vouch for public websites -- on iPhone or Android


def test_a_name_outside_the_constraints_is_left_out_instead_of_breaking_https(tmp_path):
    # e.g. a VPN address in 100.64.0.0/10 added via GARVIS_MOBILE_EXTRA_HOSTS
    material = ensure_tls_material(tmp_path, ["192.168.1.20", "100.101.102.103", "garvis.example.com"])
    assert material.hosts == ("192.168.1.20",)
    _verify(material, "192.168.1.20")  # still valid: the whole certificate wasn't poisoned


def test_without_name_constraints_extra_hosts_are_kept(tmp_path):
    material = ensure_tls_material(tmp_path, ["192.168.1.20", "100.101.102.103"], name_constraints=False)
    assert material.hosts == ("192.168.1.20", "100.101.102.103")
    _verify(material, "100.101.102.103")


# ------------------------------------------------------------- the server
class _Rig:
    def __init__(self, client, manager, store, pairing, web):
        self.client, self.manager, self.store, self.pairing, self.web = client, manager, store, pairing, web

    async def pair(self, user_agent: str) -> str:
        code = self.pairing.new_pairing_code()
        response = await self.client.post(
            "/api/pair", json={"code": code.code}, headers={**POST, "User-Agent": user_agent}
        )
        assert response.status == 200
        token = response.cookies[COOKIE_NAME].value
        self.client.session.cookie_jar.clear()
        return token


def _headers(token: str, user_agent: str, post: bool = True) -> dict:
    headers = {"Cookie": f"{COOKIE_NAME}={token}", "User-Agent": user_agent}
    return {**headers, **POST} if post else headers


@pytest.fixture
async def rig(tmp_path):
    events = EventBus()
    web = FakeWebsiteTool()
    agent = make_agent(tmp_path, GatedLLM(gated=False), events=events, tools=[web])
    store = ProtocolStore(tmp_path / "protocols")
    manager = TaskManager(agent, events, tmp_path / "tasks.json", store, available_tools=lambda: {"open_website"})
    manager.start()
    service = MobileService(
        agent, events, make_voice(), tasks=manager, protocols=store, available_tools=lambda: {"open_website"}
    )
    service.start()
    pairing = PairingManager(tmp_path / "devices.json")
    app = create_app(
        service,
        pairing,
        allowed_networks=parse_networks(""),
        allowed_hosts=None,
        ca_cert_der=b"0\x82fake-der",
        max_audio_bytes=1_000_000,
    )
    client = TestClient(TestServer(app, host="127.0.0.1"))
    await client.start_server()
    yield _Rig(client, manager, store, pairing, web)
    await service.stop()
    await manager.stop()
    await client.close()


@pytest.mark.parametrize(
    ("user_agent", "label"),
    [(GALAXY_S24_CHROME, "Android"), (GALAXY_SAMSUNG_INTERNET, "Android"), (IPHONE_SAFARI, "iPhone")],
)
async def test_pairing_records_the_device_type(rig, user_agent, label):
    token = await rig.pair(user_agent)
    record = rig.pairing.verify(token)
    assert record is not None and record.label == label


async def test_tasks_carry_the_real_device_for_each_platform(rig):
    android = await rig.pair(GALAXY_S24_CHROME)
    iphone = await rig.pair(IPHONE_SAFARI)
    for token, ua, request_id in (
        (android, GALAXY_S24_CHROME, "android-req-01"),
        (iphone, IPHONE_SAFARI, "iphone-req-001"),
    ):
        response = await rig.client.post(
            "/api/messages", json={"text": "hello", "request_id": request_id}, headers=_headers(token, ua)
        )
        assert response.status == 202
    await wait_until(lambda: all(t.terminal for t in rig.manager.list()))
    listing = await (await rig.client.get("/api/tasks", headers=_headers(android, GALAXY_S24_CHROME, False))).json()
    assert sorted(t["device"] for t in listing["tasks"]) == ["Android", "iPhone"]
    protocol = rig.store.save(
        Protocol(name="Weather", steps=[Step(type="open_website", params={"url": "https://w.example"})])
    )
    run = await rig.client.post(
        f"/api/protocols/{protocol.id}/run",
        json={"request_id": "android-run-01"},
        headers=_headers(android, GALAXY_S24_CHROME),
    )
    assert run.status == 202 and (await run.json())["device"] == "Android"


async def test_android_requests_get_the_same_authentication_and_errors(rig):
    wrong = await rig.client.post(
        "/api/pair", json={"code": "00000000"}, headers={**POST, "User-Agent": GALAXY_S24_CHROME}
    )
    assert wrong.status == 403 and "pairing code" in (await wrong.json())["error"].lower()
    unpaired = await rig.client.get("/api/tasks", headers={"User-Agent": GALAXY_S24_CHROME})
    assert unpaired.status == 401
    token = await rig.pair(GALAXY_S24_CHROME)
    no_header = await rig.client.post(
        "/api/messages",
        json={"text": "x"},
        headers={"Cookie": f"{COOKIE_NAME}={token}", "User-Agent": GALAXY_S24_CHROME},
    )
    assert no_header.status == 403  # anti-forgery header still required
    cross_site = await rig.client.post(
        "/api/messages",
        json={"text": "x"},
        headers={**_headers(token, GALAXY_S24_CHROME), "Origin": "https://evil.example"},
    )
    assert cross_site.status == 403
    rig.pairing.revoke(token)
    revoked = await rig.client.get("/api/tasks", headers=_headers(token, GALAXY_S24_CHROME, False))
    assert revoked.status == 401
    assert rig.manager.list() == []


async def test_ca_download_is_der_for_both_platforms(rig):
    response = await rig.client.get("/garvis-ca.crt", headers={"User-Agent": GALAXY_S24_CHROME})
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("application/x-x509-ca-cert")
    assert response.headers["Content-Disposition"].endswith('.crt"')


# ---------------------------------------------------------- installable app
async def test_manifest_meets_chrome_android_install_requirements(rig):
    manifest = await (await rig.client.get("/manifest.webmanifest")).json()
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/" and manifest["id"] == "/"
    assert manifest["name"] and manifest["short_name"]
    icons = {(i["sizes"], i.get("purpose", "any")): i["src"] for i in manifest["icons"]}
    assert ("192x192", "any") in icons and ("512x512", "any") in icons and ("512x512", "maskable") in icons
    for (sizes, _purpose), src in icons.items():
        response = await rig.client.get(src)
        assert response.status == 200 and response.headers["Content-Type"] == "image/png", src
        image = QImage.fromData(await response.read())
        assert f"{image.width()}x{image.height()}" == sizes
    sw = await (await rig.client.get("/sw.js")).text()
    assert "addEventListener('fetch'" in sw  # Chrome's installability needs a fetch handler
    for src in icons.values():
        assert f"'{src}'" in sw


def test_page_keeps_iphone_support_and_adds_android_behaviour():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # iPhone: unchanged Home Screen support
    assert 'name="apple-mobile-web-app-capable"' in html and 'rel="apple-touch-icon"' in html
    # Android: the composer stays above Chrome's on-screen keyboard
    assert "interactive-widget=resizes-content" in html
    assert "Pair this phone" in html and "Pair iPhone" not in html
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "IS_ANDROID" in js and "Safari › Microphone" in js  # platform-specific microphone help, both kept
    assert "popstate" in js  # Android back gesture closes a sheet instead of leaving the app


# ------------------------------------------------------------- the desktop
class _FakePairing:
    def new_pairing_code(self):
        import time

        from app.mobile.pairing import PairingCode

        return PairingCode(code="12345678", expires_at=time.time() + 300)

    def cancel_pairing_code(self):
        pass

    def device_count(self):
        return 1


class _FakeMobile:
    def __init__(self, data_dir: Path) -> None:
        self.is_running = True
        self.urls = ["https://192.168.10.156:8443/"]
        self.ca_fingerprint = "AB:CD:EF"
        self.pairing = _FakePairing()
        self.data_dir = data_dir
        self.ca_cert_der = b"0\x82der-bytes"


def test_pairing_dialog_guides_android_to_the_native_app_and_keeps_iphone(tmp_path):
    from app.ui import main_window

    QApplication.instance() or QApplication([])
    dialog = main_window.MobileDialog(_FakeMobile(tmp_path))
    text = dialog._text.toPlainText()
    assert "Certificate Trust Settings" in text and "Add to Home Screen" in text  # iPhone kept
    assert "Garvis-Mobile.apk" in text and "nothing to install in Android Settings" in text
    assert "GARVIS_MOBILE_REMOTE_ACCESS=tailscale" in text  # remote access is off until set up
    assert "AB:CD:EF" in text and "1234 5678" in text
    assert not hasattr(dialog, "save_android_certificate")  # the Chrome CA-install route is gone
    dialog.accept()


def test_pairing_dialog_lists_and_unpairs_single_devices(tmp_path):
    from app.mobile.pairing import PairingManager
    from app.ui import main_window

    QApplication.instance() or QApplication([])
    pairing = PairingManager(tmp_path / "devices.json")
    for _ in range(2):
        pairing.redeem(pairing.new_pairing_code().code, label="Android")
    mobile = _FakeMobile(tmp_path)
    mobile.pairing = pairing
    mobile.remote_urls = ["https://100.101.102.103:8443/"]
    dialog = main_window.MobileDialog(mobile)
    assert dialog.device_combo.count() == 2
    assert "https://100.101.102.103:8443/" in dialog._text.toPlainText()
    dialog.device_combo.setCurrentIndex(0)
    dialog._unpair_selected()
    assert pairing.device_count() == 1 and dialog.device_combo.count() == 1
    assert "Unpaired that device" in dialog._text.toPlainText()
    dialog.accept()


def test_desktop_labels_show_the_actual_phone(qapp=None):
    from PySide6.QtCore import Signal

    import app.ui.main_window as main_window
    from tests.test_ui_dashboard import _FakeBridge

    QApplication.instance() or QApplication([])

    class Bridge(_FakeBridge):
        remote_user_message = Signal(str, str)
        device = "Android"

        def mobile_device_label(self) -> str:
            return self.device

    original = main_window.AgentBridge
    main_window.AgentBridge = Bridge
    try:
        window = main_window.MainWindow()
        window._bridge.remote_user_message.emit("from the Galaxy", "mobile")
        window._bridge.device = "iPhone"
        window._bridge.remote_user_message.emit("from the iPhone", "mobile")
        text = window._conversation_view.toPlainText()
        assert "You (Android): from the Galaxy" in text and "You (iPhone): from the iPhone" in text
        window.close()
    finally:
        main_window.AgentBridge = original


def test_tools_page_lists_both_phone_platforms():
    from app.ui.dashboard import integration_rows

    rows = {r["name"]: r for r in integration_rows({"mobile_enabled": True})}
    assert rows["iPhone interface"]["action"] == ("Pair iPhone…", "pair")
    assert rows["Android interface"]["action"] == ("Pair Android phone…", "pair")
    assert json.dumps(rows)  # serializable, no widgets leaked
