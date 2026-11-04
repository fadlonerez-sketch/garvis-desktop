"""Reaching Garvis over Tailscale: `tailscale status --json` parsing, the
least-privilege grants policy, and the pinned HTTPS probe in
scripts/tailscale_check.py (run against a real TLS listener made with
Garvis's own certificate code -- and an impostor CA, which must fail).

No Tailscale, network or admin rights needed: everything runs on 127.0.0.1."""
from __future__ import annotations

import json
import re
import socket
import ssl
import sys
import threading
from pathlib import Path

import pytest

from app.mobile.network import parse_tailscale_status
from app.mobile.tls import ensure_tls_material
from scripts import tailscale_check

STATUS = {
    "Version": "1.90.6-t1234",
    "BackendState": "Running",
    "Self": {"HostName": "DESKTOP-EXAMPLE", "DNSName": "desktop-example.tail1a2b3.ts.net.", "OS": "windows",
             "TailscaleIPs": ["100.101.1.2", "fd7a:115c:a1e0::1"], "Online": True, "KeyExpiry": "2027-04-30T00:00:00Z"},
    "CurrentTailnet": {"Name": "someone@example.com", "MagicDNSSuffix": "tail1a2b3.ts.net", "MagicDNSEnabled": True},
    "Health": [],
    "Peer": {
        "nodekey:aa": {"HostName": "Example Phone", "DNSName": "example-phone.tail1a2b3.ts.net.",
                       "OS": "android", "TailscaleIPs": ["100.90.8.7"], "Online": True, "CurAddr": "",
                       "Relay": "fra", "ExitNode": False},
        "nodekey:bb": {"HostName": "old-laptop", "DNSName": "old-laptop.tail1a2b3.ts.net.", "OS": "linux",
                       "TailscaleIPs": ["100.77.1.1"], "Online": False, "CurAddr": "203.0.113.5:41641", "Relay": "lhr"},
    },
}


def test_status_parsing_and_connection_kinds():
    status = parse_tailscale_status(STATUS)
    assert status.backend_state == "Running" and status.version.startswith("1.90")
    assert status.self_node.dns_name == "desktop-example.tail1a2b3.ts.net"
    assert status.self_node.ips[0] == "100.101.1.2"
    assert status.magic_dns_enabled and status.magic_dns_suffix == "tail1a2b3.ts.net"
    assert not status.exit_node_in_use
    phone = status.find_peer("example-phone")
    assert phone is status.find_peer("100.90.8.7") is status.find_peer("Example Phone")
    assert phone.connection == "DERP relay (fra)"  # no direct path: relayed, still end-to-end encrypted
    assert status.find_peer("old-laptop").connection == "direct (203.0.113.5:41641)"
    assert status.find_peer("nobody") is None


def test_exit_node_use_is_detected():
    data = json.loads(json.dumps(STATUS))
    data["Peer"]["nodekey:bb"]["ExitNode"] = True
    assert parse_tailscale_status(data).exit_node_in_use
    assert parse_tailscale_status({**STATUS, "ExitNodeStatus": {"Online": True}}).exit_node_in_use


def test_signed_out_status_has_no_self_addresses():
    status = parse_tailscale_status({"Version": "1.90.6", "BackendState": "NeedsLogin", "Self": {}})
    assert status.backend_state == "NeedsLogin" and status.self_node.ips == () and status.peers == ()


def _hujson(text: str) -> dict:
    no_comments = re.sub(r"//[^\n]*", "", text)
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", no_comments))


def test_policy_grants_the_phone_only_tcp_8443_on_the_pc():
    policy = _hujson(tailscale_check.build_policy("100.101.1.2", "100.90.8.7"))
    assert policy["hosts"] == {"garvis-pc": "100.101.1.2", "garvis-phone": "100.90.8.7"}
    assert policy["grants"] == [{"src": ["garvis-phone"], "dst": ["garvis-pc"], "ip": ["tcp:8443"]}]
    assert "acls" not in policy  # no allow-all left behind
    accept = [t for t in policy["tests"] if t.get("accept")]
    assert accept == [policy["tests"][0]] and accept[0]["accept"] == ["garvis-pc:8443"]
    assert "garvis-pc:3389" in policy["tests"][0]["deny"]
    assert policy["tests"][1] == {"src": "garvis-pc", "deny": ["garvis-phone:8443", "garvis-phone:22",
                                                              "garvis-phone:443"]}


@pytest.mark.parametrize("pc,phone", [("192.168.10.156", "100.90.8.7"), ("100.101.1.2", "8.8.8.8"),
                                      ("fd7a:115c:a1e0::1", "100.90.8.7")])
def test_policy_refuses_addresses_outside_the_tailnet(pc, phone):
    with pytest.raises(ValueError):
        tailscale_check.build_policy(pc, phone)


def test_broad_python_allow_rules_are_flagged():
    rules = [{"id": "Garvis-Mobile-TCP8443-Tailscale", "remote": "100.90.8.7"},
             {"id": "Garvis-Mobile-TCP8443-Phone", "remote": "192.168.10.200"},
             {"id": "TCP Query User{x}", "remote": "Any"}]
    assert [r["id"] for r in tailscale_check.broad_rules(rules)] == ["TCP Query User{x}"]


# ------------------------------------------------------------------ HTTPS probe
class _TlsHello:
    """A one-endpoint HTTPS server on 127.0.0.1 with a server certificate from `material`."""

    def __init__(self, material) -> None:
        self.context = material.server_ssl_context()
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.hosts: list[str] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                raw, _ = self.sock.accept()
            except OSError:
                return
            try:
                with self.context.wrap_socket(raw, server_side=True) as tls:
                    request = tls.recv(65536).decode("ascii", "replace")
                    self.hosts.append(re.search(r"\r\nHost: ([^\r]+)", request).group(1))
                    body = b'{"api": 1, "server": "garvis", "paired": false, "remote": true}'
                    tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            except (OSError, ssl.SSLError):
                pass

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def garvis_tls(tmp_path):
    material = ensure_tls_material(tmp_path / "pc", ["192.168.10.156", "desktop-test.local", "127.0.0.1"])
    server = _TlsHello(material)
    yield material, server
    server.close()


def test_probe_verifies_the_garvis_ca_and_the_pcs_permanent_name(garvis_tls):
    material, server = garvis_tls
    hello = tailscale_check.probe_hello("127.0.0.1", server.port, material.ca_cert_path, "desktop-test.local")
    assert hello["remote"] is True
    assert server.hosts == [f"desktop-test.local:{server.port}"]  # the Host header the app sends


def test_probe_refuses_a_certificate_that_does_not_name_the_pc(garvis_tls):
    material, server = garvis_tls
    with pytest.raises(ssl.SSLCertVerificationError):
        tailscale_check.probe_hello("127.0.0.1", server.port, material.ca_cert_path, "other-pc.local")


def test_probe_refuses_an_impostor_ca(garvis_tls, tmp_path):
    _, server = garvis_tls
    impostor = ensure_tls_material(tmp_path / "impostor", ["desktop-test.local"])
    with pytest.raises(ssl.SSLCertVerificationError):
        tailscale_check.probe_hello("127.0.0.1", server.port, Path(impostor.ca_cert_path), "desktop-test.local")


# ------------------------------------------------------------ firewall script
FIREWALL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tailscale_firewall.ps1"


def _firewall_check(*args: str):
    import subprocess

    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(FIREWALL_SCRIPT), "-Check", *args],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows Firewall script")
@pytest.mark.skipif(not FIREWALL_SCRIPT.exists(),
                    reason="scripts/tailscale_firewall.ps1 (Garvis Mobile remote access) is not part of this repository")
@pytest.mark.parametrize("address", ["8.8.8.8", "192.168.10.200", "100.128.0.1", "100.63.255.255", "fd00::1"])
def test_firewall_script_admits_only_tailnet_addresses(address):
    result = _firewall_check("-PhoneIp", address)  # -Check: read-only, no admin
    assert result.returncode != 0 and "is not a Tailscale address" in result.stderr + result.stdout


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows Firewall script")
@pytest.mark.skipif(not FIREWALL_SCRIPT.exists(),
                    reason="scripts/tailscale_firewall.ps1 (Garvis Mobile remote access) is not part of this repository")
def test_firewall_script_check_accepts_the_phones_tailnet_addresses():
    result = _firewall_check("-PhoneIp", "100.90.8.7,fd7a:115c:a1e0::9")
    assert result.returncode == 0, result.stderr
    assert "Tailscale adapter" in result.stdout
