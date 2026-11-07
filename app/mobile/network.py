"""Local-network helpers for the mobile server: which client addresses may
connect, and which addresses this PC is reachable at (for the certificate
and the URL shown in the pairing dialog)."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ConfigurationError

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

DEFAULT_ALLOWED_NETWORKS: tuple[str, ...] = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "127.0.0.0/8",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)
"""Private, link-local and loopback ranges only. Deliberately excludes
100.64.0.0/10 (carrier-grade NAT / overlay VPNs) and every public range."""


TAILSCALE_NETWORKS: tuple[IPNetwork, ...] = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fd7a:115c:a1e0::/48"),
)
"""Addresses Tailscale assigns to the devices of a tailnet. Only trusted for
remote access when the connection also ARRIVED on this PC's own Tailscale
address (see is_tailscale_connection) -- a CGNAT source address alone
proves nothing."""


def _address(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not text:
        return None
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address


def is_tailscale_address(text: str | None) -> bool:
    address = _address(text)
    return address is not None and any(
        address.version == network.version and address in network for network in TAILSCALE_NETWORKS
    )


def is_tailscale_connection(remote: str | None, local: str | None) -> bool:
    """The peer is a tailnet device AND it reached this PC through this PC's
    Tailscale interface (the socket's local address is a tailnet address)."""
    return is_tailscale_address(remote) and is_tailscale_address(local)


TAILSCALE_CLI = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe"
    if sys.platform == "win32"
    else Path("/usr/bin/tailscale")
)


def _tailscale_cli_addresses(cli: Path = TAILSCALE_CLI) -> list[str]:
    """`tailscale ip -4` -- for when the host name doesn't resolve to the
    tailnet address. Read-only, local, no window; [] if it isn't there."""
    if not cli.exists():
        return []
    try:
        result = subprocess.run(
            [str(cli), "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError, subprocess.SubprocessError:
        return []
    return [line.strip() for line in result.stdout.splitlines() if is_tailscale_address(line.strip())]


@dataclass(frozen=True)
class TailnetNode:
    """One device as `tailscale status --json` describes it."""

    host_name: str
    dns_name: str
    os: str
    ips: tuple[str, ...]
    online: bool
    direct_address: str
    """ip:port of a direct (peer-to-peer) WireGuard path; empty if none."""
    relay: str
    """Home DERP region code (e.g. "fra"); used for traffic only while there is no direct path."""
    peer_relay: str
    key_expiry: str
    is_exit_node: bool

    @property
    def connection(self) -> str:
        """How traffic to this peer currently flows."""
        if self.direct_address:
            return f"direct ({self.direct_address})"
        if self.peer_relay:
            return f"Tailscale peer relay ({self.peer_relay})"
        if self.relay:
            return f"DERP relay ({self.relay})"
        return "no path yet (idle)"


@dataclass(frozen=True)
class TailnetStatus:
    version: str
    backend_state: str
    """"Running" when signed in and connected; "NeedsLogin", "Stopped", ... otherwise."""
    self_node: TailnetNode | None
    magic_dns_enabled: bool
    magic_dns_suffix: str
    exit_node_in_use: bool
    peers: tuple[TailnetNode, ...]
    health: tuple[str, ...]

    def find_peer(self, name_or_ip: str) -> TailnetNode | None:
        wanted = name_or_ip.strip().lower().rstrip(".")
        for peer in self.peers:
            if wanted in peer.ips or wanted in (
                peer.host_name.lower(),
                peer.dns_name.lower(),
                peer.dns_name.lower().split(".", 1)[0],
            ):
                return peer
        return None


def _node(raw: dict) -> TailnetNode:
    return TailnetNode(
        host_name=str(raw.get("HostName") or ""),
        dns_name=str(raw.get("DNSName") or "").rstrip("."),
        os=str(raw.get("OS") or ""),
        ips=tuple(str(ip) for ip in raw.get("TailscaleIPs") or ()),
        online=bool(raw.get("Online")),
        direct_address=str(raw.get("CurAddr") or ""),
        relay=str(raw.get("Relay") or ""),
        peer_relay=str(raw.get("PeerRelay") or ""),
        key_expiry=str(raw.get("KeyExpiry") or ""),
        is_exit_node=bool(raw.get("ExitNode")),
    )


def parse_tailscale_status(data: dict) -> TailnetStatus:
    """The parts of `tailscale status --json` that matter for reaching Garvis."""
    tailnet = data.get("CurrentTailnet") or {}
    peers = tuple(_node(raw) for raw in (data.get("Peer") or {}).values() if isinstance(raw, dict))
    return TailnetStatus(
        version=str(data.get("Version") or ""),
        backend_state=str(data.get("BackendState") or ""),
        self_node=_node(data["Self"]) if isinstance(data.get("Self"), dict) else None,
        magic_dns_enabled=bool(tailnet.get("MagicDNSEnabled")),
        magic_dns_suffix=str(tailnet.get("MagicDNSSuffix") or data.get("MagicDNSSuffix") or ""),
        exit_node_in_use=bool(data.get("ExitNodeStatus")) or any(peer.is_exit_node for peer in peers),
        peers=peers,
        health=tuple(str(item) for item in data.get("Health") or ()),
    )


def tailscale_status(cli: Path = TAILSCALE_CLI) -> TailnetStatus | None:
    """`tailscale status --json` -- read-only and local; None if Tailscale
    isn't installed or its service isn't answering."""
    if not cli.exists():
        return None
    try:
        result = subprocess.run(
            [str(cli), "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(result.stdout)
    except OSError, subprocess.SubprocessError, ValueError:
        return None
    return parse_tailscale_status(data) if isinstance(data, dict) else None


def detect_tailscale_addresses() -> list[str]:
    """This PC's Tailscale addresses (empty when Tailscale isn't running)."""
    found: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
    except OSError:
        infos = []
    for info in infos:
        candidate = str(info[4][0])
        if is_tailscale_address(candidate) and candidate.split("%", 1)[0] not in found:
            found.append(candidate.split("%", 1)[0])
    if not any("." in address for address in found):
        found = [*_tailscale_cli_addresses(), *found]
    return list(dict.fromkeys(found))


def parse_networks(spec: str) -> list[IPNetwork]:
    """Parse a comma-separated CIDR list; empty means DEFAULT_ALLOWED_NETWORKS."""
    items = [item.strip() for item in spec.split(",") if item.strip()] or list(DEFAULT_ALLOWED_NETWORKS)
    networks: list[IPNetwork] = []
    for item in items:
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise ConfigurationError(f"Invalid network in GARVIS_MOBILE_ALLOWED_NETWORKS: '{item}'") from exc
    return networks


def is_client_allowed(remote: str | None, networks: Iterable[IPNetwork]) -> bool:
    """True if `remote` (a peer IP string, possibly IPv4-mapped IPv6 or with
    a zone id) falls inside one of `networks`. Unknown/unparseable -> False."""
    if not remote:
        return False
    try:
        address = ipaddress.ip_address(remote.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return any(address.version == network.version and address in network for network in networks)


def _primary_ipv4() -> str | None:
    """The address of the interface that holds the default route. A UDP
    "connect" sends no packets; it only makes the OS pick a source address."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed anywhere real
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def detect_lan_addresses() -> list[str]:
    """Private, non-loopback IPv4 addresses of this PC, primary interface first."""
    candidates: list[str] = []
    primary = _primary_ipv4()
    if primary:
        candidates.append(primary)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(str(info[4][0]))
    except OSError:
        pass
    result: list[str] = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.is_private and not address.is_loopback and not address.is_link_local:
            if candidate not in result:
                result.append(candidate)
    return result


def local_hostname() -> str:
    """This PC's mDNS name (<computer-name>.local), lower-case."""
    return f"{socket.gethostname().split('.')[0].lower()}.local"
