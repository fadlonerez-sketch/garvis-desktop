"""Read-only check: can Garvis Mobile reach this PC over Tailscale?

Reports, without changing anything:
- Tailscale: installed, version, signed in, this PC's tailnet addresses and
  MagicDNS name, whether an exit node is in use (it must not be), and --
  for the phone given with --phone -- whether it is online and how traffic
  flows: direct, Tailscale peer relay or DERP relay.
- Whether this network lets the PC reach Tailscale's servers at all (some
  campus/guest networks block them).
- Garvis: remote access setting, the port, and an HTTPS probe of
  https://<this PC's tailnet IP>:<port>/api/v1/hello made exactly like the
  app makes it -- trusting only Garvis's own CA (data/mobile/ca.crt) and
  checking the PC's permanent name. Expect `"remote": true`.
- Windows Firewall: the Garvis Tailscale rule (scripts/tailscale_firewall.ps1)
  and any broad inbound allow rule for Garvis's Python.

  python scripts/tailscale_check.py [--phone NAME_OR_IP] [--ping]
  python scripts/tailscale_check.py --phone NAME_OR_IP --policy   # print the grants policy

`--ping` runs `tailscale ping` to the phone (tailnet-only traffic) to show
whether a direct path or a DERP relay is used. Nothing is ever sent to
Garvis except GET /api/v1/hello, which carries no token. Exit code 0 when
every required check passes.
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mobile.network import (  # noqa: E402
    TAILSCALE_CLI,
    TailnetStatus,
    is_tailscale_address,
    local_hostname,
    tailscale_status,
)

FIREWALL_RULE_ID = "Garvis-Mobile-TCP8443-Tailscale"
CONTROL_HOSTS = ("controlplane.tailscale.com", "login.tailscale.com")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def build_policy(pc_ip: str, phone_ip: str, port: int = 8443) -> str:
    """A least-privilege tailnet policy (HuJSON): the phone may open TCP `port`
    on the PC and nothing else; nothing may reach the phone. Replaces the
    default allow-all policy -- add grants for any other devices you use."""
    for address in (pc_ip, phone_ip):
        if not is_tailscale_address(address) or ":" in address:
            raise ValueError(f"{address!r} is not a Tailscale IPv4 address (100.64.0.0/10).")
    return f"""// Garvis: least-privilege tailnet policy (grants).
// Admin console > Access controls > JSON editor. This REPLACES the default allow-all policy:
// if other devices in this tailnet must reach each other, add grants for them before saving.
// The console runs the "tests" below on save and refuses a policy that fails them.
{{
  "hosts": {{
    "garvis-pc":    "{pc_ip}",
    "garvis-phone": "{phone_ip}",
  }},

  "grants": [
    // The phone may open exactly one thing on the PC: Garvis's HTTPS API.
    {{"src": ["garvis-phone"], "dst": ["garvis-pc"], "ip": ["tcp:{port}"]}},
  ],

  "tests": [
    {{"src": "garvis-phone", "accept": ["garvis-pc:{port}"],
      "deny": ["garvis-pc:22", "garvis-pc:80", "garvis-pc:443", "garvis-pc:445", "garvis-pc:3389", "garvis-pc:8080"]}},
    // Nothing is granted towards the phone.
    {{"src": "garvis-pc", "deny": ["garvis-phone:{port}", "garvis-phone:22", "garvis-phone:443"]}},
  ],
}}
"""


def probe_hello(address: str, port: int, ca_file: Path, canonical_host: str, timeout: float = 10.0) -> dict:
    """GET /api/v1/hello over TLS as the app does it: connect to `address`,
    trust only `ca_file`, require the certificate to name `canonical_host`."""
    context = ssl.create_default_context(cafile=str(ca_file))  # only Garvis's CA, no system roots
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    with socket.create_connection((address, port), timeout=timeout) as raw, \
            context.wrap_socket(raw, server_hostname=canonical_host) as tls:
        tls.sendall(f"GET /api/v1/hello HTTP/1.1\r\nHost: {canonical_host}:{port}\r\n"
                    "Accept: application/json\r\nConnection: close\r\n\r\n".encode("ascii"))
        data = b""
        while chunk := tls.recv(65536):
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:200].decode('utf-8', 'replace')}")
    return json.loads(body)


def control_plane_reachable(host: str, timeout: float = 8.0) -> str:
    """'' if a verified TLS connection to `host` works, else why not."""
    try:
        with socket.create_connection((host, 443), timeout=timeout) as raw, \
                ssl.create_default_context().wrap_socket(raw, server_hostname=host):
            return ""
    except (OSError, ssl.SSLError) as exc:
        return f"{type(exc).__name__}: {exc}"


def firewall_report(program_hint: str = "python") -> tuple[dict | None, list[dict]]:
    """The Garvis Tailscale rule (or None) and every enabled inbound allow rule
    for a Python program (via read-only Get-NetFirewall* cmdlets)."""
    script = rf"""
$ErrorActionPreference = 'SilentlyContinue'
function Describe($r) {{
  [pscustomobject]@{{ id=$r.Name; action=[string]$r.Action; enabled=[string]$r.Enabled; profile=[string]$r.Profile;
    port=(($r | Get-NetFirewallPortFilter).LocalPort -join ','); protocol=[string]($r | Get-NetFirewallPortFilter).Protocol;
    remote=(($r | Get-NetFirewallAddressFilter).RemoteAddress -join ',');
    interface=(($r | Get-NetFirewallInterfaceFilter).InterfaceAlias -join ',');
    program=($r | Get-NetFirewallApplicationFilter).Program }}
}}
$rule = Get-NetFirewallRule -Name '{FIREWALL_RULE_ID}'
$python = Get-NetFirewallApplicationFilter | Where-Object {{ $_.Program -like '*{program_hint}*' }} | Get-NetFirewallRule |
  Where-Object {{ $_.Direction -eq 'Inbound' -and $_.Action -eq 'Allow' -and $_.Enabled -eq 'True' }}
@{{ rule = $(if ($rule) {{ Describe $rule }}); python = @($python | ForEach-Object {{ Describe $_ }}) }} | ConvertTo-Json -Depth 4
"""
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=90, creationflags=_NO_WINDOW)
        data = json.loads(result.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, []
    rule = data.get("rule")
    python = data.get("python") or []
    python = python if isinstance(python, list) else [python]
    return (rule if isinstance(rule, dict) and rule.get("id") else None,
            [r for r in python if isinstance(r, dict) and r.get("id")])


def broad_rules(rules: list[dict]) -> list[dict]:
    """Allow rules that admit anyone (RemoteAddress Any) -- they would undo the narrow rule."""
    return [r for r in rules if r.get("id") != FIREWALL_RULE_ID and "any" in str(r.get("remote", "")).lower()]


def tailscale_ping(target: str) -> str:
    try:
        result = subprocess.run([str(TAILSCALE_CLI), "ping", "--c", "3", "--timeout", "5s", target],
                                capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(tailscale ping failed: {exc})"
    return (result.stdout + result.stderr).strip()


def _line(ok: bool | None, text: str) -> None:
    print(f"  [{'OK' if ok else ('--' if ok is None else 'FAIL')}] {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phone", help="the phone's Tailscale name or 100.x address")
    parser.add_argument("--ping", action="store_true", help="tailscale ping the phone (direct vs DERP)")
    parser.add_argument("--policy", action="store_true", help="print the least-privilege grants policy and exit")
    args = parser.parse_args(argv)

    from app.core.config import get_settings

    settings = get_settings()
    port = settings.mobile_port
    status: TailnetStatus | None = tailscale_status()
    me = status.self_node if status else None
    pc_ipv4 = next((ip for ip in (me.ips if me else ()) if "." in ip), "")
    phone = status.find_peer(args.phone) if status and args.phone else None

    if args.policy:
        if not pc_ipv4 or phone is None:
            print("Need Tailscale running on this PC and --phone naming a device in the tailnet.", file=sys.stderr)
            return 2
        print(build_policy(pc_ipv4, next(ip for ip in phone.ips if "." in ip), port))
        return 0

    failed = False
    print("Tailscale on this PC")
    if not TAILSCALE_CLI.exists():
        _line(False, f"not installed ({TAILSCALE_CLI} missing) -- see docs/TAILSCALE.md")
        failed = True
    elif status is None:
        _line(False, "installed, but the Tailscale service isn't answering")
        failed = True
    else:
        _line(True, f"version {status.version}")
        running = status.backend_state == "Running"
        _line(running, f"state {status.backend_state}" + ("" if running else " -- sign in from the tray icon"))
        failed |= not running
        if me:
            _line(bool(pc_ipv4), f"this PC: {', '.join(me.ips) or 'no address'}  name {me.dns_name or '?'}")
            _line(None, f"node key expires {me.key_expiry or 'never (expiry disabled)'}")
        _line(status.magic_dns_enabled or None, f"MagicDNS {'on' if status.magic_dns_enabled else 'off'}"
              + (f" ({status.magic_dns_suffix})" if status.magic_dns_suffix else ""))
        _line(not status.exit_node_in_use, "no exit node in use" if not status.exit_node_in_use
              else "an EXIT NODE is in use -- turn it off (Garvis needs only the tailnet)")
        failed |= status.exit_node_in_use
        for item in status.health:
            _line(None, f"Tailscale health: {item}")
    for host in CONTROL_HOSTS:
        problem = control_plane_reachable(host)
        _line(not problem, f"{host} reachable from this network" if not problem else
              f"{host} NOT reachable from this network ({problem}) -- this network blocks Tailscale")
        failed |= bool(problem)

    if args.phone:
        print(f"\nPhone '{args.phone}'")
        if phone is None:
            _line(False, "not in this tailnet (sign the phone in to the same tailnet and approve it)")
            failed = True
        else:
            _line(phone.online, f"{phone.host_name} ({phone.os}) {', '.join(phone.ips)} "
                  + ("online" if phone.online else "OFFLINE -- is the Tailscale VPN on in the phone?"))
            _line(None, f"path: {phone.connection}")
            failed |= not phone.online
            if args.ping:
                print("\n".join("      " + line for line in tailscale_ping(phone.ips[0]).splitlines()))

    print("\nGarvis")
    remote = (settings.mobile_remote_access or "off").strip().lower()
    _line(settings.mobile_enabled, f"mobile interface {'enabled' if settings.mobile_enabled else 'DISABLED'}")
    _line(remote == "tailscale", f"GARVIS_MOBILE_REMOTE_ACCESS={remote}")
    failed |= not settings.mobile_enabled or remote != "tailscale"
    ca_file = settings.mobile_data_dir / "ca.crt"
    if pc_ipv4 and ca_file.exists():
        try:
            hello = probe_hello(pc_ipv4, port, ca_file, local_hostname())
            ok = hello.get("server") == "garvis" and hello.get("remote") is True
            _line(ok, f"https://{pc_ipv4}:{port}/api/v1/hello -> certificate OK (Garvis CA, {local_hostname()}), "
                  f"remote={hello.get('remote')}")
            failed |= not ok
        except (OSError, ssl.SSLError, RuntimeError, ValueError) as exc:
            _line(False, f"https://{pc_ipv4}:{port}/api/v1/hello failed: {type(exc).__name__}: {exc} "
                  "(is Garvis running, restarted since the .env change?)")
            failed = True
    else:
        _line(None, "HTTPS probe skipped (no tailnet address yet, or Garvis hasn't created its CA)")

    print("\nWindows Firewall (read-only)")
    rule, python_rules = firewall_report()
    if rule is None:
        _line(False, f"{FIREWALL_RULE_ID} missing -- run scripts\\tailscale_firewall.ps1 as administrator")
        failed = True
    else:
        narrow = rule.get("remote", "").lower() not in ("", "any") and "100.64.0.0" not in rule.get("remote", "")
        _line(narrow, f"{FIREWALL_RULE_ID}: enabled={rule.get('enabled')} TCP {rule.get('port')} "
              f"from {rule.get('remote')} on '{rule.get('interface')}'"
              + ("" if narrow else " -- not limited to the phone; re-run the script with -PhoneIp"))
    for broad in broad_rules(python_rules):
        _line(False, f"broad allow rule {broad.get('id')} ({broad.get('profile')}, TCP {broad.get('port')}, "
              f"from {broad.get('remote')}) for {broad.get('program')}")
        failed = True
    print("\n" + ("Some checks failed." if failed else "All required checks passed."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
