"""HTTPS material for the mobile server.

Phone browsers (iOS Safari, Chrome on Android) only expose the microphone to
secure contexts, and installing the web app needs one too, so plain HTTP on
a LAN address is not enough. Garvis therefore creates its own
small certificate authority the first time the mobile interface starts:

- `ca.crt` / `ca.key`: the local root. Only the certificate is ever served
  (for the one-time install on the phone); the key never leaves
  `mobile_data_dir`. By default it carries X.509 name constraints limiting
  it to private IP ranges and *.local names, so even a stolen key could not
  be used to impersonate a real website to the phone.
- `server.crt` / `server.key`: the HTTPS certificate for this PC's current
  LAN addresses, signed by the CA. It is re-issued automatically (no
  re-install on the phone) when the PC's address changes or it nears expiry.
- On Windows both private keys are stored encrypted (PKCS#8, AES-256) under
  a random passphrase that is itself sealed with DPAPI for the current
  Windows user (`keys.dpapi`, see keyprotect.py), so a copied data folder or
  backup does not leak them. Keys written by older versions in clear are
  re-encrypted on the next start. If DPAPI can't open the passphrase (e.g.
  another Windows account) Garvis stops with an error instead of silently
  creating a new CA that the phone would not trust.

Apple's rules for certificates from a user-trusted root are followed (SAN
entries, serverAuth EKU, ECDSA P-256, a validity well under 825 days), and
these also satisfy Chrome's verifier on Android. Strict verifiers reject the
whole server certificate if any one of its names falls outside the CA's
name constraints, so such names are left out of it (and logged) rather than
breaking HTTPS for every name.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import os
import secrets
import socket
import ssl
from dataclasses import dataclass, field
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from app.core.logging import get_logger, log_extra
from app.mobile import keyprotect

logger = get_logger("mobile.tls")

CA_VALIDITY_DAYS = 3650
SERVER_VALIDITY_DAYS = 397
RENEW_BEFORE_DAYS = 30

_PERMITTED_NETWORKS = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8")
_PERMITTED_DNS = ("local", "localhost")


@dataclass(frozen=True)
class TlsMaterial:
    ca_cert_path: Path
    server_cert_path: Path
    server_key_path: Path
    hosts: tuple[str, ...]
    ca_fingerprint_sha256: str
    """Colon-separated upper-case hex, as iOS shows it under the profile's
    "More Details" and Android under the installed certificate's details --
    lets the user check they installed the right CA."""
    key_passphrase: bytes | None = field(default=None, repr=False)
    """Unlocks server.key (None: stored in clear, e.g. not Windows)."""

    def ca_cert_der(self) -> bytes:
        cert = x509.load_pem_x509_certificate(self.ca_cert_path.read_bytes())
        return cert.public_bytes(serialization.Encoding.DER)

    def server_ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(self.server_cert_path), str(self.server_key_path), password=self.key_passphrase)
        return context


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _write_private(path: Path, data: bytes) -> None:
    """Write via a temp file + rename; owner-only permissions where the OS
    supports them (on Windows the file inherits the user-profile ACL)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def _key_pem(key: ec.EllipticCurvePrivateKey, passphrase: bytes | None) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(passphrase) if passphrase else serialization.NoEncryption(),
    )


_ENCRYPTED_PEM = b"-----BEGIN ENCRYPTED PRIVATE KEY-----"


def _key_passphrase(directory: Path) -> bytes | None:
    """The passphrase that encrypts the private keys, sealed with DPAPI in
    `keys.dpapi` (created on first use). None where DPAPI doesn't exist."""
    if not keyprotect.available():
        return None
    path = directory / "keys.dpapi"
    if path.exists():
        try:
            return keyprotect.unprotect(path.read_bytes())
        except (OSError, keyprotect.KeyProtectionError) as exc:
            log_extra(logger, 40, "mobile.tls.keys_locked", error=str(exc))
            raise ValueError(
                f"Garvis's HTTPS keys in {directory} can't be unlocked by this Windows account. Run Garvis as "
                "the account that created them, or reset them (delete that folder) and install a freshly "
                "built phone app."
            ) from exc
    passphrase = secrets.token_bytes(32)
    _write_private(path, keyprotect.protect(passphrase))
    return passphrase


def _load_key(path: Path, passphrase: bytes | None) -> tuple[ec.EllipticCurvePrivateKey, bool]:
    """The key in `path`, and whether it is still stored in clear (to be re-encrypted)."""
    data = path.read_bytes()
    encrypted = data.lstrip().startswith(_ENCRYPTED_PEM)
    if encrypted and passphrase is None:
        raise ValueError(f"{path.name} is encrypted but no passphrase is available on this system.")
    key = serialization.load_pem_private_key(data, password=passphrase if encrypted else None)
    return key, not encrypted and passphrase is not None


def _fingerprint(cert: x509.Certificate) -> str:
    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def _general_name(host: str) -> x509.GeneralName | None:
    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        return None
    return x509.DNSName(host.lower()) if host else None


def _permitted_by(ca_cert: x509.Certificate, host: str) -> bool:
    """Whether the CA's name constraints (if any) allow `host` in a server certificate."""
    try:
        constraints = ca_cert.extensions.get_extension_for_class(x509.NameConstraints).value
    except x509.ExtensionNotFound:
        return True
    general = _general_name(host)
    if general is None:
        return False
    permitted = constraints.permitted_subtrees or []
    if isinstance(general, x509.IPAddress):
        nets = [n.value for n in permitted if isinstance(n, x509.IPAddress)]
        return any(general.value.version == net.version and general.value in net for net in nets)
    name = general.value.lower().rstrip(".")
    domains = [n.value.lower().lstrip(".") for n in permitted if isinstance(n, x509.DNSName)]
    return any(name == d or name.endswith("." + d) for d in domains)


def _create_ca(name_constraints: bool) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, f"Garvis Local CA ({socket.gethostname()})"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Garvis (this PC only)"),
        ]
    )
    now = _now()
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
    )
    if name_constraints:
        permitted: list[x509.GeneralName] = [
            x509.IPAddress(ipaddress.ip_network(net)) for net in _PERMITTED_NETWORKS
        ]
        permitted += [x509.DNSName(name) for name in _PERMITTED_DNS]
        builder = builder.add_extension(
            x509.NameConstraints(permitted_subtrees=permitted, excluded_subtrees=None), critical=True
        )
    return builder.sign(key, hashes.SHA256()), key


def _create_server_cert(
    hosts: tuple[str, ...], ca_cert: x509.Certificate, ca_key: ec.EllipticCurvePrivateKey
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    names = [general for host in hosts if (general := _general_name(host)) is not None]
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=SERVER_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    return cert, key


def _server_cert_is_current(
    cert_path: Path, key_path: Path, ca_cert: x509.Certificate, hosts: tuple[str, ...]
) -> bool:
    if not cert_path.exists() or not key_path.exists():
        return False
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (ValueError, x509.ExtensionNotFound):
        return False
    if cert.issuer != ca_cert.subject:
        return False
    try:
        cert.verify_directly_issued_by(ca_cert)
    except (InvalidSignature, ValueError, TypeError):
        return False
    if cert.not_valid_after_utc - _now() < dt.timedelta(days=RENEW_BEFORE_DAYS):
        return False
    wanted = {general for host in hosts if (general := _general_name(host)) is not None}
    return wanted.issubset(set(san))


def ensure_tls_material(directory: Path, hosts: list[str], *, name_constraints: bool = True) -> TlsMaterial:
    """Load the CA and server certificate from `directory`, creating the CA
    on first use and (re-)issuing the server certificate whenever it doesn't
    cover `hosts`, wasn't issued by this CA, or expires within 30 days."""
    unique_hosts = tuple(dict.fromkeys(host.strip() for host in hosts if host and host.strip()))
    if not unique_hosts:
        raise ValueError("At least one host is needed for the server certificate.")
    directory.mkdir(parents=True, exist_ok=True)
    ca_cert_path = directory / "ca.crt"
    ca_key_path = directory / "ca.key"
    server_cert_path = directory / "server.crt"
    server_key_path = directory / "server.key"

    passphrase = _key_passphrase(directory)
    if ca_cert_path.exists() and ca_key_path.exists():
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
        ca_key, in_clear = _load_key(ca_key_path, passphrase)
        if in_clear:
            _write_private(ca_key_path, _key_pem(ca_key, passphrase))
            log_extra(logger, 20, "mobile.tls.key_encrypted", key="ca")
    else:
        ca_cert, ca_key = _create_ca(name_constraints)
        _write_private(ca_key_path, _key_pem(ca_key, passphrase))
        _write_private(ca_cert_path, ca_cert.public_bytes(serialization.Encoding.PEM))
        log_extra(logger, 20, "mobile.tls.ca_created", fingerprint=_fingerprint(ca_cert))

    allowed = tuple(host for host in unique_hosts if _permitted_by(ca_cert, host))
    if len(allowed) != len(unique_hosts):
        log_extra(logger, 30, "mobile.tls.hosts_outside_name_constraints",
                  dropped=[host for host in unique_hosts if host not in allowed])
    if not allowed:
        raise ValueError("None of the host names is allowed by the local CA's name constraints.")
    unique_hosts = allowed
    if not _server_cert_is_current(server_cert_path, server_key_path, ca_cert, unique_hosts):
        cert, key = _create_server_cert(unique_hosts, ca_cert, ca_key)
        _write_private(server_key_path, _key_pem(key, passphrase))
        _write_private(server_cert_path, cert.public_bytes(serialization.Encoding.PEM))
        log_extra(logger, 20, "mobile.tls.server_cert_issued", hosts=list(unique_hosts))
    else:
        server_key, in_clear = _load_key(server_key_path, passphrase)
        if in_clear:
            _write_private(server_key_path, _key_pem(server_key, passphrase))
            log_extra(logger, 20, "mobile.tls.key_encrypted", key="server")

    return TlsMaterial(
        ca_cert_path=ca_cert_path,
        server_cert_path=server_cert_path,
        server_key_path=server_key_path,
        hosts=unique_hosts,
        ca_fingerprint_sha256=_fingerprint(ca_cert),
        key_passphrase=passphrase,
    )
