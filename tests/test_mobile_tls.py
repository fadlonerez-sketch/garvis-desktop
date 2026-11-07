"""Local CA + server certificate for the mobile HTTPS endpoint."""

from __future__ import annotations

import datetime as dt
import ipaddress

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from app.mobile import keyprotect
from app.mobile.tls import ensure_tls_material


def _load(path):
    return x509.load_pem_x509_certificate(path.read_bytes())


def test_first_run_creates_a_name_constrained_ca_and_a_matching_server_cert(tmp_path):
    material = ensure_tls_material(tmp_path, ["192.168.1.20", "pc.local", "localhost"])
    ca = _load(material.ca_cert_path)
    server = _load(material.server_cert_path)

    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is True
    constraints = ca.extensions.get_extension_for_class(x509.NameConstraints)
    assert constraints.critical
    permitted = constraints.value.permitted_subtrees
    assert x509.IPAddress(ipaddress.ip_network("192.168.0.0/16")) in permitted
    assert x509.DNSName("local") in permitted

    server.verify_directly_issued_by(ca)
    san = server.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert x509.IPAddress(ipaddress.ip_address("192.168.1.20")) in san
    assert x509.DNSName("pc.local") in san
    eku = server.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    # Apple: certificates from a user-trusted root must be valid <= 825 days.
    assert server.not_valid_after_utc - server.not_valid_before_utc < dt.timedelta(days=825)
    assert material.ca_fingerprint_sha256.count(":") == 31


def test_ca_is_reused_and_server_cert_only_reissued_when_hosts_change(tmp_path):
    first = ensure_tls_material(tmp_path, ["192.168.1.20", "localhost"])
    server_bytes = first.server_cert_path.read_bytes()

    again = ensure_tls_material(tmp_path, ["192.168.1.20", "localhost"])
    assert again.ca_fingerprint_sha256 == first.ca_fingerprint_sha256
    assert again.server_cert_path.read_bytes() == server_bytes

    moved = ensure_tls_material(tmp_path, ["192.168.1.77", "localhost"])
    assert moved.ca_fingerprint_sha256 == first.ca_fingerprint_sha256  # no re-install on the phone
    assert moved.server_cert_path.read_bytes() != server_bytes
    san = _load(moved.server_cert_path).extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert x509.IPAddress(ipaddress.ip_address("192.168.1.77")) in san


def test_name_constraints_can_be_turned_off_for_a_new_ca(tmp_path):
    material = ensure_tls_material(tmp_path, ["localhost"], name_constraints=False)
    ca = _load(material.ca_cert_path)
    assert not any(isinstance(ext.value, x509.NameConstraints) for ext in ca.extensions)


def test_ca_download_is_the_public_certificate_only(tmp_path):
    material = ensure_tls_material(tmp_path, ["localhost"])
    der = material.ca_cert_der()
    assert x509.load_der_x509_certificate(der).subject == _load(material.ca_cert_path).subject
    assert b"PRIVATE KEY" not in der


# ------------------------------------------------- private keys at rest (DPAPI)
windows_only = pytest.mark.skipif(not keyprotect.available(), reason="DPAPI is Windows-only")


@windows_only
def test_private_keys_are_encrypted_at_rest_under_a_dpapi_sealed_passphrase(tmp_path):
    material = ensure_tls_material(tmp_path, ["192.168.1.20", "localhost"])
    for name in ("ca.key", "server.key"):
        assert (tmp_path / name).read_bytes().startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----"), name
    sealed = (tmp_path / "keys.dpapi").read_bytes()
    assert keyprotect.unprotect(sealed) == material.key_passphrase and material.key_passphrase not in sealed
    assert "key_passphrase" not in repr(material)
    material.server_ssl_context()  # OpenSSL opens the encrypted server key with it


@windows_only
def test_keys_written_in_clear_by_an_older_version_are_encrypted_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(keyprotect, "available", lambda: False)
    old = ensure_tls_material(tmp_path, ["192.168.1.20", "localhost"])
    assert b"ENCRYPTED" not in (tmp_path / "ca.key").read_bytes()
    server_bytes = old.server_cert_path.read_bytes()
    monkeypatch.undo()

    new = ensure_tls_material(tmp_path, ["192.168.1.20", "localhost"])
    assert new.ca_fingerprint_sha256 == old.ca_fingerprint_sha256  # same CA: the phone keeps trusting it
    assert new.server_cert_path.read_bytes() == server_bytes
    for name in ("ca.key", "server.key"):
        assert (tmp_path / name).read_bytes().startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----"), name
    new.server_ssl_context()


@windows_only
def test_a_passphrase_this_account_cant_open_stops_instead_of_replacing_the_ca(tmp_path):
    ensure_tls_material(tmp_path, ["localhost"])
    ca_bytes = (tmp_path / "ca.crt").read_bytes()
    (tmp_path / "keys.dpapi").write_bytes(b"not a DPAPI blob")
    with pytest.raises(ValueError, match="can't be unlocked"):
        ensure_tls_material(tmp_path, ["localhost"])
    assert (tmp_path / "ca.crt").read_bytes() == ca_bytes


def test_tailnet_addresses_stay_out_of_the_name_constrained_certificate(tmp_path):
    # The CA may only vouch for private LAN ranges and *.local; a Tailscale (100.64/10) address or a
    # *.ts.net name is left out. The phone reaches the PC's tailnet IP and checks the PC's .local name.
    material = ensure_tls_material(tmp_path, ["192.168.1.20", "pc.local", "100.101.102.103", "pc.tail1.ts.net"])
    assert material.hosts == ("192.168.1.20", "pc.local")
