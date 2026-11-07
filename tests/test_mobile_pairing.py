"""PairingManager: one-time codes, device tokens, expiry and revocation."""

from __future__ import annotations

import json

import pytest

from app.mobile.pairing import MAX_CODE_ATTEMPTS, PairingError, PairingManager


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _manager(tmp_path, clock=None, **kwargs) -> PairingManager:
    return PairingManager(tmp_path / "devices.json", clock=clock or _Clock(), **kwargs)


def test_code_is_eight_random_digits_and_single_use(tmp_path):
    manager = _manager(tmp_path)
    code = manager.new_pairing_code()
    assert len(code.code) == 8 and code.code.isdigit()
    token = manager.redeem(code.display, label="iPhone")  # "1234 5678" form is accepted
    assert manager.verify(token) is not None
    with pytest.raises(PairingError):
        manager.redeem(code.code)


def test_codes_are_not_predictable(tmp_path):
    manager = _manager(tmp_path)
    codes = {manager.new_pairing_code().code for _ in range(50)}
    assert len(codes) > 45


def test_no_code_means_no_pairing(tmp_path):
    with pytest.raises(PairingError, match="No pairing code"):
        _manager(tmp_path).redeem("12345678")


def test_expired_code_is_refused(tmp_path):
    clock = _Clock()
    manager = _manager(tmp_path, clock, code_ttl_seconds=60)
    code = manager.new_pairing_code()
    clock.now += 61
    with pytest.raises(PairingError):
        manager.redeem(code.code)


def test_code_is_burnt_after_too_many_wrong_guesses(tmp_path):
    manager = _manager(tmp_path)
    code = manager.new_pairing_code()
    wrong = "00000000" if code.code != "00000000" else "11111111"
    for _ in range(MAX_CODE_ATTEMPTS - 1):
        with pytest.raises(PairingError, match="Wrong"):
            manager.redeem(wrong)
    with pytest.raises(PairingError, match="Too many"):
        manager.redeem(wrong)
    with pytest.raises(PairingError):
        manager.redeem(code.code)  # even the right code is useless now


def test_only_token_hashes_are_stored_and_survive_restart(tmp_path):
    manager = _manager(tmp_path)
    token = manager.redeem(manager.new_pairing_code().code)
    stored = (tmp_path / "devices.json").read_text(encoding="utf-8")
    assert token not in stored
    assert len(json.loads(stored)["devices"]) == 1
    assert _manager(tmp_path).verify(token) is not None


def test_tokens_expire(tmp_path):
    clock = _Clock()
    manager = _manager(tmp_path, clock, token_ttl_days=1)
    token = manager.redeem(manager.new_pairing_code().code)
    clock.now += 86401
    assert manager.verify(token) is None
    assert manager.device_count() == 0


def test_revoke_and_revoke_all(tmp_path):
    manager = _manager(tmp_path)
    first = manager.redeem(manager.new_pairing_code().code)
    second = manager.redeem(manager.new_pairing_code().code)
    manager.revoke(first)
    assert manager.verify(first) is None and manager.verify(second) is not None
    manager.new_pairing_code()
    assert manager.revoke_all() == 1
    assert manager.verify(second) is None
    with pytest.raises(PairingError):
        manager.redeem("12345678")  # revoke_all also cancels any open code


@pytest.mark.parametrize("token", [None, "", "x" * 300, "not-a-real-token"])
def test_garbage_tokens_are_rejected(tmp_path, token):
    assert _manager(tmp_path).verify(token) is None


def test_corrupt_store_starts_empty_instead_of_crashing(tmp_path):
    (tmp_path / "devices.json").write_text("{not json", encoding="utf-8")
    assert _manager(tmp_path).device_count() == 0
