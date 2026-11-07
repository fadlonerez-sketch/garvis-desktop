"""Pairing a phone (iPhone or Android) with this PC.

1. The desktop's phone dialog asks for a one-time pairing code: 8 random
   digits, valid for a few minutes, burnt after 5 wrong guesses.
2. The phone submits it once; in exchange it gets a 256-bit random device
   token, stored as an HttpOnly/Secure/SameSite=Strict cookie (never
   readable by page JavaScript).
3. Only a SHA-256 hash of each token is kept on disk (devices.json), with an
   expiry. "Unpair all" in the dialog revokes every device at once.

Each device has a `kind`: "phone" (the iPhone web app, the Garvis Android
app) or "station" (Garvis Eye Station, the tablet face). A station is a
display/voice client only: the native API gives its token a narrower set of
routes and never treats it as a phone (phone actions, reminder ringing).

Thread-safe: the desktop dialog runs on the Qt thread while the server
redeems/verifies on the asyncio loop thread.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger, log_extra

logger = get_logger("mobile.pairing")

CODE_DIGITS = 8
DEVICE_KINDS = ("phone", "station")
MAX_CODE_ATTEMPTS = 5
MAX_DEVICES = 10


class PairingError(Exception):
    """A pairing attempt was refused. The message is safe to show the user."""


@dataclass(frozen=True)
class PairingCode:
    code: str
    expires_at: float

    @property
    def display(self) -> str:
        return f"{self.code[:4]} {self.code[4:]}"


@dataclass(frozen=True)
class DeviceRecord:
    id: str
    label: str
    created_at: float
    expires_at: float
    kind: str = "phone"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PairingManager:
    def __init__(
        self,
        store_path: Path,
        *,
        code_ttl_seconds: float = 300.0,
        token_ttl_days: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store_path = store_path
        self._code_ttl = code_ttl_seconds
        self._token_ttl = token_ttl_days * 86400.0
        self._clock = clock
        self._lock = threading.Lock()
        self._code: PairingCode | None = None
        self._failed_attempts = 0
        self._devices: dict[str, DeviceRecord] = self._load()
        """SHA-256 of the device token -> its record."""

    # ----------------------------------------------------------- storage
    def _load(self) -> dict[str, DeviceRecord]:
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except OSError, ValueError:
            log_extra(logger, 30, "mobile.pairing.store_unreadable")
            return {}
        devices: dict[str, DeviceRecord] = {}
        for item in raw.get("devices", []):
            try:
                record = DeviceRecord(
                    id=str(item["id"]),
                    label=str(item.get("label", ""))[:80],
                    created_at=float(item["created_at"]),
                    expires_at=float(item["expires_at"]),
                    kind=item.get("kind") if item.get("kind") in DEVICE_KINDS else "phone",
                )
                token_hash = str(item["token_sha256"])
            except KeyError, TypeError, ValueError:
                continue
            devices[token_hash] = record
        return devices

    def _save_locked(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "devices": [
                {
                    "id": record.id,
                    "label": record.label,
                    "created_at": record.created_at,
                    "expires_at": record.expires_at,
                    "kind": record.kind,
                    "token_sha256": token_hash,
                }
                for token_hash, record in self._devices.items()
            ]
        }
        tmp = self._store_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, self._store_path)

    def _drop_expired_locked(self) -> bool:
        now = self._clock()
        expired = [key for key, record in self._devices.items() if record.expires_at <= now]
        for key in expired:
            del self._devices[key]
        return bool(expired)

    # ------------------------------------------------------ pairing codes
    def new_pairing_code(self) -> PairingCode:
        """Issue a fresh code, replacing (and invalidating) any previous one."""
        code = "".join(secrets.choice("0123456789") for _ in range(CODE_DIGITS))
        with self._lock:
            self._code = PairingCode(code=code, expires_at=self._clock() + self._code_ttl)
            self._failed_attempts = 0
            return self._code

    def cancel_pairing_code(self) -> None:
        with self._lock:
            self._code = None

    def redeem(self, code: str, label: str = "", kind: str = "phone") -> str:
        """Exchange a valid pairing code for a new device token (returned
        once, never stored in clear). Raises PairingError."""
        if kind not in DEVICE_KINDS:
            raise ValueError(f"Unknown device kind {kind!r}.")
        submitted = "".join(ch for ch in str(code) if ch.isdigit())
        with self._lock:
            active = self._code
            if active is None or active.expires_at <= self._clock():
                self._code = None
                raise PairingError(
                    "No pairing code is active. Open the phone dialog on the PC (Settings > Phone > Pair phone...) for a new code."
                )
            if not hmac.compare_digest(submitted.encode(), active.code.encode()):
                self._failed_attempts += 1
                if self._failed_attempts >= MAX_CODE_ATTEMPTS:
                    self._code = None
                    log_extra(logger, 30, "mobile.pairing.code_burnt")
                    raise PairingError("Too many wrong codes. Generate a new code on the PC.")
                raise PairingError("Wrong pairing code.")
            self._code = None  # single use
            self._drop_expired_locked()
            if len(self._devices) >= MAX_DEVICES:
                raise PairingError("Too many paired devices. Use 'Unpair all' on the PC first.")
            token = secrets.token_urlsafe(32)
            now = self._clock()
            record = DeviceRecord(
                id=secrets.token_hex(4),
                label=label[:80],
                created_at=now,
                expires_at=now + self._token_ttl,
                kind=kind,
            )
            self._devices[_hash_token(token)] = record
            self._save_locked()
        log_extra(logger, 20, "mobile.pairing.device_paired", device=record.id, kind=kind)
        return token

    # -------------------------------------------------------- device tokens
    def verify(self, token: str | None) -> DeviceRecord | None:
        if not token or len(token) > 256:
            return None
        token_hash = _hash_token(token)
        with self._lock:
            record = self._devices.get(token_hash)
            if record is None:
                return None
            if record.expires_at <= self._clock():
                del self._devices[token_hash]
                self._save_locked()
                return None
            return record

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            if self._devices.pop(_hash_token(token), None) is not None:
                self._save_locked()

    def rotate(self, token: str | None) -> tuple[str, DeviceRecord] | None:
        """Swap a valid device token for a fresh one with a new expiry (same
        device id); the old token stops working at once. None if invalid."""
        if not token or len(token) > 256:
            return None
        old_hash = _hash_token(token)
        with self._lock:
            record = self._devices.get(old_hash)
            if record is None or record.expires_at <= self._clock():
                return None
            new_token = secrets.token_urlsafe(32)
            now = self._clock()
            renewed = DeviceRecord(
                id=record.id,
                label=record.label,
                created_at=record.created_at,
                expires_at=now + self._token_ttl,
                kind=record.kind,
            )
            del self._devices[old_hash]
            self._devices[_hash_token(new_token)] = renewed
            self._save_locked()
        log_extra(logger, 20, "mobile.pairing.token_rotated", device=record.id)
        return new_token, renewed

    def devices(self) -> list[DeviceRecord]:
        with self._lock:
            if self._drop_expired_locked():
                self._save_locked()
            return sorted(self._devices.values(), key=lambda r: r.created_at)

    def revoke_device(self, device_id: str) -> bool:
        """Unpair one device by its id (as listed by devices())."""
        with self._lock:
            key = next((k for k, r in self._devices.items() if r.id == device_id), None)
            if key is None:
                return False
            del self._devices[key]
            self._save_locked()
        log_extra(logger, 20, "mobile.pairing.device_revoked", device=device_id)
        return True

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._devices)
            self._devices.clear()
            self._code = None
            self._save_locked()
        log_extra(logger, 20, "mobile.pairing.all_revoked", count=count)
        return count

    def device_count(self, kind: str | None = None) -> int:
        """Paired devices (of one `kind` if given)."""
        with self._lock:
            if self._drop_expired_locked():
                self._save_locked()
            return sum(1 for record in self._devices.values() if kind is None or record.kind == kind)

    @property
    def token_ttl_seconds(self) -> float:
        return self._token_ttl
