"""Actions Garvis asks the paired Android phone to perform.

The PC never controls the phone directly. The assistant's `phone_action`
tool files a request here; the Garvis app on the phone picks it up through
its authenticated event stream, decides locally whether it may run it, asks
the user first where the phone's rules say so, executes it with standard
Android APIs (or its accessibility service, if the user enabled it) and
reports what really happened. The tool waits for that report -- an action
is "done" only when the phone says so.

Guarantees:
- Only actions in ACTIONS exist, each with validated parameters; each needs
  a capability the user switched on in the phone app (reported by the phone,
  re-checked on the phone before running).
- Consequential actions (calling/messaging/sharing, reading the screen)
  are marked `confirm`: the phone always asks the user, every time.
- The phone's "Stop automation" switch pauses everything; requests then
  fail immediately with that reason.
- A request nobody answers expires (the phone drops it from its list).
"""
from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger, log_extra
from app.tools.base import RiskLevel, Tool, ToolResult

logger = get_logger("mobile.phone_actions")

CAPABILITIES = ("apps", "alarms", "device", "communication", "screen_control", "screen_reading", "images")
"""What the user can switch on in the phone app. "images" (saving Garvis's
generated images to the gallery) is always on."""

ACTIONS: dict[str, dict[str, Any]] = {
    "open_app": {"capability": "apps", "required": ("app",)},
    "open_url": {"capability": "apps", "required": ("url",)},
    "web_search": {"capability": "apps", "required": ("query",)},
    "navigate_to": {"capability": "apps", "required": ("destination",)},
    "open_settings": {"capability": "apps", "required": ("page",)},
    "set_alarm": {"capability": "alarms", "required": ("hour", "minute")},
    "set_timer": {"capability": "alarms", "required": ("seconds",)},
    "flashlight": {"capability": "device", "required": ("on",)},
    "media": {"capability": "device", "required": ("command",)},
    "dial": {"capability": "communication", "required": ("number",), "confirm": True},
    "compose_sms": {"capability": "communication", "required": ("number",), "confirm": True},
    "share_text": {"capability": "communication", "required": ("text",), "confirm": True},
    "go_back": {"capability": "screen_control", "required": ()},
    "go_home": {"capability": "screen_control", "required": ()},
    "recents": {"capability": "screen_control", "required": ()},
    "notifications": {"capability": "screen_control", "required": ()},
    "quick_settings": {"capability": "screen_control", "required": ()},
    "click_text": {"capability": "screen_control", "required": ("text",)},
    "type_text": {"capability": "screen_control", "required": ("text",)},
    "scroll": {"capability": "screen_control", "required": ("direction",)},
    "read_screen": {"capability": "screen_reading", "required": (), "confirm": True},
    "save_image": {"capability": "images", "required": ("image_id",)},
}
_URL = re.compile(r"^https?://", re.IGNORECASE)
_NUMBER = re.compile(r"^[0-9+*#() -]{3,32}$")
_SETTINGS_PAGES = ("wifi", "bluetooth", "display", "sound", "battery", "notifications", "accessibility",
                   "location", "app", "main")
_MEDIA = ("play_pause", "next", "previous", "volume_up", "volume_down", "mute")
DEVICE_FRESH_SECONDS = 75.0
"""A phone counts as connected if it polled within this window."""
RESULT_STATUSES = ("done", "failed", "declined", "unsupported")


class PhoneActionError(ValueError):
    """Invalid request or no phone able to run it; user-facing."""


def validate_params(action: str, params: dict[str, Any]) -> dict[str, Any]:
    spec = ACTIONS.get(action)
    if spec is None:
        raise PhoneActionError(f"Unknown phone action {action!r}.")
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    missing = [name for name in spec["required"] if name not in clean]
    if missing:
        raise PhoneActionError(f"{action} needs: {', '.join(missing)}.")
    for key, value in clean.items():
        if isinstance(value, str) and len(value) > 1000:
            raise PhoneActionError(f"'{key}' is too long.")
    if action == "open_url" and not _URL.match(str(clean["url"])):
        raise PhoneActionError("Only http(s) links can be opened.")
    if action in ("dial", "compose_sms") and not _NUMBER.match(str(clean["number"])):
        raise PhoneActionError("That isn't a phone number.")
    if action == "open_settings" and clean["page"] not in _SETTINGS_PAGES:
        raise PhoneActionError(f"page must be one of {', '.join(_SETTINGS_PAGES)}.")
    if action == "media" and clean["command"] not in _MEDIA:
        raise PhoneActionError(f"command must be one of {', '.join(_MEDIA)}.")
    if action == "set_alarm" and not (0 <= int(clean["hour"]) <= 23 and 0 <= int(clean["minute"]) <= 59):
        raise PhoneActionError("Invalid alarm time.")
    if action == "set_timer" and not 1 <= int(clean["seconds"]) <= 24 * 3600:
        raise PhoneActionError("A timer must be between 1 second and 24 hours.")
    if action == "scroll" and clean["direction"] not in ("up", "down"):
        raise PhoneActionError("direction must be 'up' or 'down'.")
    return clean


@dataclass
class PhoneDevice:
    device_id: str
    label: str = ""
    capabilities: set[str] = field(default_factory=set)
    paused: bool = False
    accessibility: bool = False
    last_seen: float = 0.0


@dataclass
class PendingAction:
    id: str
    device_id: str
    action: str
    params: dict[str, Any]
    confirm: bool
    created_at: float
    expires_at: float
    future: asyncio.Future

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "action": self.action, "params": self.params, "confirm": self.confirm,
                "created_at": self.created_at, "expires_at": self.expires_at}


class PhoneActionBroker:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._devices: dict[str, PhoneDevice] = {}
        self._pending: dict[str, PendingAction] = {}
        self._listeners: list[Callable[[], None]] = []
        self.history: list[dict[str, Any]] = []

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:  # noqa: BLE001
                logger.exception("phone_actions.listener_failed")

    # -------------------------------------------------------------- devices
    def seen(self, device_id: str, label: str = "") -> None:
        device = self._devices.setdefault(device_id, PhoneDevice(device_id=device_id))
        device.last_seen = self._clock()
        if label:
            device.label = label

    def report(self, device_id: str, capabilities: list[str], *, paused: bool, accessibility: bool) -> PhoneDevice:
        device = self._devices.setdefault(device_id, PhoneDevice(device_id=device_id))
        device.capabilities = {c for c in capabilities if c in CAPABILITIES} | {"images"}
        device.paused, device.accessibility, device.last_seen = bool(paused), bool(accessibility), self._clock()
        if paused:
            for pending in [p for p in self._pending.values() if p.device_id == device_id]:
                self._resolve(pending, {"status": "declined", "message": "Automation is stopped on the phone."})
        self._notify()
        return device

    def forget(self, device_id: str) -> None:
        self._devices.pop(device_id, None)
        for pending in [p for p in self._pending.values() if p.device_id == device_id]:
            self._resolve(pending, {"status": "failed", "message": "The phone was unpaired."})

    def connected_devices(self) -> list[PhoneDevice]:
        now = self._clock()
        return sorted((d for d in self._devices.values() if now - d.last_seen <= DEVICE_FRESH_SECONDS),
                      key=lambda d: d.last_seen, reverse=True)

    def describe(self) -> dict[str, Any]:
        devices = self.connected_devices()
        if not devices:
            return {"connected": False}
        device = devices[0]
        return {"connected": True, "paused": device.paused, "capabilities": sorted(device.capabilities),
                "accessibility": device.accessibility}

    # -------------------------------------------------------------- actions
    def _target(self, capability: str) -> PhoneDevice:
        devices = self.connected_devices()
        if not devices:
            raise PhoneActionError("The Garvis app isn't connected on the phone right now (open it and try again).")
        for device in devices:
            if device.paused:
                continue
            if capability in device.capabilities:
                return device
        if all(d.paused for d in devices):
            raise PhoneActionError("Automation is stopped in the phone app (Phone control > Stop automation).")
        raise PhoneActionError(f"The '{capability}' capability is switched off in the phone app "
                               "(Phone control). Ask the user to enable it there.")

    async def request(self, action: str, params: dict[str, Any], *, timeout: float = 90.0) -> dict[str, Any]:
        clean = validate_params(action, params)
        spec = ACTIONS[action]
        device = self._target(spec["capability"])
        now = self._clock()
        pending = PendingAction(id=secrets.token_hex(8), device_id=device.device_id, action=action, params=clean,
                                confirm=bool(spec.get("confirm")), created_at=now, expires_at=now + timeout,
                                future=asyncio.get_running_loop().create_future())
        self._pending[pending.id] = pending
        log_extra(logger, 20, "phone_actions.requested", action=action, request=pending.id)
        self._notify()
        try:
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=timeout)
        except TimeoutError:
            self._resolve(pending, {"status": "failed", "message": "The phone didn't answer in time."})
            return pending.future.result()
        finally:
            self._pending.pop(pending.id, None)
            self._notify()

    def pending_for(self, device_id: str) -> list[dict[str, Any]]:
        return [p.public() for p in self._pending.values() if p.device_id == device_id]

    def complete(self, device_id: str, action_id: str, status: str, message: str = "",
                 data: dict[str, Any] | None = None) -> bool:
        pending = self._pending.get(action_id)
        if pending is None or pending.device_id != device_id or status not in RESULT_STATUSES:
            return False
        self._resolve(pending, {"status": status, "message": str(message)[:500], "data": data or {}})
        return True

    def _resolve(self, pending: PendingAction, result: dict[str, Any]) -> None:
        if pending.future.done():
            return
        pending.future.set_result(result)
        self.history.append({"action": pending.action, "status": result["status"], "time": self._clock()})
        del self.history[:-50]
        log_extra(logger, 20, "phone_actions.finished", action=pending.action, status=result["status"])


class PhoneActionInput(BaseModel):
    action: Literal[
        "open_app", "open_url", "web_search", "navigate_to", "open_settings", "set_alarm", "set_timer",
        "flashlight", "media", "dial", "compose_sms", "share_text", "go_back", "go_home", "recents",
        "notifications", "quick_settings", "click_text", "type_text", "scroll", "read_screen", "save_image",
    ]
    app: str | None = Field(default=None, description="open_app: the app's name as shown on the phone.")
    url: str | None = Field(default=None, description="open_url: an http(s) link.")
    query: str | None = Field(default=None, description="web_search: what to search for.")
    destination: str | None = Field(default=None, description="navigate_to: address or place.")
    page: str | None = Field(default=None, description="open_settings: wifi, bluetooth, display, sound, battery, "
                             "notifications, accessibility, location, app or main.")
    hour: int | None = Field(default=None, description="set_alarm: 0-23.")
    minute: int | None = Field(default=None, description="set_alarm: 0-59.")
    seconds: int | None = Field(default=None, description="set_timer: duration in seconds.")
    label: str | None = Field(default=None, description="set_alarm/set_timer: optional label.")
    on: bool | None = Field(default=None, description="flashlight: true = on.")
    command: str | None = Field(default=None, description="media: play_pause, next, previous, volume_up, "
                                "volume_down or mute.")
    number: str | None = Field(default=None, description="dial/compose_sms: phone number.")
    text: str | None = Field(default=None, description="share_text/type_text: the text; click_text: the visible "
                             "label of the button or item to tap.")
    direction: str | None = Field(default=None, description="scroll: up or down.")
    image_id: str | None = Field(default=None, description="save_image: id returned by generate_image.")


class PhoneActionTool(Tool):
    name = "phone_action"
    description = (
        "Do something on the user's Android phone through the Garvis app: open an app or link, search, "
        "navigate, open a settings page, set an alarm/timer, flashlight, media keys, open the dialer or an SMS "
        "draft (the user presses call/send), share text, save a generated image to the gallery, and -- only if "
        "the user enabled screen control -- back/home/recents, tap a visible item by its label, type, scroll, "
        "or read the visible screen. The phone asks the user before calls, messages, sharing and screen "
        "reading. Returns what the phone actually did; if it fails or is declined, say so. This acts on the "
        "PHONE, not the PC."
    )
    risk_level = RiskLevel.LOW
    input_model = PhoneActionInput

    def __init__(self, broker: PhoneActionBroker, timeout: float = 90.0) -> None:
        self._broker = broker
        self._timeout = timeout

    async def execute(self, args: PhoneActionInput) -> ToolResult:
        params = args.model_dump(exclude={"action"}, exclude_none=True)
        try:
            result = await self._broker.request(args.action, params, timeout=self._timeout)
        except PhoneActionError as exc:
            return ToolResult.fail(str(exc))
        if result.get("status") == "done":
            return ToolResult.ok({"status": "done", "message": result.get("message", ""),
                                  **({"data": result["data"]} if result.get("data") else {})})
        reason = {"declined": "The user declined it on the phone", "unsupported": "The phone can't do that",
                  "failed": "It failed on the phone"}.get(result.get("status"), "It failed on the phone")
        message = result.get("message") or ""
        return ToolResult.fail(f"{reason}{': ' + message if message else '.'}")
