"""The parts of Garvis the native phone app adds on top of the shared
Agent and TaskManager: reminders, images and phone actions. Built once by
AgentBridge (app/mobile/build.py:build_companion) and handed both to the
Agent (as tools) and to the native API (as routes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.presence import PresenceHub
from app.core.voice_ownership import VoiceOwnershipManager
from app.images.service import ImageService
from app.mobile.phone_actions import PhoneActionBroker
from app.reminders.phrasing import ReminderPhraser
from app.reminders.store import Reminder, ReminderStore
from app.reminders.tools import ReminderZone


@dataclass
class Companion:
    reminders: ReminderStore
    zone: ReminderZone
    images: ImageService
    phone: PhoneActionBroker
    remote_access: str = "off"
    phraser: ReminderPhraser = field(default_factory=ReminderPhraser)
    pc_voice_language: str = "en"
    presence: PresenceHub = field(default_factory=PresenceHub)
    """Garvis's display state (app/core/presence.py), served at /api/v1/presence."""
    voice_ownership: VoiceOwnershipManager = field(default_factory=VoiceOwnershipManager)
    """Which one endpoint may use a microphone (app/core/voice_ownership.py), at /api/v1/voice/*."""
    """The language the PC's Piper voice speaks; other reminders are spoken by the phone."""

    def spoken(self, reminder: Reminder) -> str:
        """What Garvis says when this reminder rings (stored, else composed now)."""
        return reminder.spoken or self.phraser.compose(reminder.id, reminder.text, remember=False).text

    def features(self) -> dict[str, Any]:
        return {
            "reminders": True,
            "calls": True,
            "images": {
                "available": self.images.available,
                "provider": self.images.provider_name,
                "reason": self.images.unavailable_reason,
            },
            "phone_actions": True,
            "remote_access": self.remote_access,
        }
