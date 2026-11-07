"""Composition root for the mobile interface, mirroring app/voice/build.py.

Takes the Agent/EventBus/VoicePipeline that AgentBridge already built --
it never calls build_agent() or build_voice_pipeline() itself, so the
phone and the desktop share one Agent, one conversation and one set of
providers, tools and permission rules.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from app.agent.agent import Agent
from app.core.config import Settings
from app.core.events import EventBus
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger, log_extra
from app.mobile.companion import Companion
from app.mobile.network import parse_networks
from app.mobile.pairing import PairingManager
from app.mobile.server import MobileServer
from app.mobile.service import MobileService
from app.voice.pipeline import VoicePipeline

logger = get_logger("mobile.build")


def build_companion(settings: Settings) -> Companion:
    """Reminders, image jobs and the phone-action broker for the native app."""
    from app.images.providers import build_image_provider
    from app.images.service import ImageService
    from app.mobile.phone_actions import PhoneActionBroker
    from app.reminders.phrasing import ReminderPhraser, voice_language
    from app.reminders.schedule import zone_database_available
    from app.reminders.store import ReminderStore
    from app.reminders.tools import ReminderZone

    if not zone_database_available():
        # Phone reminders carry IANA zones; without tzdata only one-offs can be accepted.
        log_extra(logger, 40, "reminders.tzdata_missing", python=sys.executable)
    remote = (settings.mobile_remote_access or "off").strip().lower()
    if remote not in ("off", "tailscale"):
        raise ConfigurationError(f"GARVIS_MOBILE_REMOTE_ACCESS must be 'off' or 'tailscale', not {remote!r}.")
    provider, reason = build_image_provider(settings)
    phraser = ReminderPhraser(name_en=settings.user_name, name_he=settings.user_name_he, gender=settings.user_gender)
    return Companion(
        reminders=ReminderStore(
            settings.data_dir / "reminders" / "reminders.json",
            phrase=lambda reminder: phraser.compose(reminder.id, reminder.text).text,
        ),
        zone=ReminderZone(settings.reminders_timezone),
        images=ImageService(settings.data_dir / "images", provider, reason),
        phone=PhoneActionBroker(),
        remote_access=remote,
        phraser=phraser,
        pc_voice_language=voice_language(settings.tts_voice),
    )


def build_companion_tools(settings: Settings, companion: Companion, phone_ready: Callable[[], bool]) -> list:
    """The assistant's reminder, image and phone tools (registered on the shared Agent)."""
    from app.images.tools import GenerateImageTool
    from app.mobile.phone_actions import PhoneActionTool
    from app.reminders.tools import build_reminder_tools

    return [
        *build_reminder_tools(companion.reminders, companion.zone, phone_ready),
        GenerateImageTool(companion.images),
        PhoneActionTool(companion.phone, timeout=settings.mobile_phone_action_timeout_seconds),
    ]


def build_mobile_server(
    settings: Settings,
    *,
    agent: Agent,
    events: EventBus,
    voice: VoicePipeline,
    is_desktop_session_active: Callable[[], bool],
    tasks=None,
    protocols=None,
    available_tools: Callable[[], set[str] | None] = lambda: None,
    companion: Companion | None = None,
) -> MobileServer:
    service = MobileService(
        agent,
        events,
        voice,
        is_desktop_session_active=is_desktop_session_active,
        max_message_chars=settings.mobile_max_message_chars,
        max_audio_seconds=settings.mobile_max_audio_seconds,
        min_audio_seconds=settings.stt_min_duration_seconds,
        stt_timeout_seconds=settings.mobile_stt_timeout_seconds,
        tts_timeout_seconds=settings.mobile_tts_timeout_seconds,
        max_tts_chars=settings.mobile_max_tts_chars,
        tasks=tasks,
        protocols=protocols,
        available_tools=available_tools,
    )
    pairing = PairingManager(
        settings.mobile_data_dir / "devices.json",
        code_ttl_seconds=settings.mobile_pairing_code_ttl_seconds,
        token_ttl_days=settings.mobile_device_token_days,
    )
    if companion is not None:
        companion.reminders.add_listener(service.poke)
        companion.phone.add_listener(service.poke)

        def _image_changed(job) -> None:
            if job.status == "completed":
                service.post_entry("image", image_id=job.id, prompt=job.prompt, origin=job.origin)
            else:
                service.poke()

        companion.images.add_listener(_image_changed)
    return MobileServer(
        service,
        pairing,
        host=settings.mobile_host,
        port=settings.mobile_port,
        allowed_networks=parse_networks(settings.mobile_allowed_networks),
        data_dir=settings.mobile_data_dir,
        extra_hosts=[host.strip() for host in settings.mobile_extra_hosts.split(",") if host.strip()],
        name_constraints=settings.mobile_ca_name_constraints,
        max_audio_bytes=settings.mobile_max_audio_bytes,
        companion=companion,
        remote_access=companion.remote_access if companion is not None else "off",
        remote_hosts=[host.strip() for host in settings.mobile_remote_hosts.split(",") if host.strip()],
    )
