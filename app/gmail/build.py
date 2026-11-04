"""Composition helper for Gmail, mirroring app/browser/build.py: reads
config from Settings so nothing about the OAuth client secret / token
paths is hardcoded elsewhere.
"""
from __future__ import annotations

from app.core.config import Settings
from app.gmail.manager import GmailManager


def build_gmail_manager(settings: Settings) -> GmailManager:
    return GmailManager(
        client_secret_path=settings.gmail_client_secret_path,
        token_path=settings.gmail_token_path,
    )
