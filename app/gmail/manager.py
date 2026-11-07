"""Gmail API access via OAuth 2.0 (installed-app flow).

This is the ONLY module that imports Google's auth/API client libraries,
mirroring how `app/browser/manager.py` is the only module that imports
Playwright: tools in `app/tools/gmail.py` talk to `GmailManager`, never to
Google's SDKs directly. The libraries are imported lazily inside
`_build_service_sync`, exactly like Playwright/faster-whisper/winsound
elsewhere, so importing this module never requires them to be installed.

Credential/token lifecycle:
  - `Settings.gmail_client_secret_path`: the OAuth client secret downloaded
    from Google Cloud Console (gitignored, user-provided; see README
    "Setting up Gmail"). Garvis never sees or stores the user's password.
  - `Settings.gmail_token_path`: the cached user token (gitignored),
    written after the first interactive consent and refreshed
    automatically after that.
  - The interactive consent flow and any token refresh are both blocking
    network/browser calls, so both run via `asyncio.to_thread`, matching
    how BrowserManager/PiperProvider keep blocking work off the event loop.
  - An expired-but-refreshable token is refreshed silently and re-saved. A
    *revoked* (or otherwise unrefreshable) token raises `GmailAuthError`
    with a message telling the user to re-authenticate -- never the raw
    exception internals, and never the token/secret itself. Structured log
    fields never include credentials either (and `app/core/logging.py`'s
    `redact()` masks common secret-shaped keys defensively either way).

A single API service client is lazily built on first use and kept alive
across tool calls (cheaper than rebuilding per call), the same caching
strategy `BrowserManager` uses for its browser/page.
"""

from __future__ import annotations

import asyncio
import base64
from email.mime.text import MIMEText
from pathlib import Path

from app.core.exceptions import GmailAuthError, GmailError
from app.core.logging import get_logger, log_extra

logger = get_logger("gmail")

_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]
_MAX_BODY_CHARS = 8000


class GmailManager:
    """Owns Gmail OAuth credentials and a lazily-built API service client."""

    def __init__(self, client_secret_path: Path, token_path: Path) -> None:
        self._client_secret_path = client_secret_path
        self._token_path = token_path
        self._service = None
        self._lock = asyncio.Lock()

    @property
    def is_authenticated(self) -> bool:
        return self._service is not None

    async def _ensure_service(self):
        async with self._lock:
            if self._service is not None:
                return self._service
            self._service = await asyncio.to_thread(self._build_service_sync)
            return self._service

    def _build_service_sync(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise GmailError(
                "Gmail dependencies are not installed. Run: pip install "
                "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            ) from exc

        creds = self._load_cached_credentials(Credentials)
        creds = self._refresh_if_needed(creds, Request)

        if not creds or not creds.valid:
            creds = self._run_interactive_consent(InstalledAppFlow)

        self._save_credentials(creds)

        try:
            return build("gmail", "v1", credentials=creds)
        except Exception as exc:
            log_extra(logger, 40, "gmail.service_build_failed")
            raise GmailError(f"Could not start the Gmail client: {exc}") from exc

    def _load_cached_credentials(self, credentials_cls):
        if not self._token_path.exists():
            return None
        try:
            return credentials_cls.from_authorized_user_file(str(self._token_path), _SCOPES)
        except Exception as exc:
            log_extra(logger, 30, "gmail.token_load_failed")
            raise GmailAuthError(
                "Garvis's saved Gmail credentials are corrupted or unreadable. Delete "
                f"{self._token_path} and try again to re-authenticate."
            ) from exc

    def _refresh_if_needed(self, creds, request_cls):
        if creds is None or not creds.expired or not creds.refresh_token:
            return creds
        try:
            creds.refresh(request_cls())
        except Exception as exc:
            log_extra(logger, 30, "gmail.token_refresh_failed")
            raise GmailAuthError(
                "Garvis's Gmail authorization has expired or was revoked. Delete "
                f"{self._token_path} and try again to re-authenticate."
            ) from exc
        return creds

    def _run_interactive_consent(self, flow_cls):
        if not self._client_secret_path.exists():
            raise GmailError(
                f"No Gmail OAuth client secret found at {self._client_secret_path}. "
                "See README 'Setting up Gmail' for how to create one."
            )
        try:
            flow = flow_cls.from_client_secrets_file(str(self._client_secret_path), _SCOPES)
            return flow.run_local_server(port=0)
        except Exception as exc:
            log_extra(logger, 40, "gmail.consent_flow_failed")
            raise GmailAuthError(f"Gmail sign-in did not complete: {exc}") from exc

    def _save_credentials(self, creds) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(creds.to_json(), encoding="utf-8")

    # ------------------------------------------------------------------ API
    async def search(self, query: str, max_results: int) -> list[dict]:
        service = await self._ensure_service()
        return await asyncio.to_thread(self._search_sync, service, query, max_results)

    def _search_sync(self, service, query: str, max_results: int) -> list[dict]:
        try:
            listing = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        except Exception as exc:
            log_extra(logger, 40, "gmail.search_failed")
            raise GmailError(f"Gmail search failed: {exc}") from exc
        return [self._get_summary_sync(service, item["id"]) for item in listing.get("messages", [])]

    def _get_summary_sync(self, service, message_id: str) -> dict:
        try:
            msg = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"],
                )
                .execute()
            )
        except Exception as exc:
            log_extra(logger, 40, "gmail.message_summary_failed")
            raise GmailError(f"Could not read message '{message_id}': {exc}") from exc
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        return {
            "id": message_id,
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
        }

    async def read_message(self, message_id: str) -> dict:
        service = await self._ensure_service()
        return await asyncio.to_thread(self._read_message_sync, service, message_id)

    def _read_message_sync(self, service, message_id: str) -> dict:
        try:
            msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        except Exception as exc:
            log_extra(logger, 40, "gmail.message_read_failed")
            raise GmailError(f"Could not read message '{message_id}': {exc}") from exc
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body = self._extract_body(msg.get("payload", {}))
        return {
            "id": message_id,
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "body": body[:_MAX_BODY_CHARS],
            "truncated": len(body) > _MAX_BODY_CHARS,
        }

    @staticmethod
    def _extract_body(payload: dict) -> str:
        def _decode(data: str) -> str:
            return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")

        body_data = payload.get("body", {}).get("data")
        if payload.get("mimeType") == "text/plain" and body_data:
            return _decode(body_data)
        for part in payload.get("parts", []) or []:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                return _decode(part["body"]["data"])
        for part in payload.get("parts", []) or []:
            text = GmailManager._extract_body(part)
            if text:
                return text
        return _decode(body_data) if body_data else ""

    async def send(self, to: str, subject: str, body: str) -> str:
        service = await self._ensure_service()
        return await asyncio.to_thread(self._send_sync, service, to, subject, body)

    def _send_sync(self, service, to: str, subject: str, body: str) -> str:
        raw = self._encode_message(to, subject, body)
        try:
            sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        except Exception as exc:
            log_extra(logger, 40, "gmail.send_failed")
            raise GmailError(f"Could not send email to '{to}': {exc}") from exc
        return sent.get("id", "")

    async def create_draft(self, to: str, subject: str, body: str) -> str:
        service = await self._ensure_service()
        return await asyncio.to_thread(self._create_draft_sync, service, to, subject, body)

    def _create_draft_sync(self, service, to: str, subject: str, body: str) -> str:
        raw = self._encode_message(to, subject, body)
        try:
            draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        except Exception as exc:
            log_extra(logger, 40, "gmail.draft_failed")
            raise GmailError(f"Could not create draft to '{to}': {exc}") from exc
        return draft.get("id", "")

    @staticmethod
    def _encode_message(to: str, subject: str, body: str) -> str:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
