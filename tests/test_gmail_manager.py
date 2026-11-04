"""Hermetic tests for GmailManager, using fake google-auth/googleapiclient
modules injected via sys.modules -- no real network, no real OAuth
consent, and no requirement that the google-* packages even be installed
(mirrors how test_browser_manager.py fakes Playwright)."""
from __future__ import annotations

import sys
import types

import pytest

from app.core.exceptions import GmailAuthError, GmailError
from app.gmail.manager import GmailManager


# --------------------------------------------------------------------- fakes
class FakeCredentials:
    """Stand-in for google.oauth2.credentials.Credentials."""

    def __init__(self, valid=True, expired=False, refresh_token="rt", token="at", broken_refresh=False):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.token = token
        self._broken_refresh = broken_refresh
        self.refreshed = False

    def refresh(self, request) -> None:
        if self._broken_refresh:
            raise RuntimeError("invalid_grant: token has been revoked")
        self.refreshed = True
        self.valid = True
        self.expired = False
        self.token = "refreshed-token"

    def to_json(self) -> str:
        return f'{{"token": "{self.token}"}}'

    @classmethod
    def from_authorized_user_file(cls, path, scopes):
        if path == "corrupt":
            raise ValueError("bad json")
        return cls(valid=False, expired=True)  # simulate a stale-on-disk token needing refresh


class FakeFlow:
    def __init__(self, creds):
        self._creds = creds

    @classmethod
    def from_client_secrets_file(cls, path, scopes):
        return cls(FakeCredentials(valid=True, expired=False))

    def run_local_server(self, port=0):
        return self._creds


class FakeMessagesResource:
    def __init__(self, store: dict[str, dict], fail: bool = False):
        self._store = store
        self._fail = fail

    def list(self, userId, q, maxResults):
        if self._fail:
            return _FailingExecute()
        matched = [{"id": mid} for mid in list(self._store)[:maxResults]]
        return _StaticExecute({"messages": matched})

    def get(self, userId, id, format, metadataHeaders=None):
        if self._fail:
            return _FailingExecute()
        if id not in self._store:
            return _FailingExecute(f"Message {id} not found")
        return _StaticExecute(self._store[id])

    def send(self, userId, body):
        if self._fail:
            return _FailingExecute()
        return _StaticExecute({"id": "sent-123"})


class FakeDraftsResource:
    def __init__(self, fail: bool = False):
        self._fail = fail

    def create(self, userId, body):
        if self._fail:
            return _FailingExecute()
        return _StaticExecute({"id": "draft-456"})


class FakeUsersResource:
    def __init__(self, store, fail=False):
        self._messages = FakeMessagesResource(store, fail=fail)
        self._drafts = FakeDraftsResource(fail=fail)

    def messages(self):
        return self._messages

    def drafts(self):
        return self._drafts


class FakeGmailService:
    def __init__(self, store=None, fail: bool = False):
        self._users = FakeUsersResource(store or {}, fail=fail)

    def users(self):
        return self._users


class _StaticExecute:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FailingExecute:
    def __init__(self, message: str = "Gmail API error 500"):
        self._message = message

    def execute(self):
        raise RuntimeError(self._message)


# --------------------------------------------------------------- installer
def _install_fake_google(monkeypatch: pytest.MonkeyPatch, *, service, flow_cls=FakeFlow) -> None:
    for name in (
        "google.auth",
        "google.auth.transport",
        "google.auth.transport.requests",
        "google.oauth2",
        "google.oauth2.credentials",
        "google_auth_oauthlib",
        "google_auth_oauthlib.flow",
        "googleapiclient",
        "googleapiclient.discovery",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["google.auth.transport.requests"].Request = object
    sys.modules["google.oauth2.credentials"].Credentials = FakeCredentials
    sys.modules["google_auth_oauthlib.flow"].InstalledAppFlow = flow_cls
    sys.modules["googleapiclient.discovery"].build = lambda name, version, credentials: service


# ------------------------------------------------------------------- tests
@pytest.mark.asyncio
async def test_first_time_auth_runs_interactive_consent_and_saves_token(tmp_path, monkeypatch):
    service = FakeGmailService()
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")
    assert not manager.is_authenticated

    result = await manager.search("in:inbox", 5)

    assert result == []  # empty fake store
    assert manager.is_authenticated
    assert (tmp_path / "token.json").exists()


@pytest.mark.asyncio
async def test_missing_client_secret_raises_gmail_error(tmp_path, monkeypatch):
    _install_fake_google(monkeypatch, service=FakeGmailService())
    manager = GmailManager(
        client_secret_path=tmp_path / "does_not_exist.json", token_path=tmp_path / "token.json"
    )

    with pytest.raises(GmailError, match="No Gmail OAuth client secret"):
        await manager.search("in:inbox", 5)


@pytest.mark.asyncio
async def test_expired_token_is_refreshed_and_reused_silently(tmp_path, monkeypatch):
    service = FakeGmailService()
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "token.json").write_text('{"token": "stale"}')

    await manager.search("in:inbox", 5)

    saved = (tmp_path / "token.json").read_text()
    assert "refreshed-token" in saved  # the refreshed credential was re-saved, not the stale one


@pytest.mark.asyncio
async def test_revoked_token_raises_gmail_auth_error(tmp_path, monkeypatch):
    """Regression coverage for Phase 9's 'handle expired/revoked tokens
    correctly' requirement: a refresh failure must surface as a distinct,
    safe GmailAuthError telling the user to re-authenticate -- not a raw
    exception, and not silently falling back to a broken client."""

    class _RevokedCredentials(FakeCredentials):
        @classmethod
        def from_authorized_user_file(cls, path, scopes):
            return cls(valid=False, expired=True, refresh_token="rt", broken_refresh=True)

    _install_fake_google(monkeypatch, service=FakeGmailService())
    sys.modules["google.oauth2.credentials"].Credentials = _RevokedCredentials

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "token.json").write_text('{"token": "stale"}')

    with pytest.raises(GmailAuthError, match="re-authenticate") as excinfo:
        await manager.search("in:inbox", 5)

    # the user-facing message must not leak the raw provider error text
    assert "invalid_grant" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_corrupted_token_file_raises_gmail_auth_error(tmp_path, monkeypatch):
    _install_fake_google(monkeypatch, service=FakeGmailService())

    class _CorruptCredentials(FakeCredentials):
        @classmethod
        def from_authorized_user_file(cls, path, scopes):
            raise ValueError("bad json")

    sys.modules["google.oauth2.credentials"].Credentials = _CorruptCredentials

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "token.json").write_text("not json")

    with pytest.raises(GmailAuthError, match="corrupted or unreadable"):
        await manager.search("in:inbox", 5)


@pytest.mark.asyncio
async def test_search_returns_message_summaries(tmp_path, monkeypatch):
    store = {
        "m1": {
            "payload": {"headers": [{"name": "Subject", "value": "Hi"}, {"name": "From", "value": "a@b.com"}]},
            "snippet": "hello there",
        }
    }
    service = FakeGmailService(store=store)
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    results = await manager.search("in:inbox", 10)

    assert results == [
        {"id": "m1", "subject": "Hi", "from": "a@b.com", "date": "", "snippet": "hello there"}
    ]


@pytest.mark.asyncio
async def test_read_message_extracts_plain_text_body(tmp_path, monkeypatch):
    import base64

    encoded = base64.urlsafe_b64encode(b"the email body").decode("utf-8")
    store = {
        "m1": {
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": encoded},
                "headers": [
                    {"name": "Subject", "value": "Hi"},
                    {"name": "From", "value": "a@b.com"},
                    {"name": "To", "value": "me@example.com"},
                ],
            }
        }
    }
    service = FakeGmailService(store=store)
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    result = await manager.read_message("m1")

    assert result["body"] == "the email body"
    assert result["subject"] == "Hi"
    assert result["to"] == "me@example.com"


@pytest.mark.asyncio
async def test_read_nonexistent_message_raises_gmail_error(tmp_path, monkeypatch):
    _install_fake_google(monkeypatch, service=FakeGmailService(store={}))

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    with pytest.raises(GmailError, match="Could not read message"):
        await manager.read_message("missing")


@pytest.mark.asyncio
async def test_send_returns_sent_message_id(tmp_path, monkeypatch):
    service = FakeGmailService()
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    message_id = await manager.send("bob@example.com", "Subject", "Body text")

    assert message_id == "sent-123"


@pytest.mark.asyncio
async def test_send_api_failure_raises_gmail_error(tmp_path, monkeypatch):
    service = FakeGmailService(fail=True)
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    with pytest.raises(GmailError, match="Could not send email"):
        await manager.send("bob@example.com", "Subject", "Body text")


@pytest.mark.asyncio
async def test_create_draft_returns_draft_id(tmp_path, monkeypatch):
    service = FakeGmailService()
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    draft_id = await manager.create_draft("bob@example.com", "Subject", "Body text")

    assert draft_id == "draft-456"


@pytest.mark.asyncio
async def test_search_api_failure_raises_gmail_error_not_crash(tmp_path, monkeypatch):
    service = FakeGmailService(fail=True)
    _install_fake_google(monkeypatch, service=service)

    manager = GmailManager(
        client_secret_path=tmp_path / "secret.json", token_path=tmp_path / "token.json"
    )
    (tmp_path / "secret.json").write_text("{}")

    with pytest.raises(GmailError, match="Gmail search failed"):
        await manager.search("in:inbox", 5)
