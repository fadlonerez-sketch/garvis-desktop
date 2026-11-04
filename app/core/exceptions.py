"""Exception hierarchy for Garvis."""
from __future__ import annotations


class GarvisError(Exception):
    """Base class for all Garvis errors."""


class ConfigurationError(GarvisError):
    """Raised when configuration is missing or invalid."""


class LLMError(GarvisError):
    """Raised when the LLM provider fails or is unreachable."""


class LLMUnavailableError(LLMError):
    """Raised when an LLM backend (e.g. OmniRoute, Groq) cannot be reached, times
    out, or otherwise cannot currently serve requests."""


class LLMTimeoutError(LLMUnavailableError):
    """The provider didn't answer in time. The request may still have been
    processed (and billed), so it is never resent to the same model."""




class LLMAuthenticationError(LLMError):
    """Raised when an LLM provider rejects our credentials (e.g. bad/missing
    API key)."""


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider throttles us (HTTP 429 or equivalent)."""


class LLMBillingError(LLMError):
    """The provider account can't pay for requests (e.g. Anthropic's "credit
    balance is too low"). Retrying can't help, so it is reported at once."""


class LLMBadRequestError(LLMError):
    """Raised when the provider rejects the request itself (HTTP 400/413/422):
    resending the same request can't succeed, so it is never retried."""


class LLMBilledError(LLMError):
    """A request the provider answered (and billed) whose answer Garvis can't
    use. The usage travels with the error so it is still accounted for, and
    the request is neither retried nor handed to another model."""

    def __init__(self, message: str, usage=None, served_model: str | None = None,
                 request_id: str | None = None) -> None:
        super().__init__(message)
        self.usage = usage
        self.served_model = served_model
        self.request_id = request_id


class LLMModelMismatchError(LLMBilledError):
    """Raised when a response says it was served by a different model than the
    one Garvis requested -- Garvis never accepts a silent model switch."""


class LLMRefusalError(LLMBilledError):
    """The model declined the request on safety grounds (Claude
    `stop_reason: "refusal"`). Never routed to another model to get around it."""


class LLMBudgetExceededError(LLMError):
    """Raised *before* a request is sent when it would break a local spending
    limit (app/llm/usage.py:BudgetGuard). `limit` names the limit."""

    def __init__(self, message: str, limit: str = "") -> None:
        super().__init__(message)
        self.limit = limit


class VoiceError(GarvisError):
    """Base class for STT/TTS errors."""


class SpeechToTextError(VoiceError):
    """Raised when speech-to-text fails (e.g. microphone unavailable)."""


class TextToSpeechError(VoiceError):
    """Raised when text-to-speech fails."""


class ToolError(GarvisError):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when the agent requests a tool that is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool input arguments fail schema validation."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""


class WindowPlacementError(GarvisError):
    """A window could not be put on the desktop it was meant for (the user's
    desktop or Garvis's workspace). Never answered by opening it elsewhere."""


class BrowserError(GarvisError):
    """Raised for any browser-automation failure: launch, navigation, or
    page interaction (click/type/press key)."""


class DesktopError(GarvisError):
    """Raised for any desktop-control failure: an unsupported platform, an
    off-screen coordinate, an unknown/blocked key, a window that can't be
    found or focused, or the OS refusing injected input."""


class WorkspaceError(DesktopError):
    """Raised when the dedicated Garvis workspace (virtual desktop) can't be
    created, found, or switched to."""


class GmailError(GarvisError):
    """Raised for any Gmail-API failure: request errors, malformed
    responses, or missing configuration."""


class GmailAuthError(GmailError):
    """Raised when Gmail OAuth credentials are missing, expired, or revoked
    and cannot be silently refreshed -- the user must re-authenticate."""


class PermissionDeniedError(GarvisError):
    """Raised when an action is blocked by the permission system."""


class ConfirmationRequiredError(GarvisError):
    """Raised internally to signal a tool call is pending user confirmation."""

    def __init__(self, confirmation_id: str, message: str = "Confirmation required"):
        super().__init__(message)
        self.confirmation_id = confirmation_id


class ConfirmationDeniedError(GarvisError):
    """Raised when the user declines a pending confirmation request."""


class ConfirmationTimeoutError(GarvisError):
    """Raised when a confirmation request is not answered in time."""
