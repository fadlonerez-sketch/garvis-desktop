"""Standard tool interface.

Every capability Garvis exposes to the LLM -- calculator, web search,
opening apps, Gmail, browser control -- implements this interface and is
looked up through the ToolRegistry. The LLM never executes arbitrary code;
it can only request a *named, registered* tool with *schema-validated*
arguments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from pydantic import BaseModel


class RiskLevel(IntEnum):
    LOW = 0
    """No confirmation required: answering, searching, reading, calculating."""
    MEDIUM = 1
    """May require confirmation depending on policy: writing files, authenticated
    site navigation."""
    HIGH = 2
    """Always requires explicit user confirmation: sending mail, deleting files,
    purchases, irreversible actions."""


class ToolResult(BaseModel):
    success: bool
    data: dict | list | str | int | float | bool | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: dict | list | str | int | float | bool | None = None) -> "ToolResult":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)


class Tool(ABC):
    """Base class for a registered Garvis tool.

    Subclasses declare `name`, `description`, `risk_level`, and a Pydantic
    `input_model` used to validate LLM-provided arguments before execution.
    """

    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    input_model: type[BaseModel]
    presentation: str = "none"
    """Whether a successful call shows the user something worth presenting
    on the desktop: "none" (background work, the default), "subtle" or
    "prominent" -- see app/agent/presentation.py."""
    requires_confirmation: bool = False
    """Explicit override. If False, confirmation is still enforced by the
    security layer for HIGH risk tools regardless of this flag."""

    async def confirmation_reason(self, args: BaseModel) -> str | None:
        """Per-call escalation: return a reason when THIS particular call
        (e.g. clicking a "Send" button, a PowerShell command that changes
        state) needs the user's confirmation even though the tool as a whole
        doesn't. Decided by deterministic code, never by the LLM. It can only
        add a confirmation -- see PermissionChecker.evaluate_call()."""
        return None

    def parameters_schema(self) -> dict:
        return self.input_model.model_json_schema()

    def validate_arguments(self, raw_arguments: dict) -> BaseModel:
        return self.input_model.model_validate(raw_arguments)

    @abstractmethod
    async def execute(self, args: BaseModel) -> ToolResult:
        """Perform the tool's action. Must not raise for expected failures --
        return ToolResult.fail(...) instead so the agent can react."""
