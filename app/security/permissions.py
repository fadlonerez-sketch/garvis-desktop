"""Permission policy: independent of the LLM, cannot be overridden by it.

The agent asks `PermissionChecker.evaluate(tool)` before every tool call.
This is the single source of truth for whether a call may proceed
immediately or must go through a user confirmation first. The LLM only
ever sees the *outcome* (the tool ran, or was denied, or is pending
confirmation) -- it has no way to bypass this layer because tool
execution never happens except through it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.llm.base import Message
from app.security.egress import url_egress_reason
from app.tools.base import RiskLevel, Tool


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class PermissionResult(BaseModel):
    decision: PermissionDecision
    reason: str


class PermissionChecker:
    """Maps a tool's declared risk level to a permission decision.

    Policy (matches the product spec):
      LOW    -> always ALLOW.
      MEDIUM -> REQUIRE_CONFIRMATION only if the tool itself asks for it
                (`tool.requires_confirmation`); otherwise ALLOW.
      HIGH   -> always REQUIRE_CONFIRMATION. No tool can opt out of this.
    Plus, via evaluate_call(): any otherwise-ALLOWed call whose arguments the
    tool itself flags as consequential (Tool.confirmation_reason).
    """

    async def evaluate_call(
        self, tool: Tool, args: BaseModel, history: list[Message] | None = None
    ) -> PermissionResult:
        """`evaluate(tool)`, plus the tool's own per-call check for ALLOW
        decisions. The per-call check can only escalate ALLOW to
        REQUIRE_CONFIRMATION -- never relax a tool-level decision -- and if
        it fails, the call is confirmed rather than silently allowed.

        `history` (the conversation, for calls the model asked for) adds the
        egress taint rule: a URL the model composed after local data entered
        the conversation asks first (app/security/egress.py)."""
        decision = self.evaluate(tool)
        if decision.decision != PermissionDecision.ALLOW:
            return decision
        try:
            reason = await tool.confirmation_reason(args)
            if not reason and history is not None:
                url = tool.egress_url(args)
                reason = url_egress_reason(url, history) if url else None
        except Exception as exc:  # a broken check must fail closed
            reason = f"Garvis couldn't check whether this '{tool.name}' call is safe ({exc})."
        if reason:
            return PermissionResult(decision=PermissionDecision.REQUIRE_CONFIRMATION, reason=reason)
        return decision

    def evaluate(self, tool: Tool) -> PermissionResult:
        if tool.risk_level == RiskLevel.HIGH:
            return PermissionResult(
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                reason=f"'{tool.name}' is high-risk and always requires confirmation.",
            )
        if tool.risk_level == RiskLevel.MEDIUM and tool.requires_confirmation:
            return PermissionResult(
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                reason=f"'{tool.name}' is configured to require confirmation.",
            )
        return PermissionResult(decision=PermissionDecision.ALLOW, reason="Low-risk or pre-approved.")
