from pydantic import BaseModel

from app.security.permissions import PermissionChecker, PermissionDecision
from app.tools.base import RiskLevel, Tool, ToolResult


class _Args(BaseModel):
    pass


class _LowRiskTool(Tool):
    name = "low"
    description = "low risk"
    risk_level = RiskLevel.LOW
    input_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.ok()


class _MediumRiskNoConfirmTool(Tool):
    name = "medium_no_confirm"
    description = "medium risk, no confirm required"
    risk_level = RiskLevel.MEDIUM
    requires_confirmation = False
    input_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.ok()


class _MediumRiskConfirmTool(Tool):
    name = "medium_confirm"
    description = "medium risk, confirm required"
    risk_level = RiskLevel.MEDIUM
    requires_confirmation = True
    input_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.ok()


class _HighRiskTool(Tool):
    name = "high"
    description = "high risk"
    risk_level = RiskLevel.HIGH
    input_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.ok()


def test_low_risk_allowed():
    result = PermissionChecker().evaluate(_LowRiskTool())
    assert result.decision == PermissionDecision.ALLOW


def test_medium_risk_without_confirmation_flag_allowed():
    result = PermissionChecker().evaluate(_MediumRiskNoConfirmTool())
    assert result.decision == PermissionDecision.ALLOW


def test_medium_risk_with_confirmation_flag_requires_confirmation():
    result = PermissionChecker().evaluate(_MediumRiskConfirmTool())
    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION


def test_high_risk_always_requires_confirmation_even_if_flag_unset():
    tool = _HighRiskTool()
    assert tool.requires_confirmation is False  # default, unset
    result = PermissionChecker().evaluate(tool)
    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION
