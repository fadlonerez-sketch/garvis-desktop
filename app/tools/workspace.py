"""switch_workspace: move the user's VIEW between their desktop and Garvis's.

Garvis's own browser work happens on the Garvis virtual desktop without
ever switching the view; this tool is for when the user wants to *watch*
(or when desktop mouse/keyboard work must happen there -- input only
reaches the desktop currently shown). LOW risk: it changes what's on
screen, not any data, and "primary" always brings the user straight back.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import DesktopError
from app.desktop.workspace import WorkspaceManager
from app.tools.base import RiskLevel, Tool, ToolResult


class SwitchWorkspaceInput(BaseModel):
    target: Literal["garvis", "primary", "status"] = Field(
        description="'garvis' shows Garvis's own desktop, 'primary' returns to the user's desktop, "
        "'status' just reports which desktop is showing."
    )


class SwitchWorkspaceTool(Tool):
    name = "switch_workspace"
    description = (
        "Show Garvis's separate virtual desktop (where its research browser runs) or return to "
        "the user's own desktop. Mouse/keyboard tools act on whichever desktop is showing."
    )
    risk_level = RiskLevel.LOW
    input_model = SwitchWorkspaceInput

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def execute(self, args: SwitchWorkspaceInput) -> ToolResult:
        try:
            if args.target == "garvis":
                await self._workspace.show_garvis()
            elif args.target == "primary":
                await self._workspace.show_primary()
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        status = await self._workspace.status()
        if not status.get("available"):
            return ToolResult.fail(f"The Garvis workspace is unavailable: {status.get('reason')}")
        return ToolResult.ok(status)
