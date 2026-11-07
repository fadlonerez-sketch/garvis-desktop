"""Local date/time tool."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.tools.base import RiskLevel, Tool, ToolResult


class DateTimeInput(BaseModel):
    pass


class DateTimeTool(Tool):
    name = "get_current_datetime"
    description = "Return the current local date and time on the user's Windows machine."
    risk_level = RiskLevel.LOW
    input_model = DateTimeInput

    async def execute(self, args: DateTimeInput) -> ToolResult:
        now = datetime.now().astimezone()
        return ToolResult.ok(
            {
                "iso": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "timezone": now.tzname(),
            }
        )
