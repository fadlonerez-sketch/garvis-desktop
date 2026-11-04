"""Tools exposing persistent memory to the LLM."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.memory import MemoryStore
from app.tools.base import RiskLevel, Tool, ToolResult


class RememberFactInput(BaseModel):
    content: str = Field(description="The fact or preference to remember, in plain language")


class RememberFactTool(Tool):
    name = "remember_fact"
    description = "Save a fact or preference the user explicitly asked Garvis to remember."
    risk_level = RiskLevel.LOW
    input_model = RememberFactInput

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute(self, args: RememberFactInput) -> ToolResult:
        item = await self._store.remember(args.content)
        return ToolResult.ok({"id": item.id, "content": item.content})


class RecallFactsInput(BaseModel):
    query: str | None = Field(default=None, description="Optional keyword to filter remembered facts")


class RecallFactsTool(Tool):
    name = "recall_facts"
    description = "Retrieve previously remembered facts, optionally filtered by keyword."
    risk_level = RiskLevel.LOW
    input_model = RecallFactsInput

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute(self, args: RecallFactsInput) -> ToolResult:
        items = await self._store.recall(args.query)
        return ToolResult.ok([{"id": i.id, "content": i.content} for i in items])
