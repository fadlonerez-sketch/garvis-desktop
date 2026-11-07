"""Central registry of tools available to the agent.

The agent discovers tools only through this registry. There is no path
from the LLM to arbitrary Python execution: a tool call names a registered
tool, its arguments are validated against that tool's Pydantic schema, and
only then is `execute()` invoked.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import ToolNotFoundError
from app.llm.base import ToolSpec
from app.tools.base import Tool

_PROPERTY_MAPS = ("properties", "$defs", "definitions", "patternProperties")


def compact_schema(node: Any, _in_property_map: bool = False) -> Any:
    """Drop pydantic's auto-generated `title` annotations (e.g. "title":
    "Expression" next to the property named "expression"). They repeat the
    property/model names, carry no meaning for the model, and are sent with
    every request. A *property* named "title" is kept."""
    if isinstance(node, dict):
        return {
            key: compact_schema(value, _in_property_map=key in _PROPERTY_MAPS and not _in_property_map)
            for key, value in node.items()
            if _in_property_map or not (key == "title" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [compact_schema(item) for item in node]
    return node


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"No tool named '{name}' is registered") from exc

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool_specs(self) -> list[ToolSpec]:
        """Tool descriptions in the shape the LLM provider expects."""
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=compact_schema(tool.parameters_schema()),
            )
            for tool in self._tools.values()
        ]
