import pytest

from app.core.exceptions import ToolNotFoundError
from app.tools.calculator import CalculatorTool
from app.tools.datetime_tool import DateTimeTool
from app.tools.registry import ToolRegistry


def test_register_and_get():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    tool = registry.get("calculator")
    assert tool.name == "calculator"


def test_get_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("does_not_exist")


def test_duplicate_registration_raises():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    with pytest.raises(ValueError):
        registry.register(CalculatorTool())


def test_tool_specs_includes_schema():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())

    specs = registry.tool_specs()
    names = {s.name for s in specs}
    assert names == {"calculator", "get_current_datetime"}
    calc_spec = next(s for s in specs if s.name == "calculator")
    assert "expression" in calc_spec.parameters["properties"]
