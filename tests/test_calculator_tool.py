import pytest
from pydantic import ValidationError

from app.tools.calculator import CalculatorInput, CalculatorTool


@pytest.mark.asyncio
async def test_calculator_basic_expression():
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "(2 + 3) * 4"})
    result = await tool.execute(args)
    assert result.success
    assert result.data == 20


@pytest.mark.asyncio
async def test_calculator_division_by_zero():
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "1 / 0"})
    result = await tool.execute(args)
    assert not result.success
    assert "zero" in result.error.lower()


@pytest.mark.asyncio
async def test_calculator_rejects_non_arithmetic():
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "__import__('os').system('echo hi')"})
    result = await tool.execute(args)
    assert not result.success


def test_calculator_input_schema_validation():
    with pytest.raises(ValidationError):
        CalculatorInput.model_validate({})
