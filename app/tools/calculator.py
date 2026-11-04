"""Calculator tool: evaluate arithmetic expressions safely.

Uses Python's `ast` module to evaluate a restricted grammar (numbers and
+ - * / // % ** and parentheses) instead of `eval`, so the LLM cannot use
this tool to execute arbitrary code.
"""
from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from app.tools.base import RiskLevel, Tool, ToolResult

_MAX_EXPONENT = 10_000
_MAX_INT_BITS = 100_000  # ~30,000 digits: plenty for arithmetic, far from freezing the agent's event loop


class CalculatorError(ValueError):
    pass


def _bounded_pow(base, exponent):
    if abs(base) not in (0, 1):
        if abs(exponent) > _MAX_EXPONENT:
            raise CalculatorError(f"Exponent too large (limit {_MAX_EXPONENT}).")
        if isinstance(base, int) and isinstance(exponent, int) and exponent * base.bit_length() > _MAX_INT_BITS:
            raise CalculatorError("Result too large.")
    return operator.pow(base, exponent)


def _bounded_mul(left, right):
    if isinstance(left, int) and isinstance(right, int) and left.bit_length() + right.bit_length() > _MAX_INT_BITS:
        raise CalculatorError("Result too large.")
    return operator.mul(left, right)


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _bounded_mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _bounded_pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {expression!r}") from exc
    return _eval_node(tree.body)


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(2 + 3) * 4 / 2'")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **, parentheses). "
        "Use this for any math instead of computing it yourself."
    )
    risk_level = RiskLevel.LOW
    input_model = CalculatorInput

    async def execute(self, args: CalculatorInput) -> ToolResult:
        try:
            result = safe_eval(args.expression)
        except CalculatorError as exc:
            return ToolResult.fail(str(exc))
        except ZeroDivisionError:
            return ToolResult.fail("Division by zero")
        except (OverflowError, ValueError):  # float overflow; int too large to convert
            return ToolResult.fail("Result too large")
        return ToolResult.ok(result)
