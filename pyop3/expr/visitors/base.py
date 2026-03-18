from __future__ import annotations

import functools
import numbers
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.expr
import pyop3.node
from pyop3 import utils


class ExpressionVisitor(pyop3.node.NodeVisitor):

    @functools.singledispatchmethod
    def children(self, node, /):
        return super().children(node)

    @children.register(numbers.Number)
    def _(self, node, /):
        return idict()


class OverloadedExpressionEvaluator(ExpressionVisitor):
    """Mixin class defining handlers for commonly overloaded operations."""

    @functools.singledispatchmethod
    def process(self, obj: pyop3.expr.ExpressionT, /) -> Any:
        return super().process(obj)

    @process.register(numbers.Number)
    def _(self, num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
        return num

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.Add, visited, /, *args, **kwargs) -> Any:
        a, b = visited.values()
        return a + b

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.Sub, visited, /, *args, **kwargs) -> Any:
        a, b = visited.values()
        return a - b

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.Mul, visited, /, *args, **kwargs) -> Any:
        a, b = visited.values()
        return a * b

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.Modulo, visited, /, *args, **kwargs) -> Any:
        a, b = visited.values()
        return a % b

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.FloorDiv, visited, /, *args, **kwargs) -> Any:
        a, b = visited.values()
        return a // b

    @process.register
    @pyop3.node.postorder
    def _(self, _: pyop3.expr.Neg, visited, /, *args, **kwargs) -> Any:
        a = utils.just_one(visited.values())
        return -a
