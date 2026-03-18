from __future__ import annotations

import functools
import numbers

import numpy as np

import pyop3.expr
from pyop3.expr.visitors.base import OverloadedExpressionEvaluator


class ArraywiseEvaluator(OverloadedExpressionEvaluator):

    @functools.singledispatchmethod
    def process(self, obj: pyop3.expr.ExpressionT, /) -> pyop3.expr.ExpressionT:
        return super().process(obj)

    @process.register
    def _(self, scalar: pyop3.expr.Scalar, /) -> numbers.Number:
        return scalar.value

    @process.register
    def _(self, dat: pyop3.expr.Dat, /) -> np.ndarray:
        return dat.data_ro


def evaluate_arraywise(expr: pyop3.expr.ExpressionT) -> TODO:
    return ArraywiseEvaluator()(expr)
