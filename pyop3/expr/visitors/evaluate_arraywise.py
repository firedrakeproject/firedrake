from __future__ import annotations

import functools
import numbers

import numpy as np

import pyop3.expr
from pyop3.expr.visitors.base import OverloadedExpressionEvaluator


class ArraywiseEvaluator(OverloadedExpressionEvaluator):

    def __init__(self, *, include_ghosts: bool = False) -> None:
        self._include_ghosts = include_ghosts
        # Pass reuse_results=False because numpy arrays are not hashable
        super().__init__(reuse_results=False)

    @functools.singledispatchmethod
    def process(self, obj: pyop3.expr.ExpressionT, /) -> pyop3.expr.ExpressionT:
        return super().process(obj)

    @process.register
    def _(self, scalar: pyop3.expr.Scalar, /) -> numbers.Number:
        return scalar.value

    @process.register
    def _(self, dat: pyop3.expr.Dat, /) -> np.ndarray:
        return dat.data_ro_with_halos if self._include_ghosts else dat.data_ro


def evaluate_arraywise(expr: pyop3.expr.ExpressionT, *, include_ghosts: bool = False):
    """Evaluate an expression with NumPy operations on the arrays of its Dats.

    Parameters
    ----------
    expr
        The expression to evaluate.
    include_ghosts
        If `True`, read each Dat through its view with ghosts, which also
        includes the constrained entries. If `False`, read only the owned
        and unconstrained entries.

    """
    return ArraywiseEvaluator(include_ghosts=include_ghosts)(expr)
