from __future__ import annotations

import functools
import itertools
import numbers
from types import NoneType
from typing import Any, Hashable

from immutabledict import immutabledict as idict

import pyop3.axis_tree
from pyop3 import utils
from pyop3.collections import OrderedFrozenSet
from pyop3.cache import memory_cache
from pyop3.node import Visitor, LabelledTreeVisitor, postorder
from pyop3.labeled_tree import parent_path

from .layout import compute_layouts  # noqa: F401
from .size import compute_axis_tree_size, compute_axis_tree_component_size  # noqa: F401


# used?
class BufferCollector(LabelledTreeVisitor):

    EMPTY = OrderedFrozenSet()

    def __init__(self, expr_collector: ExprBufferCollector | None = None, *, shallow: bool = False) -> None:
        self._lazy_expr_collector = expr_collector
        self.shallow = shallow
        super().__init__()

    def __call__(self, tree):
        return super().__call__(tree) | self._collect_expr_buffers(tree.size)

    @classmethod
    # @memory_cache(heavy=True)
    def maybe_singleton(cls, comm) -> Self:
        return cls()

    @functools.singledispatchmethod
    def process(self, obj: Any, /, path: ConcretePathT) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(pyop3.axis_tree.Axis)
    @postorder
    def _(self, axis: pyop3.axis_tree.Axis, /, path: ConcretePathT, visited: tuple[OrderedFrozenSet, ...]) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(
            *(self._collect_expr_buffers(c.size) for c in axis.components),
            *visited,
        )

    # TODO: is this necessary now that we have EMPTY?
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, /, path: ConcretePathT) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    def _collect_expr_buffers(self, expr) -> OrderedFrozenSet:
        from pyop3.expr.visitors import BufferCollector as ExprBufferCollector

        if self._lazy_expr_collector is None:
            self._lazy_expr_collector = ExprBufferCollector(self, shallow=True)

        return self._lazy_expr_collector._safe_call(expr, OrderedFrozenSet())


def collect_buffers(axis_tree: AbstractAxisTree) -> OrderedFrozenSet:
    return BufferCollector()(axis_tree)


def get_block_shape(axis_tree: AbstractAxisTree) -> tuple[int, ...]:
    """Detect any common innermost integer shape in an axis tree."""
    axis_tree = axis_tree.materialize()

    block_shape = []
    while not axis_tree.is_empty:
        if not utils.is_single_valued(axis_tree.leaves):
            break
        leaf_axis = utils.single_valued(axis_tree.leaves)

        if not isinstance(leaf_axis.size, numbers.Integral):
            break
        block_shape.insert(0, leaf_axis.size)

        for leaf_path in axis_tree.leaf_paths:
            axis_tree = axis_tree.drop_node(parent_path(leaf_path))
    return tuple(block_shape)
