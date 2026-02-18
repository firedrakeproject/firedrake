from __future__ import annotations

import functools
import itertools
import numbers
from types import NoneType
from typing import Any, Hashable

from immutabledict import immutabledict as idict

import pyop3.tree.axis_tree as op3_tree
from pyop3 import utils
from pyop3.cache import memory_cache
from pyop3.node import Visitor, LabelledTreeVisitor
from pyop3.tree.labelled_tree import parent_path
from pyop3.utils import OrderedFrozenSet
from pyop3.expr.visitors import BufferCollector as ExprBufferCollector, DiskCacheKeyGetter as ExprDiskCacheKeyGetter

from .layout import compute_layouts  # noqa: F401
from .size import compute_axis_tree_size, compute_axis_tree_component_size  # noqa: F401


class DiskCacheKeyGetter(LabelledTreeVisitor):

    EMPTY = None

    def __init__(self, renamer=None, expr_getter=None):
        from pyop3.insn.visitors import Renamer
        if renamer is None:
            renamer = Renamer()
        self._renamer = renamer
        self._lazy_expr_getter = expr_getter
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Any, path: ConcretePathT, /) -> Hashable:
        return super().process(obj)

    @process.register(op3_tree.Axis)
    @LabelledTreeVisitor.postorder
    def _(self, axis: op3_tree.Axis, path: ConcretePathT, /, visited) -> Hashable:
        new_label = self._renamer.add(axis)
        key = [type(axis), new_label]
        for component in axis.components:
            component_key = get_disk_cache_key(component, renamer=self._renamer)
            key.append(component_key)
        return (tuple(key), visited)

    # FIXME: Maybe not needed any more
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, path: ConcretePathT, /) -> Hashable:
        assert not path, "Must be at tree root"
        return None

    def _get_expr_disk_cache_key(self, expr: ExpressionT) -> Hashable:
        if self._lazy_expr_getter is None:
            self._lazy_expr_getter = ExprDiskCacheKeyGetter(self._renamer, self)

        return self._lazy_expr_getter._safe_call(expr)


@functools.singledispatch
def get_disk_cache_key(axis_tree: op3_tree.AxisTree, renamer=None) -> Hashable:
    return DiskCacheKeyGetter(renamer)(axis_tree)


@get_disk_cache_key.register(op3_tree.AxisComponent)
def _(component: op3_tree.AxisComponent, renamer=None) -> tuple:
    if renamer is None:
        renamer = Renamer()
    expr_renamer = ExprDiskCacheKeyGetter(renamer)
    return (component.label, expr_renamer(component.size))


@get_disk_cache_key.register(op3_tree.AxisComponentRegion)
def _(component: op3_tree.AxisComponent, renamer) -> tuple:
    expr_renamer = ExprDiskCacheKeyGetter(renamer)
    return (component.label, expr_renamer(component.size))


class BufferCollector(LabelledTreeVisitor):

    EMPTY = OrderedFrozenSet()

    def __init__(self, expr_collector: ExprBufferCollector | None = None) -> None:
        self._lazy_expr_collector = expr_collector
        super().__init__()

    def __call__(self, tree):
        return super().__call__(tree) | self._collect_expr_buffers(tree.size)

    @classmethod
    @memory_cache(heavy=True)
    def maybe_singleton(cls, comm) -> Self:
        return cls()

    @functools.singledispatchmethod
    def process(self, obj: Any, /, path: ConcretePathT) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(op3_tree.Axis)
    @LabelledTreeVisitor.postorder
    def _(self, axis: op3_tree.Axis, /, path: ConcretePathT, visited: tuple[OrderedFrozenSet, ...]) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(
            *(self._collect_expr_buffers(c.size) for c in axis.components),
            *visited,
        )

    # TODO: is this necessary now that we have EMPTY?
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, /, path: ConcretePathT) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    def _collect_expr_buffers(self, expr) -> OrderedFrozenSet:
        if self._lazy_expr_collector is None:
            self._lazy_expr_collector = ExprBufferCollector(self)

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
