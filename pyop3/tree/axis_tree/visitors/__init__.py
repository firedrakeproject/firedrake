from __future__ import annotations

import functools
import itertools
from types import NoneType
from typing import Any, Hashable

from immutabledict import immutabledict as idict

import pyop3.tree.axis_tree as op3_tree
from pyop3 import utils
from pyop3.node import Visitor, LabelledTreeVisitor
from pyop3.utils import OrderedFrozenSet
from pyop3.expr.visitors import BufferCollector as ExprBufferCollector, get_disk_cache_key as get_expr_disk_cache_key

from .layout import compute_layouts  # noqa: F401
from .size import compute_axis_tree_size, compute_axis_tree_component_size  # noqa: F401


class _DiskCacheKeyGetter(LabelledTreeVisitor):

    EMPTY = None

    def __init__(self, renamer=None):
        if renamer is None:
            renamer = Renamer()
        self._renamer = renamer
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
            component_key = (component.label, get_expr_disk_cache_key(component.size, self._renamer))
            key.append(component_key)
        return (tuple(key), visited)

    # FIXME: Maybe not needed any more
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, path: ConcretePathT, /) -> Hashable:
        assert not path, "Must be at tree root"
        return None

def get_disk_cache_key(axis_tree: op3_tree.AxisTree, renamer=None) -> Hashable:
    return _DiskCacheKeyGetter(renamer)(axis_tree)


class BufferCollector(LabelledTreeVisitor):

    EMPTY = OrderedFrozenSet()

    def __init__(self, expr_collector: ExprBufferCollector | None = None, *, _internal: bool = True) -> None:
        if expr_collector is None:
            expr_collector = ExprBufferCollector(self)
        self.expr_collector = expr_collector
        super().__init__()

    def __call__(self, axis_tree: op3_tree.AbstractAxisTree):
        target_buffers = OrderedFrozenSet().union(
            *(
                self._maybe_collect_expr(target.expr)
                for targets in itertools.chain(*axis_tree.targets.values())
                for target in targets
            )
        )
        return super().__call__(axis_tree) | target_buffers

    @functools.singledispatchmethod
    def process(self, obj: Any, /, path: ConcretePathT) -> OrderedSet:
        return super().process(obj)

    @process.register(op3_tree.Axis)
    @LabelledTreeVisitor.postorder
    def _(self, axis: op3_tree.Axis, /, path: ConcretePathT, visited: OrderedFrozenSet) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(
            *(self._maybe_collect_expr(c.size) for c in axis.components),
            *visited,
        )

    # TODO: is this necessary now that we have EMPTY?
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, /, path: ConcretePathT) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    def _maybe_collect_expr(self, expr):
        # trick to stop recursing
        if self.expr_collector.get_cache_key(expr) in self.expr_collector._seen_keys:
            return OrderedFrozenSet()
        else:
            return self.expr_collector(expr)


def collect_buffers(axis_tree: AbstractAxisTree) -> OrderedFrozenSet:
    return BufferCollector()(axis_tree)
