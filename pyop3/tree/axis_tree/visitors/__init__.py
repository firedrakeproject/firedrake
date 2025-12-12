from __future__ import annotations

import functools
import itertools

from immutabledict import immutabledict as idict

import pyop3.tree.axis_tree as op3_tree
from pyop3 import utils
from pyop3.node import Visitor, LabelledTreeVisitor
from pyop3.utils import OrderedFrozenSet
from pyop3.expr.visitors import BufferCollector as ExprBufferCollector

from .layout import compute_layouts  # noqa: F401
from .size import compute_axis_tree_size, compute_axis_tree_component_size  # noqa: F401


# TODO: implement a TreeVisitor class, similar to the other visitor but cannot do certain caches
# class _DiskCacheKeyGetter(Visitor):
#     def __init__(self, renamer=None):
#         if renamer is None:
#             renamer = Renamer()
#         self._renamer = renamer
#         super().__init__()


def get_disk_cache_key(axis_tree: op3_tree.AxisTree, renamer=None) -> Hashable:
    if renamer is None:
        raise NotImplementedError
    # return _DiskCacheKeyGetter(renamer)(axis_tree)
    return _get_disk_cache_key(axis_tree, renamer, idict())


def _get_disk_cache_key(axis_tree, renamer, path):
    axis = axis_tree.node_map[path]

    key = [get_axis_key(axis, renamer)]

    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if axis_tree.node_map[path_]:
            key.append(_get_disk_cache_key(axis_tree, renamer, path_))
        else:
            key.append(())

    return tuple(key)


def get_axis_key(axis, renamer):
    from pyop3.expr.visitors import get_disk_cache_key as get_expr_disk_cache_key
    # FIXME: We assume that axes are already relabelled in the tree
    key = [type(axis), axis.label]
    for component in axis.components:
        component_key = (component.label, get_expr_disk_cache_key(component.size, renamer))
        key.append(component_key)
    return tuple(key)


class BufferCollector(LabelledTreeVisitor):

    def __init__(self, expr_collector: ExprBufferCollector | None = None) -> None:
        if expr_collector is None:
            expr_collector = ExprBufferCollector(self)
        self.expr_collector = expr_collector
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Instruction) -> OrderedSet:
        return super().process(obj)

    @process.register(op3_tree.Axis)
    @LabelledTreeVisitor.postorder
    def _(self, axis: op3_tree.Axis, path, /, *visited) -> OrderedFrozenSet:
        buffers = OrderedFrozenSet().union(
            *(self.expr_collector(c.size) for c in axis.components),
        )
        return buffers.union(*visited)


# def collect_buffers(axis_tree: AbstractAxisTree) -> OrderedFrozenSet:
#     return _collect_buffers(axis_tree, idict())
#
#
# def _collect_buffers(axis_tree, path):
#     axis = axis_tree.node_map[path]
#
#     key = [get_axis_key(axis, renamer)]
#
#     for component in axis.components:
#         path_ = path | {axis.label: component.label}
#
#         if axis_tree.node_map[path_]:
#             key.append(_get_disk_cache_key(axis_tree, renamer, path_))
#         else:
#             key.append(())
#
#     return tuple(key)
