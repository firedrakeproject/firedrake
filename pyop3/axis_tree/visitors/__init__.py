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

    # @process.register(pyop3.axis_tree.Axis)
    # @postorder
    # def _(self, axis: pyop3.axis_tree.Axis, path: ConcretePathT, /, visited) -> Hashable:
    #     new_label = self._renamer.add(axis)
    #     key = [type(axis), new_label]
    #     for component in axis.components:
    #         component_key = get_disk_cache_key(component, renamer=self._renamer)
    #         key.append(component_key)
    #     return (tuple(key), visited)

    # FIXME: Maybe not needed any more
    @process.register(NoneType)  # empty/unit tree
    def _(self, none: None, path: ConcretePathT, /) -> Hashable:
        assert not path, "Must be at tree root"
        return None

    def _get_expr_disk_cache_key(self, expr: ExpressionT) -> Hashable:
        from pyop3.expr.visitors import DiskCacheKeyGetter as ExprDiskCacheKeyGetter

        if self._lazy_expr_getter is None:
            self._lazy_expr_getter = ExprDiskCacheKeyGetter(self._renamer, self)

        return self._lazy_expr_getter._safe_call(expr)


# @functools.singledispatch
# def get_disk_cache_key(axis_tree: pyop3.axis_tree.AxisTree, renamer=None) -> Hashable:
#     return DiskCacheKeyGetter(renamer)(axis_tree)


# @get_disk_cache_key.register(pyop3.axis_tree.AxisComponent)
# def _(component: pyop3.axis_tree.AxisComponent, renamer=None) -> tuple:
#     if renamer is None:
#         renamer = Renamer()
#     return component.disk_cache_key(renamer)
    # from pyop3.expr.visitors import DiskCacheKeyGetter as ExprDiskCacheKeyGetter
    # expr_renamer = ExprDiskCacheKeyGetter(renamer)
    # return (component.label, expr_renamer(component.size))


# @get_disk_cache_key.register(pyop3.axis_tree.AxisComponentRegion)
# def _(component: pyop3.axis_tree.AxisComponent, renamer) -> tuple:
#     from pyop3.expr.visitors import DiskCacheKeyGetter as ExprDiskCacheKeyGetter
#     expr_renamer = ExprDiskCacheKeyGetter(renamer)
#     return (component.label, expr_renamer(component.size))


class BufferCollector(LabelledTreeVisitor):

    EMPTY = OrderedFrozenSet()

    def __init__(self, expr_collector: ExprBufferCollector | None = None, *, shallow: bool = False) -> None:
        self._lazy_expr_collector = expr_collector
        self.shallow = shallow
        super().__init__()

    def __call__(self, tree):
        result = super().__call__(tree) | self._collect_expr_buffers(tree.size)
        if "array_54" in str(result):
            breakpoint()
        return result

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


# class LabelCanonicalizer(LabelledTreeVisitor):
#
#     EMPTY = None
#
#     def __init__(self, relabeler):
#         self._relabeler = relabeler
#         super().__init__()
#
#     @functools.singledispatchmethod
#     def process(self, obj: Any, path: ConcretePathT, /) -> Hashable:
#         return super().process(obj)
#
#     @process.register(pyop3.axis_tree.Axis)
#     def _(self, axis: pyop3.axis_tree.Axis, path: ConcretePathT) -> Hashable:
#         relabeled_axis = canonicalize_labels(axis, self._relabeler)
#         node_map = {idict(): relabeled_axis}
#         for component in relabeled_axis.components:
#             path_ = path | idict({axis.label: component.label})
#             relabeled_path = idict({relabeled_axis.label: component.label})
#             if self._tree.node_map[path_]:
#                 subnode_map = self._call(path_)
#                 for subpath, subaxis in subnode_map.items():
#                     node_map[relabeled_path | subpath] = subaxis
#             else:
#                 node_map[relabeled_path] = None
#         return idict(node_map)
#
#
# @functools.singledispatch
# def canonicalize_labels(axis_tree: pyop3.axis_tree.AxisTree, relabeler: Renamer) -> AxisTree:
#     raise TypeError
#
# @canonicalize_labels.register(pyop3.axis_tree.AxisTree)
# def _(axis_tree: pyop3.axis_tree.AxisTree, relabeler: Renamer) -> AxisTree:
#     node_map = LabelCanonicalizer(relabeler)(axis_tree)
#     return axis_tree.__record_init__(_node_map=node_map)
#
# @canonicalize_labels.register(pyop3.axis_tree.IndexedAxisTree)
# def _(axes: pyop3.axis_tree.IndexedAxisTree, relabeler):
#     node_map = LabelCanonicalizer(relabeler)(axes)
#     unindexed = canonicalize_labels(axes.unindexed, relabeler)
#     targets = _canonicalize_target_labels(axes.targets, relabeler)
#     return axes.__record_init__(_node_map=node_map, _unindexed=unindexed, _targets=targets)
#
# @canonicalize_labels.register(pyop3.axis_tree._UnitAxisTree)
# def _(axes: pyop3.axis_tree.UnitIndexedAxisTree, relabeler):
#     return axes
#
# @canonicalize_labels.register(pyop3.axis_tree.AxisForest)
# def _(axes: pyop3.axis_tree.UnitIndexedAxisTree, relabeler):
#     return type(axes)([canonicalize_labels(t, relabeler) for t in axes.trees])
#
# @canonicalize_labels.register(pyop3.axis_tree.UnitIndexedAxisTree)
# def _(axes: pyop3.axis_tree.UnitIndexedAxisTree, relabeler):
#     unindexed = canonicalize_labels(axes.unindexed, relabeler)
#     targets = _canonicalize_target_labels(axes.targets, relabeler)
#     return axes.__record_init__(unindexed=unindexed, _targets=targets)
#
#
# @canonicalize_labels.register(pyop3.axis_tree.ContextSensitiveAxisTree)
# def _(axes: pyop3.axis_tree.ContextSensitiveAxisTree, relabeler):
#     relabeled_trees = {}
#     for ctx, tree in axes.trees.items():
#         relabeled_ctx = {}
#         for loop_id, path in ctx.items():
#             relabeled_loop_id = relabeler.add(loop_id, "loop")
#             relabeled_path = idict({
#                 relabeler.add(axis, "axis"): component
#                 for axis, component in path.items()
#             })
#             relabeled_ctx[relabeled_loop_id] = relabeled_path
#         relabeled_ctx = idict(relabeled_ctx)
#
#         relabeled_tree = canonicalize_labels(tree, relabeler)
#         relabeled_trees[relabeled_ctx] = relabeled_tree
#     relabeled_trees = idict(relabeled_trees)
#     return axes.__record_init__(trees=relabeled_trees)
#
#
# def _canonicalize_target_labels(targets, relabeler):
#     from pyop3.expr.visitors import canonicalize_labels as relabel_expr
#
#     relabeled_targets = {}
#     for path, axis_targetss in targets.items():
#         relabeled_path = idict({
#             relabeler[axis_label]: component_label
#             for axis_label, component_label in path.items()
#         })
#         relabeled_axis_targetss = []
#         for axis_targets in axis_targetss:
#              relabeled_axis_targetss.append(
#                 tuple(
#                     axis_target.__record_init__(axis=relabeler.add(axis_target.axis, "axis"), expr=relabel_expr(axis_target.expr, relabeler))
#                     for axis_target in axis_targets
#                 )
#             )
#         relabeled_targets[relabeled_path] = tuple(relabeled_axis_targetss)
#     return idict(relabeled_targets)
#
#
# @canonicalize_labels.register(pyop3.axis_tree.Axis)
# def _(axis, relabeler):
#     relabeled_label = relabeler.add(axis.label, "axis")
#     relabeled_components = tuple(canonicalize_labels(c, relabeler) for c in axis.components)
#     return axis.__record_init__(_label=relabeled_label, components=relabeled_components)
#
# @canonicalize_labels.register(pyop3.axis_tree.AxisComponent)
# def _(component: pyop3.axis_tree.AxisComponent, relabeler) -> tuple:
#     from pyop3.expr.visitors import canonicalize_labels as relabel_expr
#
#     relabeled_regions = tuple(canonicalize_labels(r, relabeler) for r in component.regions)
#     if component._size is not None:
#         relabeled_size = relabel_expr(component._size, relabeler)
#     else:
#         relabeled_size = None
#     return component.__record_init__(regions=relabeled_regions, _size=relabeled_size)
#
#
# @canonicalize_labels.register(pyop3.axis_tree.AxisComponentRegion)
# def _(region: pyop3.axis_tree.AxisComponent, relabeler) -> tuple:
#     from pyop3.expr.visitors import canonicalize_labels as relabel_expr
#
#     return region.__record_init__(size=relabel_expr(region.size, relabeler))
