from __future__ import annotations

import collections
import functools
import numbers
from typing import Any

from pyop3 import utils
from .tree import AbstractAxisTree, AxisForest, AxisTree, Axis, _UnitAxisTree, ContextSensitiveAxisTree, IndexedAxisTree, AxisComponent, UnitIndexedAxisTree


@functools.singledispatch
def as_axis_tree_type(arg: Any) -> AxisTreeT:
    return as_axis_tree_type(as_axis_tree(arg))


@as_axis_tree_type.register(AbstractAxisTree)
@as_axis_tree_type.register(_UnitAxisTree)
@as_axis_tree_type.register(UnitIndexedAxisTree)
@as_axis_tree_type.register(ContextSensitiveAxisTree)
@as_axis_tree_type.register(AxisForest)
def _(axis_tree, /) -> AxisTreeT:
    return axis_tree


@functools.singledispatch
def as_axis_forest(arg: Any) -> AxisForest:
    axis_tree = as_axis_tree(arg)
    return as_axis_forest(axis_tree)


@as_axis_forest.register(ContextSensitiveAxisTree)
def _(arg):
    raise TypeError


@as_axis_forest.register(AxisForest)
def _(arg):
    return arg


@as_axis_forest.register(AbstractAxisTree)
@as_axis_forest.register(_UnitAxisTree)
@as_axis_forest.register(UnitIndexedAxisTree)
def _(arg):
    return AxisForest([arg])


@as_axis_forest.register(Axis)
def _(arg):
    return as_axis_forest(as_axis_tree(arg))


@functools.singledispatch
def as_axis_tree(arg: Any) -> AxisTree | AxisForest:
    axis = as_axis(arg)
    return as_axis_tree(axis)


@as_axis_tree.register
def _(axes_per_context: collections.abc.Mapping) -> ContextSensitiveAxisTree:
    return ContextSensitiveAxisTree(axes_per_context)


@as_axis_tree.register(AxisTree)
@as_axis_tree.register(_UnitAxisTree)
@as_axis_tree.register(ContextSensitiveAxisTree)
@as_axis_tree.register(AxisForest)
def _(axes: AxisTree) -> AxisTree:
    return axes


@as_axis_tree.register
def _(axes: IndexedAxisTree) -> IndexedAxisTree:
    return axes


@as_axis_tree.register
def _(axis: Axis) -> AxisTree:
    return AxisTree(axis)


@functools.singledispatch
def as_axis(arg: Any) -> Axis:
    component = as_axis_component(arg)
    return as_axis(component)


@as_axis.register
def _(axis: Axis) -> Axis:
    return axis


@as_axis.register
def _(component: AxisComponent) -> Axis:
    return Axis(component)


@functools.singledispatch
def as_axis_component(arg: Any) -> AxisComponent:
    from pyop3 import Dat  # cyclic import

    if isinstance(arg, Dat):
        return AxisComponent(arg)
    else:
        raise TypeError(f"No handler defined for {type(arg).__name__}")


@as_axis_component.register
def _(component: AxisComponent) -> AxisComponent:
    return component


@as_axis_component.register
def _(arg: numbers.Integral) -> AxisComponent:
    return AxisComponent(arg)


@functools.singledispatch
def collect_unindexed_axis_trees(tree: AxisTreeT, /) -> tuple[AxisTree, ...]:
    raise TypeError


@collect_unindexed_axis_trees.register(AxisTree)
@collect_unindexed_axis_trees.register(_UnitAxisTree)
def _(axis_tree, /) -> tuple[AxisTree, ...]:
    return (axis_tree,)


@collect_unindexed_axis_trees.register(IndexedAxisTree)
@collect_unindexed_axis_trees.register(UnitIndexedAxisTree)
def _(indexed_axis_tree, /) -> tuple[AxisTree, ...]:
    return (indexed_axis_tree.unindexed,)


@collect_unindexed_axis_trees.register(AxisForest)
def _(axis_forest: AxisForest, /) -> tuple[AxisTree, ...]:
    return utils.unique(sum(
        (collect_unindexed_axis_trees(tree) for tree in axis_forest.trees),
        start=(),
    ))


@collect_unindexed_axis_trees.register(ContextSensitiveAxisTree)
def _(cs_axes: ContextSensitiveAxisTree, /) -> tuple[AxisTree, ...]:
    return utils.unique(sum(
        (collect_unindexed_axis_trees(tree) for tree in cs_axes.context_map.values()),
        start=(),
    ))
