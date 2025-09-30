from __future__ import annotations

import collections
import functools
import numbers
from typing import Any

from .tree import AbstractAxisTree, AxisForest, AxisTree, Axis, _UnitAxisTree, ContextSensitiveAxisTree, IndexedAxisTree, AxisComponent


@functools.singledispatch
def as_axis_forest(arg: Any) -> AxisForest:
    axis_tree = as_axis_tree(arg)
    return as_axis_forest(axis_tree)


@as_axis_forest.register(AxisForest)
def _(arg):
    return arg


@as_axis_forest.register(AbstractAxisTree)
@as_axis_forest.register(_UnitAxisTree)
@as_axis_forest.register(ContextSensitiveAxisTree)
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


