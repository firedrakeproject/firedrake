from __future__ import annotations

import numbers

from immutabledict import immutabledict as idict
from typing import Any

import numpy as np
from petsc4py import PETSc

from pyop3 import utils
from pyop3.dtypes import IntType
from pyop3.expr import AxisVar, LoopIndexVar, LinearDatBufferExpression, Dat
from pyop3.expr.base import NAN
from pyop3.expr.visitors import get_shape, replace
from pyop3.insn import exscan, Loop
from pyop3.tree import (
    Axis,
    AxisTree,
    merge_axis_trees,
)
from pyop3.tree.axis_tree.tree import full_shape, replace_exprs  # TODO: move this to visitors?

from .size import compute_axis_tree_component_size


@PETSc.Log.EventDecorator()
def compute_layouts(axis_tree: AxisTree) -> idict:
    from pyop3 import do_loop

    to_tabulate = []
    tabulated = {}

    layouts = _prepare_layouts(axis_tree, axis_tree.root, idict(), 0, to_tabulate, tabulated, ())

    # add zero for the root
    layouts = layouts | {idict(): 0}

    # TODO: There a particular cases (e.g. multiple regions but only a single
    # component or no matching regions) where it is sufficient to do an affine
    # layout instead of tabulating a start expression. We currently do not detect
    # this.
    starts = [0] * len(to_tabulate)
    for regions in _collect_regions(axis_tree):
        for i, (path, offset_dat) in enumerate(to_tabulate):
            offset_axes = axis_tree.drop_subtree(path, allow_empty_subtree=True).linearize(path)

            if not all(region in offset_axes._all_region_labels for region in regions):
                # zero-sized
                continue

            regioned_axes = offset_axes.with_region_labels(regions)
            assert not regioned_axes._all_region_labels

            ix = regioned_axes.index()

            assignee = offset_dat[ix]

            if starts[i] > 0:
                do_loop(ix, assignee.iassign(starts[i]))

            # step_size = axes.materialize().subtree(path).size or 1
            step_size = axis_tree.linearize(path, partial=True).with_region_labels(regions).size or 1

            for j, _ in enumerate(starts):
                if i != j:
                    starts[j] += step_size

    return layouts


def _prepare_layouts(axis_tree: AxisTree, axis: Axis, path_acc, layout_expr_acc, to_tabulate, tabulated, parent_axes) -> idict:
    """Traverse the axis tree and prepare zeroed arrays for offsets.

    Any axes that do not require tabulation will also be set at this point.

    """
    layouts = {}
    start = 0
    for component in axis.components:
        path_acc_ = path_acc | {axis.label: component.label}

        subtree = axis_tree.subtree(path_acc_)

        linear_axis = axis.linearize(component.label)
        parent_axes_ = parent_axes + (linear_axis,)

        # more regions below, cannot do anything here
        if subtree and subtree._all_region_labels:
            layout_expr_acc_ = layout_expr_acc
            layouts[path_acc_] = NAN

        # At the last region, can tabulate here
        elif component.has_non_trivial_regions and not subtree._all_region_labels:
            offset_axes = AxisTree.from_iterable(parent_axes_)
            if subtree:
                offset_dat = _tabulate_regions(offset_axes, subtree.size)
            else:
                offset_dat = _tabulate_regions(offset_axes, 1)
            to_tabulate.append((path_acc_, offset_dat))

            assert layout_expr_acc == 0
            layout_expr_acc_ = offset_dat.concretize()
            layouts[path_acc_] = layout_expr_acc_

        # At leaves the layout function is trivial
        elif not subtree:
            layout_expr_acc_ = layout_expr_acc + AxisVar(linear_axis) + start
            layouts[path_acc_] = layout_expr_acc_

        # Tabulate
        else:
            step_expr = _accumulate_step_sizes(subtree.size, linear_axis)

            if linear_axis not in utils.just_one(get_shape(step_expr)).nodes:
                step_expr = AxisVar(linear_axis) * step_expr

            layout_expr_acc_ = layout_expr_acc + step_expr + start
            layouts[path_acc_] = layout_expr_acc_

        start += compute_axis_tree_component_size(axis_tree, path_acc, component.label)

        if subaxis := axis_tree.node_map[path_acc_]:
            sublayouts = _prepare_layouts(axis_tree, subaxis, path_acc_, layout_expr_acc_, to_tabulate, tabulated, parent_axes_)
            layouts |= sublayouts

    return idict(layouts)


def _collect_regions(axes: AxisTree, *, path: PathT = idict()):
    # NOTE: This is now a set, not a mapping
    """
    (can think of as some sort of linearisation of the tree, should probably error if orders do not match)

    Examples
    --------

    Axis({"x": ["A", "B"], "y": ["B"]}, "a") -> [{"a": "A"}, {"a": "B"}]

    Axis({"x": ["A", "B"], "y": ["B", "A"]}, "a") -> [{"a": "A"}, {"a": "B"}]

    Axis({"x": ["A", "B"], "y": ["C"]}, "a") -> [{"a": "A"}, {"a": "B"}, {"a": "C"}]

    nested

    (assumes y is shared)
    Axis({"x": ["A", "B"]}, "a") / Axis({"y": ["X", "Y"]}, "b")

        ->

    [{"a": "A", "b": "X"}, {"a": "A", "b": "Y"}, {"a": "B", "b": "X"}, {"a": "B", "b": "Y"}]

    (if y is not shared)
    Axis({"x": ["A", "B"]}, "a") / Axis({"y": ["X", "Y"]}, "b")

        ->

    [{"a": "A", "b": "X"}, {"a": "A", "b": "Y"}, {"a": "B", "b": "X"}, {"a": "B", "b": "Y"}]

    """
    merged_regions = []  # NOTE: Could be an ordered set
    axis = axes.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        for region in component._all_regions:
            merged_region = frozenset({region.label}) if region.label is not None else frozenset()

            if axes.node_map[path_]:
                for submerged_region in _collect_regions(axes, path=path_):
                    merged_region_ = merged_region | submerged_region
                    if merged_region_ not in merged_regions:
                        merged_regions.append(merged_region_)
            else:
                if merged_region not in merged_regions:
                    merged_regions.append(merged_region)
    return tuple(merged_regions)


# TODO: singledispatch (but needs import thoughts)
# TODO: should cache this!!!
def _accumulate_step_sizes(obj: Any, axis):
    from pyop3.expr import base as expr_types

    if isinstance(obj, numbers.Number):
        return obj
    elif isinstance(obj, expr_types.Mul):
        return _accumulate_step_sizes(obj.a, axis) * _accumulate_step_sizes(obj.b, axis)
    elif isinstance(obj, expr_types.Add):
        return _accumulate_step_sizes(obj.a, axis) + _accumulate_step_sizes(obj.b, axis)
    elif isinstance(obj, LinearDatBufferExpression):
        return _accumulate_dat_expr(obj, axis).concretize()
    else:
        raise NotImplementedError


def _accumulate_dat_expr(size_expr: LinearDatBufferExpression, linear_axis: Axis, offset_axes=None):
    if offset_axes is None:
        if linear_axis not in utils.just_one(get_shape(size_expr)):
            return size_expr

        # by definition the current axis is in size_expr
        offset_axes = merge_axis_trees((utils.just_one(get_shape(size_expr)), full_shape(linear_axis.as_tree())))

        regions = False

    # do the moral equivalent of
    #
    #   for i
    #     for j  # (the current axis)
    #       offset[i, j] = offset[i, j-1] + size[i, j]


    # remove current axis as we need to scan over it
    loc = utils.just_one(path for path, axis_ in offset_axes.node_map.items() if axis_ == linear_axis)
    outer_loop_tree = offset_axes.drop_node(loc)
    assert linear_axis not in outer_loop_tree.nodes


    # this is just accumulating the arrays in size!

    # but scanning is hard...


    offset_dat = Dat.zeros(offset_axes.regionless, dtype=IntType)

    if not outer_loop_tree.is_empty:
        ix = outer_loop_tree.index()

        axis_to_loop_var_replace_map = {
            AxisVar(ax): LoopIndexVar(ix, ax)
            for ax in ix.iterset.nodes
        }

        size_expr_alt = replace(size_expr, axis_to_loop_var_replace_map)

        assignee = offset_dat[ix].concretize()
        scan_axis = replace_exprs(linear_axis, axis_to_loop_var_replace_map)
        Loop(
            ix, exscan(assignee, size_expr_alt, "+", scan_axis),
        )()

    else:
        exscan(offset_dat.concretize(), size_expr, "+", linear_axis, eager=True)

    return offset_dat


# This gets the sizes right for a particular dat, then we merge them above
def _tabulate_regions(offset_axes, step):
    # Construct a permutation
    locs = np.full(offset_axes.size, -1, dtype=IntType)
    ptr = 0
    for regions in _collect_regions(offset_axes):
        regioned_offset_axes = offset_axes.with_region_labels(regions)
        assert not regioned_offset_axes._all_region_labels

        regioned_offset_axes = type(regioned_offset_axes)(regioned_offset_axes.node_map, targets=regioned_offset_axes.targets, unindexed=regioned_offset_axes.unindexed.regionless)

        if not regioned_offset_axes.is_linear:
            raise NotImplementedError("Doesn't strictly have to be linear here")

        region_offset_dat = Dat.empty(regioned_offset_axes, dtype=IntType)

        offset_expr = utils.just_one(regioned_offset_axes.leaf_subst_layouts.values())

        region_offset_dat.assign(offset_expr, eager=True)

        region_size = regioned_offset_axes.size
        locs[ptr:ptr+region_size] = region_offset_dat.data_ro
        ptr += region_size

    # now sizes
    step_dat = Dat.zeros(offset_axes.regionless, dtype=IntType)
    step_dat.assign(step, eager=True)

    reordered_steps = step_dat.data_ro[locs]

    reordered_offsets = utils.steps(reordered_steps)
    offsets = reordered_offsets[utils.invert(locs)]

    return Dat(offset_axes.regionless, data=offsets)
