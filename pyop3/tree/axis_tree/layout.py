from __future__ import annotations

import collections
import functools
import itertools
import numbers
import operator
import sys
import typing
from collections import defaultdict

from immutabledict import immutabledict
from typing import Any, Optional

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyop3 import tree, utils

from pyop3.expr import AxisVar, LoopIndexVar, NaN  # TODO: I think these should be cyclic imports
from pyop3.expr.base import NAN, loopified_shape
from pyop3.expr.visitors import get_shape
from pyop3.expr.tensor import LinearDatBufferExpression
from pyop3.tree.axis_tree.tree import (
    UNIT_AXIS_TREE,
    Axis,
    AxisComponent,
    AxisTree,
    full_shape,
    merge_axis_trees,
    replace_exprs,
)
from pyop3.dtypes import IntType
from pyop3.tree.labelled_tree import parent_path
from pyop3.utils import (
    as_tuple,
    single_valued,
    just_one,
    merge_dicts,
    strict_int,
    steps,
)


import pyop3.extras.debug


def make_layouts(axes: AxisTree, loop_vars) -> immutabledict:
    if not axes.layout_axes.is_empty:
        inner_layouts = tabulate_again(axes.layout_axes)
    else:
        inner_layouts = immutabledict()

    return immutabledict({immutabledict(): 0}) | inner_layouts


def tabulate_again(axes):
    from pyop3 import do_loop
    from pyop3.insn import ArrayAssignment

    to_tabulate = []
    tabulated = {}

    # TODO: how to track regions?
    layouts = _prepare_layouts(axes, axes.root, immutabledict(), 0, to_tabulate, tabulated, ())
    start = 0

    ???

    """TODO NEXT

    The idea here is to just add some offset to all that need tabulating. Something like

    region axes = axes[region_slice]

    loop(region axes, offset.iassign(start))

    start += region axes.size

    but not that the offset for the existing dat has already been applied. The offset is
    only needed for the other arrays!

    """

    # this is done at the root of the tree, so can treat in a flattened manner...
    offsets = [0] * len(to_tabulate)  # not sure that this is needed
    for regions in _collect_regions(axes):
        for i, (offset_dat, mapping) in enumerate(to_tabulate):
            breakpoint()
            offset = offsets[i]
            my_axes, region_indices_dat = mapping[regions]
            # NOTE: I think that this is going to have to be a loop now.
            # this fails
            # for j in range(region_size):
            #     offset_dat[offset+j] = region_indices[my_ptr+j] + start

            idx = my_axes.index()

            indexed = offset_dat[idx].concretize()
            # assignee = LinearDatBufferExpression(indexed.buffer, indexed.layout)
            assignee = indexed  # surely?

            indexed_expr = region_indices_dat[idx].concretize()
            expression = LinearDatBufferExpression(indexed_expr.buffer, indexed_expr.layout) + start

            assignment = ArrayAssignment(assignee, expression, "write")

            # pyop3.extras.debug.enable_conditional_breakpoints()
            do_loop(idx, assignment)

            result = assignment.assignee.buffer.buffer._data
            # breakpoint()

            start += my_axes.size
            offsets[i] += region_indices_dat.size
    # utils.debug_assert(lambda: all((dat.buffer._data >= 0).all() for dat in to_tabulate.values()))
    return layouts


# TODO:
# I think the right approach here is to deal more with subtrees. If we have a mixed thing
# for instance we should eagerly split the tree apart and tabulate each part.
def _prepare_layouts(axes: AxisTree, axis: Axis, path_acc, layout_expr_acc, to_tabulate, tabulated, parent_axes) -> immutabledict:
    """Traverse the axis tree and prepare zeroed arrays for offsets.

    Any axes that do not require tabulation will also be set at this point.

    """
    layouts = {}
    start = 0
    for component in axis.components:
        path_acc_ = path_acc | {axis.label: component.label}

        subtree = axes.subtree(path_acc_)

        linear_axis = axis.linearize(component.label)
        parent_axes_ = parent_axes + (linear_axis,)

        # more regions below, cannot do anything here
        if subtree and subtree._all_region_labels:
            layout_expr_acc_ = layout_expr_acc
            layouts[path_acc_] = NAN

        # At the last region, can tabulate here
        elif component.has_non_trivial_regions and not subtree._all_region_labels:
            step_expr = _accumulate_step_sizes(subtree.size, linear_axis, parent_axes_)
            breakpoint()
            to_tabulate.append((offset_dat, step_expr))

            layout_expr_acc_ = layout_expr_acc + step_expr + start
            layouts[path_acc_] = layout_expr_acc_

        # At leaves the layout function is trivial
        elif not subtree:
            layout_expr_acc_ = layout_expr_acc + AxisVar(linear_axis) + start
            layouts[path_acc_] = layout_expr_acc_

        # Tabulate
        else:
            step_expr = _accumulate_step_sizes(subtree.size, linear_axis)
            layout_expr_acc_ = layout_expr_acc + step_expr + start
            layouts[path_acc_] = layout_expr_acc_

        start += axis_tree_component_size(axes, path_acc, component)

        if subaxis := axes.node_map[path_acc_]:
            sublayouts = _prepare_layouts(axes, subaxis, path_acc_, layout_expr_acc_, to_tabulate, tabulated, parent_axes_)
            layouts |= sublayouts

    return immutabledict(layouts)


def _collect_regions(axes: AxisTree, *, path: PathT = immutabledict()):
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


def _tabulate_steps(offset_axes, step, regions=True):
    from pyop3 import Dat
    from pyop3.expr.visitors import replace_terminals

    breakpoint()  # old

    assert step != 0
    # if isinstance(step, numbers.Integral):
    # if False:
    #     offsets = np.arange(offset_axes.max_size+1, dtype=IntType) * step
    # else:
    #     step_dat = Dat.empty(full_shape(offset_axes.regionless), dtype=IntType)
    #     step_dat.assign(step, eager=True)
    #     # offsets = steps(step_dat.buffer._data, drop_last=False)
    #     offsets = steps(step_dat.buffer._data)
    #     offset_dat = Dat(step_dat.axes, data=offsets)

    if not regions:
        step_dat = Dat.empty(full_shape(offset_axes.regionless), dtype=IntType)
        step_dat.assign(step, eager=True)
        # offsets = steps(step_dat.buffer._data, drop_last=False)
        offsets = steps(step_dat.buffer._data)
        offset_dat = Dat(step_dat.axes, data=offsets)
        return offsets

    # split this up per region
    # TODO: just do this at the same point as the rest
    region_steps = {}
    for regions in _collect_regions(offset_axes):
        regioned_offset_axes = offset_axes.with_region_labels(regions)

        step_dat = Dat.empty(full_shape(regioned_offset_axes.regionless), dtype=IntType)

        # This is needed in case 'step' is some expression in terms of the
        # un-regioned axis variables (e.g. 'i_{mesh}' instead of 'i_{(mesh, owned)}').
        # We basically retrieve the slice information from the regioned axes
        # and apply it here too.
        region_slice_map = {}
        targets = regioned_offset_axes.targets[0]
        for path in tree.accumulate_path(regioned_offset_axes.leaf_path):
            if path in targets:
                region_slice_map |= targets[path][1]

        step_expr = replace_terminals(step, region_slice_map)
        step_dat.assign(step_expr, eager=True)

        # offsets = steps(step_dat.buffer._data, drop_last=False)
        offsets = steps(step_dat.buffer._data)
        offset_dat = Dat(step_dat.axes, data=offsets)

        region_steps[regions] = (regioned_offset_axes, offset_dat)

    return region_steps


# NOTE: This is a very generic operation and I probably do something very similar elsewhere
def _axis_tree_size(axes):
    if axes.is_empty:
        return 0
    else:
        return _axis_tree_size_rec(axes, immutabledict())


def _axis_tree_size_rec(axis_tree: AxisTree, path):
    axis = axis_tree.node_map[path]

    if axis is None:
        return 1

    # The size of an axis tree is simply the product of the sizes of the
    # different nested subaxes. This remains the case even for ragged
    # inner axes.
    tree_size = 0
    for component in axis.components:
        tree_size += axis_tree_component_size(axis_tree, path, component)
    return tree_size


def axis_tree_component_size(axis_tree, path, component):
    from pyop3 import Dat, loop as loop_
    from pyop3.expr.visitors import replace_terminals, replace

    axis = axis_tree.node_map[path]
    linear_axis = axis.linearize(component.label)

    path_ = path | {axis.label: component.label}
    if axis_tree.node_map[path_]:
        subtree_size = _axis_tree_size_rec(axis_tree, path_)
    else:
        subtree_size = 1

    # the size is the sum of the array...

    # don't want to have <num> * <array> because (for example) the size of 3 * [1, 2, 1] is 4!
    # Therefore the right thing to do is to sum the internal bits.
    if not isinstance(subtree_size, numbers.Integral):
        # Consider the following cases:
        #
        # Example 1:
        #
        #   subtree size: [[2, 1, 0], [2, 4, 1]][i, j]
        #   component size: 3 (j)
        #
        # We need a new size array with free index i:
        #
        #   size = [3, 7][i]
        #
        # and therefore need to execute the loop:
        #
        #   for i < 2
        #     for j < 3
        #       size[i] += subtree[i, j]
        #
        # Example 2:
        #
        #   subtree size: [2, 1, 0][i]
        #   component size: 2 (j)
        #
        # Then the size is just the subtree size and no loop is needed.

        import pyop3.extras.debug
        # pyop3.extras.debug.maybe_breakpoint()

        subtree_size_axes, outer_loop_to_axis_var_replace_map = loopified_shape(subtree_size)
        assert subtree_size_axes.is_linear

        # think tensor contractions, look for matches
        if axis.label in subtree_size_axes.node_labels:
            # current axis is used - need to do a loop
            component_size_axes = AxisTree.from_iterable(
                (
                    ax
                    for ax in subtree_size_axes.nodes
                    if ax.label != axis.label
                )
            )
            if component_size_axes.is_empty:
                component_size_axes = UNIT_AXIS_TREE
            all_axes = subtree_size_axes
        else:
            # current axis not used, just pass it up
            return component.local_size * subtree_size
        assert all_axes.is_linear

        component_size = Dat.zeros(component_size_axes, dtype=IntType).concretize()

        i = all_axes.index()

        # Replace AxisVars with LoopIndexVars in the size expression so we can
        # access them in a loop
        # this is a bit of a weird bit: loopindex -> axis_loopindex -> loopindex(axis_loopindex)
        subtree_size_tmp = replace(subtree_size, outer_loop_to_axis_var_replace_map)

        # TODO: might need to do something similar for component_size

        axis_to_loop_var_replace_map = {
            AxisVar(ax): LoopIndexVar(i, ax)
            for ax in i.iterset.nodes
        }

        # 'index' the expressions so they can be used inside a loop
        component_size = replace(component_size, axis_to_loop_var_replace_map)
        subtree_size_expr  = replace(subtree_size_tmp, axis_to_loop_var_replace_map)

        loop_(i,
            component_size.iassign(subtree_size_expr)
        )()


        if component_size_axes is UNIT_AXIS_TREE:
            return just_one(component_size.buffer.buffer._data)
        else:
            loop_to_axis_var_replace_map_ = utils.invert_mapping(axis_to_loop_var_replace_map)
            XXX = replace(component_size, loop_to_axis_var_replace_map_)

            axis_to_loop_var_replace_map = utils.invert_mapping(outer_loop_to_axis_var_replace_map)
            return replace(XXX, axis_to_loop_var_replace_map)
    else:
        return component.local_size * subtree_size


def _drop_constant_subaxes(axis_tree, path: ConcretePathT) -> tuple[AxisTree, int]:
    """Return an axis tree consisting of non-constant bits below ``axis``."""
    assert False, "old now"
    # NOTE: dont think I need the cache here any more
    if axis_tree.node_map[path]:
        subtree = axis_tree.subtree(path)

        key = ("_truncate_axis_tree", subtree)
        try:
            raise KeyError  # TODO
            return axis_tree.cache_get(key)
        except KeyError:
            pass

        results = _truncate_axis_tree_rec(subtree, immutabledict())

        # The best result has the largest step size (as the resulting tree is
        # smaller and therefore more generic).
        # Go in reverse order because the 'best' results are appended so we resolve clashes in the best way.
        trimmed_tree, step = max(reversed(results), key=lambda result: result[1])

        best_result = (trimmed_tree, step)

        # setdefault?
        # axis_tree.cache_set(key, best_result)
        return best_result
    else:
        return AxisTree(), 1


def _truncate_axis_tree_rec(axis_tree, path) -> tuple[tuple[AxisTree, int]]:
    # NOTE: Do a post-order traversal. Need to look at the subaxes before looking
    # at this one.
    candidates_per_component = []
    axis = axis_tree.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        if axis_tree.node_map[path_]:
            candidates = _truncate_axis_tree_rec(axis_tree, path_)
        else:
            # there is nothing below here, cannot truncate anything
            # TODO: should this be a unit tree?
            candidates = ((AxisTree(), 1),)
        candidates_per_component.append(candidates)

    axis_candidates = []
    for component_candidates in itertools.product(*candidates_per_component):
        subaxis_trees, substeps = map(tuple, zip(*component_candidates))

        # We can only consider cases where all candidates have the same step.
        try:
            substep = single_valued(substeps)
        except AssertionError:
            continue

        # The new candidate consists of the per-component subtrees stuck on to the
        # current axis.
        candidate_axis_tree = AxisTree(axis)
        for component, subtree in zip(axis.components, subaxis_trees, strict=True):
            candidate_axis_tree = candidate_axis_tree.add_subtree({axis.label: component.label}, subtree)
        axis_candidate = (candidate_axis_tree, substep)
        axis_candidates.append(axis_candidate)

    # Lastly, we can also consider the case where the entire subtree (at this
    # point) is dropped. This is only valid for constant-sized axes.
    if not _axis_needs_outer_index(axis_tree, path):
        step = _axis_tree_size(axis_tree.subtree(path))
        assert isinstance(step, numbers.Integral)
        # TODO: should this be a unit tree?
        axis_candidate = (AxisTree(), step)
        axis_candidates.append(axis_candidate)

    return tuple(axis_candidates)


def _collect_offset_subaxes(axes, axis, component, *, visited):
    if not isinstance(component.count, numbers.Integral):
        axes_iter = [ax for ax in sorted(component.count.axes.nodes, key=lambda ax: ax.id) if ax not in visited]
    else:
        axes_iter = []

    # needed since we don't care about "internal" axes here
    visited_ = visited + (axis,)

    if subaxis := axes.child(axis, component):
        for subcomponent in subaxis.components:
            subaxes = _collect_offset_subaxes(axes, subaxis, subcomponent, visited=visited_)
            for ax in subaxes:
                if ax not in axes_iter:
                    axes_iter.append(ax)

    return tuple(axes_iter)




    """
    There are two conditions that we need to worry about:
        1. does the axis have a fixed size (not ragged)?
            If so then we should emit a layout function and handle any inner bits.
            We don't need any external indices to fully index the array. In fact,
            if we were the use the external indices too then the resulting layout
            array would be much larger than it has to be (each index is basically
            a new dimension in the array).

        2. Does the axis have fixed size steps?

        If we have constant steps then we should index things using an affine layout.

    Care needs to be taken with the interplay of these options:

        fixed size x fixed step : affine - great
        fixed size x variable step : need to tabulate with the current axis and
                                     everything below that isn't yet handled
        variable size x fixed step : emit an affine layout but we need to tabulate above
        variable size x variable step : add an axis to the "count" tree but do nothing else
                                        not ready for tabulation as not fully indexed

    We only ever care about axes as a whole. If individual components are ragged but
    others not then we still need to index them separately as the steps are still not
    a fixed size even for the non-ragged components.
    """


def has_constant_step(axes: AxisTree, axis, cpt, inner_loop_vars, path=immutabledict()):
    if len(axis.components) > 1 and len(cpt._all_regions) > 1:
        # must interleave
        return False

    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if subaxis := axes.child(axis, cpt):
        return all(
            # not size_requires_external_index(axes, subaxis, c, path | {axis.label: cpt.label})
            not size_requires_external_index(axes, subaxis, c, inner_loop_vars, path)
            for c in subaxis.components
        )
    else:
        return True


# def _axis_component_has_constant_step(axes: AxisTree, axis: Axis, component: AxisComponent) -> bool:
#     # An axis component has a non-constant stride if:
#     #   (a) It is multi-component and each component has multiple regions.
#     #       For example, something like:
#     #
#     #           (owned)   | (ghost)
#     #         [A0, A1, B0 | A2, B1]
#     #
#     #   (b) It has child axes whose size depends on 'outer' axis indices.
#     #   (c) Any child axis has a multi-region component, even if the axis
#     #       isn't multi-component. For example:
#     #
#     #         Axis(2, "a") -> Axis([AxisComponent({"owned": 2, "ghost": 1})], "b")
#     #
#     #       should give
#     #
#     #         [a0b0, a0b1, a1b0, a1b1, a0b2, a1b2]
#     if len(axis.components) > 1 and len(component.regions) > 1:
#         # must interleave
#         return False
#
#     # we have a constant step if none of the internal dimensions need to index themselves
#     # with the current index (numbering doesn't matter here)
#     if subaxis := axes.child(axis, cpt):
#         return all(
#             # not size_requires_external_index(axes, subaxis, c, path | {axis.label: cpt.label})
#             not size_requires_external_index(axes, subaxis, c, inner_loop_vars, path)
#             for c in subaxis.components
#         )
#     else:
#         return True


def _tabulation_needs_subaxes(axes, path) -> bool:
    """
    As we descend the axis tree to compute layout functions we are able to access
    more and more indices and can thus tabulate offsets into arrays with more and
    more shape. This function determines whether or not we are sufficiently deep in
    the tree for us to tabulate at this point (shallower is always better because
    the tabulated array is smaller).

    With this in mind, we cannot tabulate offsets if either:

      (a) It has ragged child axes whose size depends on 'outer' axis indices, or
      (b) Any child axis has a multi-region component, even if the axis
          isn't multi-component. For example:

            Axis(2, "a") -> Axis([AxisComponent({"owned": 2, "ghost": 1})], "b")

          should give

            [a0b0, a0b1, a1b0, a1b1, a0b2, a1b2]

          meaning that axis 'a' cannot be tabulated without 'b'.

    """
    if axes.node_map[path]:
        return _axis_needs_outer_index(axes, path) or _axis_contains_multiple_regions(axes, path)
    else:
        return False


def _axis_needs_outer_index(axes, path) -> bool:
    axis = axes.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        if _component_size_needs_outer_index(component, path):
            return True

        if axes.node_map[path_]:
            if _axis_needs_outer_index(axes, path_):
                return True

    return False


def _axis_contains_multiple_regions(axes, path) -> bool:
    axis = axes.node_map[path]

    if axis is None:
        return False

    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if len(component._all_regions) > 1:
            return True

        if axes.node_map[path_]:
            if _axis_contains_multiple_regions(axes, path_):
                return True

    return False


def has_fixed_size(axes, axis, component, inner_loop_vars):
    return not size_requires_external_index(axes, axis, component, inner_loop_vars)


def _axis_component_has_fixed_size(component: AxisComponent) -> bool:
    return all(_axis_component_region_has_fixed_size(r) for r in component._all_regions)


def _axis_component_region_has_fixed_size(region: AxisComponentRegion) -> bool:
    return isinstance(region.size, numbers.Integral)


def _component_size_needs_outer_index(component: AxisComponent, path):
    from pyop3.expr.visitors import collect_axis_vars

    free_axis_labels = path.keys()
    for region in component._all_regions:
        if (
            not isinstance(region.size, numbers.Integral)
            and not (set(av.axis_label for av in collect_axis_vars(region.size)) <= free_axis_labels)
        ):
            return True
    return False


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=immutabledict(),
    *,
    loop_indices=immutabledict(),
):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    assert False, "old code"
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis, indices, loop_indices=loop_indices)
    else:
        return 1


def has_halo(axes, axis):
    assert False, "old code"
    # TODO: cleanup
    return axes.comm.size > 1
    if axis.sf is not None:
        return True
    else:
        for component in axis.components:
            subaxis = axes.child(axis, component)
            if subaxis and has_halo(axes, subaxis):
                return True
        return False
    return axis.sf is not None or has_halo(axes, subaxis)


@PETSc.Log.EventDecorator()
def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    assert False, "old code"
    outer_loops = axes.outer_loops

    if axes.is_empty:
        return 0

    if not _axis_needs_outer_index(axes, axes.root, ()):
        return _axis_size(axes, axes.root)

    sizes = []

    for idxs in my_product(outer_loops):
        source_indices = merge_dicts(idx.source_exprs for idx in idxs)
        target_indices = merge_dicts(idx.target_exprs for idx in idxs)

        size = _axis_size(axes, axes.root, target_indices)
        sizes.append(size)
    return np.asarray(sizes, dtype=IntType)


def my_product(loops):
    if len(loops) > 1:
        raise NotImplementedError(
            "Now we are nesting loops so having multiple is a "
            "headache I haven't yet tackled"
        )
    # loop, *inner_loops = loops
    (loop,) = loops

    if loop.iterset.outer_loops:
        for indices in my_product(loop.iterset.outer_loops):
            context = frozenset(indices)
            for index in loop.iter(context):
                indices_ = indices + (index,)
                yield indices_
    else:
        for index in loop.iter():
            yield (index,)


def _axis_size(
    axes: AxisTree,
    axis: Axis,
    indices=immutabledict(),
    *,
    loop_indices=immutabledict(),
):
    assert False, "old code, do not use"
    return sum(
        _axis_component_size(axes, axis, cpt, indices, loop_indices=loop_indices)
        for cpt in axis.components
    )


def _axis_component_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=immutabledict(),
    *,
    loop_indices=immutabledict(),
):
    return sum(
        _axis_component_size_region(axes, axis, component, region, indices, loop_indices=loop_indices)
        for region in component._all_regions
    )


def eval_offset(
    axes,
    layouts,
    indices,
    path=None,
    *,
    loop_exprs=immutabledict(),
):
    from pyop3.expr.visitors import evaluate as eval_expr

    if path is None:
        path = immutabledict() if axes.is_empty else just_one(axes.leaf_paths)

    # if the provided indices are not a dict then we assume that they apply in order
    # as we go down the selected path of the tree
    if not isinstance(indices, collections.abc.Mapping):
        # a single index is treated like a 1-tuple
        indices = as_tuple(indices)

        indices_ = {}
        ordered_path = iter(just_one(axes.ordered_leaf_paths))
        for index in indices:
            axis_label, _ = next(ordered_path)
            indices_[axis_label] = index
        indices = indices_

    layout_subst = layouts[immutabledict(path)]

    # offset = ExpressionEvaluator(indices, loop_exprs)(layout_subst)
    # offset = eval_expr(layout_subst, path, indices)
    offset = eval_expr(layout_subst, indices)
    # breakpoint()
    return strict_int(offset)


# TODO: singledispatch (but needs import thoughts)
# TODO: should cache this!!!
def _accumulate_step_sizes(obj: Any, axis, region_axes=None):
    from pyop3.expr import base as expr_types

    if region_axes:
        return _accumulate_dat_expr(obj, axis, region_axes)

    if isinstance(obj, numbers.Number):
        return AxisVar(axis) * obj
    elif isinstance(obj, expr_types.Mul):
        # only allow linear combinations, so only one of lhs or rhs can have shape
        if get_shape(obj.a)[0].size > 1:
            assert get_shape(obj.b)[0].size == 1
            return _accumulate_step_sizes(obj.a, axis) * obj.b
        else:
            return obj.a * _accumulate_step_sizes(obj.b, axis)
    elif isinstance(obj, LinearDatBufferExpression):
        return _accumulate_dat_expr(obj, axis)
    else:
        raise NotImplementedError


def _accumulate_dat_expr(size_expr: LinearDatBufferExpression, linear_axis: Axis, region_axes=None):
    from pyop3 import Dat, exscan, loop
    from pyop3.expr.visitors import replace

    if not region_axes and linear_axis not in utils.just_one(get_shape(size_expr)):
        return AxisVar(linear_axis) * size_expr

    # do the moral equivalent of
    #
    #   for i
    #     for j  # (the current axis)
    #       offset[i, j] = offset[i, j-1] + size[i, j]

    # by definition the current axis is in size_expr
    if not region_axes:
        offset_axes = merge_axis_trees((utils.just_one(get_shape(size_expr)), full_shape(linear_axis.as_tree())))
    else:
        offset_axes = AxisTree.from_iterable(region_axes)

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
        loop(
            ix, exscan(assignee, size_expr_alt, "+", scan_axis),
        )()

    else:
        exscan(offset_dat.concretize(), size_expr, "+", linear_axis, eager=True)

    return offset_dat.concretize()
