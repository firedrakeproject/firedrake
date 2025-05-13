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

from pyop3.axtree.tree import (
    Axis,
    AxisComponent,
    AxisComponentRegion,
    AxisTree,
    AxisVar,
    NaN,
    Operator
)
from pyop3.dtypes import IntType
from pyop3.utils import (
    StrictlyUniqueDict,
    as_tuple,
    single_valued,
    is_single_valued,
    just_one,
    OrderedSet,
    merge_dicts,
    strict_int,
    strictly_all,
    steps,
)


import pyop3.extras.debug


class IntRef:
    """Pass-by-reference integer."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self


def make_layouts(axes: AxisTree, loop_vars) -> immutabledict:
    if not axes.layout_axes.is_empty:
        inner_layouts = tabulate_again(axes.layout_axes)
    else:
        inner_layouts = immutabledict()

    return immutabledict({immutabledict(): 0}) | inner_layouts


def tabulate_again(axes):
    to_tabulate = {}
    layouts = _prepare_layouts(axes, axes.root, immutabledict(), 0, (), to_tabulate)
    starts = {array: 0 for array in to_tabulate.values()}
    for region in _collect_regions(axes):
        for tree, offset_dat in to_tabulate.items():
            starts[offset_dat] += _tabulate_offset_dat(offset_dat, tree, region, starts[offset_dat])
    utils.debug_assert(lambda: all((dat.buffer._data >= 0).all() for dat in to_tabulate.values()))
    return layouts


# TODO: I think a better way to do this is to track 'free indices' and see if the 'needed indices' <= 'free'
# at which point we can tabulate.
def _prepare_layouts(axes: AxisTree, axis: Axis, path_acc, layout_expr_acc, free_axes, to_tabulate) -> immutabledict:
    """Traverse the axis tree and prepare zeroed arrays for offsets.

    Any axes that do not require tabulation will also be set at this point.

    """
    from pyop3 import Dat
    from pyop3.tensor.dat import LinearDatArrayBufferExpression, as_linear_buffer_expression

    if len(axis.components) > 1 and not all(_axis_component_has_fixed_size(c) for c in axis.components):
        # Fixing this would require deciding what to do with the start variable, which
        # might need tabulating itself.
        raise NotImplementedError(
            "Cannot yet tabulate axes with multiple components if any of them are ragged"
        )

    layouts = {}
    start = 0
    for i, component in enumerate(axis.components):
        path_acc_ = path_acc | {axis.label: component.label}

        mysubaxis = axes.child(axis, component)
        if mysubaxis:
            mysubtree = axes.subtree(mysubaxis)

        # If the axis tree has zero size but is not empty then it makes no sense to give it a layout
        if mysubaxis and not mysubtree.is_empty and _axis_tree_size(mysubtree) == 0:
            component_layout = 0
        else:
            subtree, step = _drop_constant_subaxes(axes, axis, component)

            linear_axis = Axis([component], axis.label)
            offset_axes = AxisTree.from_iterable([*free_axes, linear_axis])

            free_axes_ = free_axes + (linear_axis,)

            if subtree in to_tabulate:
                # We have already seen an identical tree elsewhere, don't need to create a new array here
                offset_dat = to_tabulate[subtree]
                offset_dat_expr = as_linear_buffer_expression(offset_dat)
                component_layout = offset_dat_expr * step + start
            else:
                if _tabulation_needs_subaxes(axes, axis, component, free_axes_):
                    # 1. Needs subindices to be able to tabulate anything, pass down
                    component_layout = NaN()
                elif subtree.is_empty:
                    # 2. Affine access
                    assert subtree.is_empty

                    # FIXME: weakness in algorithm
                    if step == 0:
                        step = 1
                    component_layout = AxisVar(axis.label) * step + start
                else:
                    # 3. Non-constant stride, must tabulate
                    offset_dat = Dat(offset_axes, data=np.full(offset_axes.size, -1, dtype=IntType))
                    to_tabulate[subtree] = offset_dat

                    # NOTE: This is really unpleasant, we want an expression type here but need
                    # axes to do the tabulation.
                    offset_dat_expr = as_linear_buffer_expression(offset_dat)
                    component_layout = offset_dat_expr * step + start

        if not isinstance(component_layout, NaN):
            layout_expr_acc_ = layout_expr_acc + component_layout
            layouts[path_acc_] = layout_expr_acc_
            free_axes_ = ()
        else:
            layouts[path_acc_] = NaN()
            layout_expr_acc_ = layout_expr_acc

        # NOTE: Not strictly necessary but it means we don't currently break with ragged
        if i < len(axis.components) - 1:
            # start += _axis_component_size(axes, axis, component)
            if subaxis := axes.child(axis, component):
                start += component.local_size * _axis_tree_size(axes.subtree(subaxis))
            else:
                start += component.local_size

        if subaxis := axes.child(axis, component):
            sublayouts = _prepare_layouts(axes, subaxis, path_acc_, layout_expr_acc_, free_axes_, to_tabulate)
            layouts |= sublayouts
            # to_tabulate |= subdats
        else:
            assert not free_axes_

    # return immutabledict(layouts), to_tabulate
    return immutabledict(layouts)


def _collect_regions(axes: AxisTree, *, axis: Axis | None = None):
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
    if axis is None:
        axis = axes.root
        axis = typing.cast(Axis, axis)  # make the type checker happy

    merged_regions = []  # NOTE: Could be an ordered set
    for component in axis.components:
        for region in component._all_regions:
            merged_region = {axis.label: region.label}

            if subaxis := axes.child(axis, component):
                for submerged_region in _collect_regions(axes, axis=subaxis):
                    merged_region_ = merged_region | submerged_region
                    if merged_region_ not in merged_regions:
                        merged_regions.append(merged_region_)
            else:
                if merged_region not in merged_regions:
                    merged_regions.append(merged_region)
    return tuple(merged_regions)


def _tabulate_offset_dat(offset_dat, axes, region, start):
    from pyop3 import Dat

    # NOTE: We don't handle regions at all. What should be done?

    step_expr = _axis_tree_size(axes)
    assert not isinstance(step_expr, numbers.Integral), "Constant steps should already be handled"

    # NOTE: If the step_expr is just an array we can avoid a loop here since we
    # would only be doing a copy.
    step_dat = Dat.empty(offset_dat.axes, dtype=IntType)
    step_dat.assign(step_expr, eager=True)

    offsets = steps(step_dat.buffer._data, drop_last=False)
    offset_dat.buffer._data[...] = offsets[:-1]

    return offsets[-1]


# NOTE: This is a very generic operation and I probably do something very similar elsewhere
def _axis_tree_size(axes):
    if axes.is_empty:
        return 0
    else:
        return _axis_tree_size_rec(axes, axes.root)


def _axis_tree_size_rec(axis_tree: AxisTree, axis: Axis):
    # The size of an axis tree is simply the product of the sizes of the
    # different nested subaxes. This remains the case even for ragged
    # inner axes.
    tree_size = 0
    for component in axis.components:
        if subaxis := axis_tree.child(axis, component):
            subtree_size = _axis_tree_size_rec(axis_tree, subaxis)
            tree_size += component.local_size * subtree_size
        else:
            tree_size += component.local_size
    return tree_size


def _drop_constant_subaxes(axis_tree, axis, component) -> tuple[AxisTree, int]:
    """Return an axis tree consisting of non-constant bits below ``axis``."""
    # NOTE: dont think I need the cache here any more
    if subaxis := axis_tree.child(axis, component):
        subtree = axis_tree.subtree(subaxis)

        key = ("_truncate_axis_tree", subtree)
        try:
            return axis.cache_get(key)
        except KeyError:
            pass

        results = _truncate_axis_tree_rec(subtree, subaxis)

        # The best result has the largest step size (as the resulting tree is
        # smaller and therefore more generic).
        # Go in reverse order because the 'best' results are appended so we resolve clashes in the best way.
        trimmed_tree, step = max(reversed(results), key=lambda result: result[1])

        # add current axis
        # tree = AxisTree(axis.copy(components=[component]))
        # tree = AxisTree(axis)
        # tree = tree.add_subtree(trimmed_tree, axis, component)

        # best_result = (tree, step)
        best_result = (trimmed_tree, step)

        # setdefault?
        axis.cache_set(key, best_result)
        return best_result
    else:
        # return AxisTree(axis.copy(components=[component])), 1
        # return AxisTree(axis), 1
        return AxisTree(), 1

def _truncate_axis_tree_rec(axis_tree, axis) -> tuple[tuple[AxisTree, int]]:
    # NOTE: Do a post-order traversal. Need to look at the subaxes before looking
    # at this one.
    candidates_per_component = []
    for component in axis.components:
        if subaxis := axis_tree.child(axis, component):
            candidates = _truncate_axis_tree_rec(axis_tree, subaxis)
        else:
            # there is nothing below here, cannot truncate anything
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
            candidate_axis_tree = candidate_axis_tree.add_subtree(subtree, axis, component)
        axis_candidate = (candidate_axis_tree, substep)
        axis_candidates.append(axis_candidate)

    # Lastly, we can also consider the case where the entire subtree (at this
    # point) is dropped. This is only valid for constant-sized axes.
    if not _axis_needs_outer_index(axis_tree, axis, (axis,)):
        step = _axis_tree_size(axis_tree.subtree(axis))
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


def _tabulation_needs_subaxes(axes, axis, component, free_axes: tuple) -> bool:
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
    if subaxis := axes.child(axis, component):
        return _axis_needs_outer_index(axes, subaxis, free_axes) or _axis_contains_multiple_regions(axes, subaxis)
    else:
        return False


def _axis_needs_outer_index(axes, axis, visited) -> bool:
    for component in axis.components:
        if any(_region_size_needs_outer_index(r, visited) for r in component._all_regions):
            return True

        if subaxis := axes.child(axis, component):
            if _axis_needs_outer_index(axes, subaxis, visited + (axis,)):
                return True

    return False


def _axis_contains_multiple_regions(axes, axis) -> bool:
    for component in axis.components:
        if len(component._all_regions) > 1:
            return True

        if subaxis := axes.child(axis, component):
            if _axis_contains_multiple_regions(axes, subaxis):
                return True

    return False


def has_fixed_size(axes, axis, component, inner_loop_vars):
    return not size_requires_external_index(axes, axis, component, inner_loop_vars)


def _axis_component_has_fixed_size(component: AxisComponent) -> bool:
    return all(_axis_component_region_has_fixed_size(r) for r in component._all_regions)


def _axis_component_region_has_fixed_size(region: AxisComponentRegion) -> bool:
    return isinstance(region.size, numbers.Integral)


# def _requires_external_index(axtree, axis, component_index, region):
#     """Return `True` if more indices are required to index the multi-axis layouts
#     than exist in the given subaxis.
#     """
#     return size_requires_external_index_region(
#         axtree, axis, component_index, region, set()
#     )
#
# def requires_external_index_region(axtree, axis, component_index, region):
#     """Return `True` if more indices are required to index the multi-axis layouts
#     than exist in the given subaxis.
#     """
#     return size_requires_external_index_region(
#         axtree, axis, component_index, region, set()
#     )
#
#
# def size_requires_external_index(axes, axis, component, inner_loop_vars, path=pmap()):
#     return any(
#         size_requires_external_index_region(axes, axis, component, region, inner_loop_vars, path=path)
#         for region in component.regions
#     )


def _region_size_needs_outer_index(region, free_axes):
    from pyop3.tensor import Dat, LinearDatArrayBufferExpression
    from pyop3.expr_visitors import collect_axis_vars

    free_axis_labels = frozenset(ax.label for ax in free_axes)

    size = region.size

    if isinstance(size, Dat):
        if size.axes.is_empty:
            leafpath = immutabledict()
        else:
            leafpath = just_one(size.axes.leaf_paths)
        layout = size.axes._subst_layouts_default[leafpath]

        # is the path sufficient? i.e. do we have enough externally provided indices
        # to correctly index the axis?
        if not size.axes.is_empty:
            for axlabel, clabel in size.axes.path(*size.axes.leaf).items():
                if axlabel not in free_axis_labels:
                    return True

    elif isinstance(size, LinearDatArrayBufferExpression):
        if not (set(v.axis_label for v in collect_axis_vars(size.layout)) <= free_axis_labels):
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
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis, indices, loop_indices=loop_indices)
    else:
        return 1


def has_halo(axes, axis):
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


def _axis_component_size_region(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    region,
    indices=immutabledict(),
    *,
    loop_indices=immutabledict(),
):
    count = _as_int(region.size, indices, loop_indices=loop_indices)
    if subaxis := axes.child(axis, component):
        return sum(
            _axis_size(
                axes,
                subaxis,
                indices | {axis.label: i},
                loop_indices=loop_indices,
            )
            for i in range(count)
        )
    else:
        return count


@functools.singledispatch
def _as_int(arg: Any, indices, path=None, *, loop_indices=immutabledict()):
    assert False, "not used?"
    from pyop3 import Dat
    from pyop3.array.dat import _ExpressionDat

    if isinstance(arg, (Dat, _ExpressionDat)):
        # TODO this might break if we have something like [:, subset]
        # I will need to map the "source" axis (e.g. slice_label0) back
        # to the "target" axis
        # return arg.get_value(indices, target_path, index_exprs)
        return arg.get_value(indices, path, loop_exprs=loop_indices)
    else:
        raise TypeError


@_as_int.register
def _(arg: numbers.Real, *args, **kwargs):
    return strict_int(arg)


def eval_offset(
    axes,
    layouts,
    indices,
    path=None,
    *,
    loop_exprs=immutabledict(),
):
    from pyop3.expr_visitors import evaluate as eval_expr

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
