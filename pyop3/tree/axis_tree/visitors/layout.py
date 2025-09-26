from __future__ import annotations

import functools
import numbers
from typing import Any

from immutabledict import immutabledict as idict

import numpy as np
from petsc4py import PETSc

from pyop2.caching import scoped_cache
from pyop3 import expr as op3_expr, utils
from pyop3.dtypes import IntType
from pyop3.expr import AxisVar, LoopIndexVar, LinearDatBufferExpression, Dat, ExpressionT
from pyop3.expr.base import NAN, get_loop_tree, loopified_shape
from pyop3.expr.visitors import get_shape, replace
from pyop3.insn import exscan, loop_
from pyop3.tree import (
    Axis,
    ConcretePathT,
    AxisTree,
    merge_axis_trees,
)
from pyop3.tree.axis_tree.tree import full_shape, loopify_axis_tree, replace_exprs  # TODO: move this to visitors?

from .size import compute_axis_tree_component_size


@scoped_cache()
@PETSc.Log.EventDecorator()
def compute_layouts(axis_tree: AxisTree) -> idict[ConcretePathT, ExpressionT]:
    """Compute the layout functions for an axis tree.

    Layout functions are expressions that take axis variables (symbolic indices
    per axis) and evaluate to an integer offset. As the simplest possible example
    consider the axis tree:

        {"A": 5}

    This tree has only a single axis and so it only has a single layout function:

        {"A": None}: i_A

    Here ``{"A": None}`` refers to the path through the tree where the layout
    resides and ``i_A`` is the layout function. Since the tree only has a single
    axis the mapping between axis indices and offsets is trivially identity.

    Note that this tree will also have the zero layout:

        {}: 0

    meaning that if you are at the root of the tree then the offset must be zero.

    Parameters
    ----------
    axis_tree :
        The axis tree to compute the layouts of.

    Returns
    -------
    layouts :
        Mapping from path through the axis tree to the layout function.

    Examples
    --------

    For more examples please refer to ``tests/pyop3/unit/test_layout.py``.

    1. Linear axis tree
    ~~~~~~~~~~~~~~~~~~~
    For the simple axis tree (equivalent to a 2D numpy array):

        {"A": 5}
        └──➤ {"B": 3}

    the layout functions are:

        {
            {}: 0,
            {"A": None}: 3*i_A,
            {"A": None, "B": None}: 3*i_A + i_B,
        }

    2. Multi-component axis tree
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For the axis tree:

        {A: [{0: 3}, {1: 4}]}}
        ├──➤ {B: 2}
        └──➤ {C: 1}

    the layout functions are:

        {
            {}: 0,
            {"A": 0}: 2*i_A,
            {"A": 0, "B": None}: 2*i_A + i_B,
            {"A": 1}: i_A + 6,
            {"A": 1, "C": None}: i_A + i_C + 6,
        }

    3. Ragged axis tree
    ~~~~~~~~~~~~~~~~~~~

    TODO

    4. Multi-region axis tree
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    For the axis tree:

        {A: [{a: 1}, {b: 1}]}
        ├──➤ {B: [{x: 2}, {y: 1}]}
        └──➤ {C: [{x: 2}, {y: 1}]}

    where 'a' and 'b' are axis components with size 1 and 'x' and 'y' are
    component *regions*, we expect to have the following data layout:

        [ a0, a1, b0, b1 || a2, b2 ]
               "x"           "y"

    In other words all entities in the 'x' region are partitioned to occur
    before those in 'y'. This means that the layout functions are as follows:

        {
            {}: 0,
            {"A": "a"}: NaN,
            {"A": "a", "B": None}: [[0, 1, 4]][i_A, i_B],
            {"A": "b"}: NaN,
            {"A": "b", "C": None}: [[2, 3, 5]][i_A, i_C],
        }

    The {"A": "a"} and {"A": "b"} entries are NaNs because they do not address
    contiguous data and so are meaningless.

    """
    return _compute_layouts(axis_tree)
    # This old approach doesn't quite work as it adds layouts for the non-existent
    # axes. I think a more robust approach is to reconstruct as needed during
    # tabulation.
    #
    # loopified_axis_tree, axis_var_replace_map = loopify_axis_tree(axis_tree)
    # loopified_layouts = _compute_layouts(loopified_axis_tree)
    #
    # if loopified_axis_tree == axis_tree:
    #     return loopified_layouts
    #
    # layouts = {}
    # for loopified_path, loopified_layout in loopified_layouts.items():
    #     path = idict({
    #         axis_label: component_label
    #         for axis_label, component_label in loopified_path.items()
    #         if axis_label in axis_tree.node_labels
    #     })
    #     layouts[path] = replace(loopified_layout, axis_var_replace_map)
    # return idict(layouts)


def _compute_layouts(axis_tree: AxisTree) -> idict[ConcretePathT, ExpressionT]:
    # First traverse the axis tree and compute everything we can.
    to_tabulate = []
    tabulated = {}
    layouts = _prepare_layouts(axis_tree, idict(), 0, to_tabulate, tabulated, ())
    # add zero for the root
    layouts = layouts | {idict(): 0}

    # Now tweak the offsets for multi-region components. This is necessary because
    # the initial traversal treats axis components in isolation and so the strides
    # across regions are not yet known.
    #
    # As an example consider the following multi-region tree:
    #
    #     {A: [1, 1]}
    #     ├──➤ {B: [{x: 2}, {y: 1}]}
    #     └──➤ {C: [{x: 2}, {y: 1}]}
    #
    # Here 'x' and 'y' are the different regions that we want to partition. Hence
    # we want the final layout to be:
    #
    #     [ a0b0, a0b1, a1c0, a1c1 || a0b2, a1c2 ]
    #                "x"                  "y"
    #
    # Since the offsets for each component are computed in isolation the current
    # relevant layout functions are:
    #
    #     {
    #         {"A": 0, "B": None}: [[0, 1, 2]][i_A, i_B]
    #         {"A": 1, "C": None}: [[0, 1, 2]][i_A, i_C]
    #     }
    #
    # whereas the correct values are:
    #
    #     {
    #         {"A": 0, "B": None}: [[0, 1, 4]][i_A, i_B]
    #         {"A": 1, "C": None}: [[2, 3, 5]][i_A, i_C]
    #     }
    #
    # To compute this we track a global offset and add it to arrays for each
    # region. In this case this results in:
    #
    # 1. regions = {"x"}, starts = [0, 0], [[0, 1, 2]][i_A, i_B], [[0, 1, 2]][i_A, i_C]
    # 2. regions = {"x"}, starts = [0, 2], [[0, 1, 2]][i_A, i_B], [[2, 3, 2]][i_A, i_C]
    # 3. regions = {"y"}, starts = [2, 2], [[0, 1, 4]][i_A, i_B], [[2, 3, 2]][i_A, i_C]
    # 4. regions = {"y"}, starts = [2, 3], [[0, 1, 4]][i_A, i_B], [[2, 3, 5]][i_A, i_C]
    #
    # TODO: There a particular cases (e.g. multiple regions but only a single
    # component or no matching regions) where it is sufficient to do an affine
    # layout instead of tabulating a start expression. We currently do not detect
    # this.
    starts = [0] * len(to_tabulate)
    for regions in _collect_regions(axis_tree):
        for i, (offset_axes, offset_dat) in enumerate(to_tabulate):

            # Axes do not match the current region set, this means that it is
            # zero-sized.
            if not all(region in offset_axes._all_region_labels for region in regions):
                continue

            regioned_axes = offset_axes.with_region_labels(regions)
            assert not regioned_axes._all_region_labels

            # Add the global offset to the values in this region
            if starts[i] > 0:  # don't bother adding 0 to things
                loop_(ix := regioned_axes.index(), offset_dat[ix].iassign(starts[i]), eager=True)

            # Figure out how large the looped-over part of the tree is (including subaxes)
            # as this will inform the stride size.
            step_size = axis_tree.linearize(offset_axes.leaf_path, partial=True).with_region_labels(regions).size or 1

            # Add to the starting offset for all arrays apart from the current one
            for j, _ in enumerate(starts):
                if i != j:
                    starts[j] += step_size

    return layouts


def _prepare_layouts(axis_tree: AxisTree, path_acc, layout_expr_acc, to_tabulate, tabulated, parent_axes) -> idict:
    """Traverse the axis tree and compute the layout functions.

    Any layout functions related to regions and thus requiring global
    tabulation are marked as such during the traversal.

    Parameters
    ----------
    layout_expr_acc :
        The accumulated layout function from the traversal of the parent axes.
        Each layout function is always the sum of the per-axis layout with this.

    """
    layouts = {}

    axis = axis_tree.node_map[path_acc]

    # Counter that tracks the offset between axis components.
    start = 0

    for component in axis.components:
        path_acc_ = path_acc | {axis.label: component.label}

        subtree = axis_tree.subtree(path_acc_)

        if not subtree.is_empty:
            subtree_has_non_trivial_regions = bool(subtree._all_region_labels)
        else:
            subtree_has_non_trivial_regions = False

        linear_axis = axis.linearize(component.label).localize()
        parent_axes_ = parent_axes + (linear_axis,)

        # The subtree contains regions so we cannot have a layout function here.
        if subtree_has_non_trivial_regions:
            layout_expr_acc_ = layout_expr_acc + start
            layouts[path_acc_] = NAN

        # At the bottom region - now can compute layouts involving all regions
        elif component.has_non_trivial_regions and not subtree_has_non_trivial_regions:
            offset_axes = AxisTree.from_iterable(parent_axes_)
            if subtree:
                offset_dat = _tabulate_regions(offset_axes, subtree.size, axis_tree.comm)
            else:
                offset_dat = _tabulate_regions(offset_axes, 1, axis_tree.comm)
            to_tabulate.append((offset_axes, offset_dat))

            assert layout_expr_acc == 0
            layout_expr_acc_ = offset_dat.concretize()
            layouts[path_acc_] = layout_expr_acc_

        # At leaves the layout function is trivial
        elif subtree.is_empty:
            layout_expr_acc_ = layout_expr_acc + AxisVar(linear_axis) + start
            layouts[path_acc_] = layout_expr_acc_

        # Tabulate
        else:
            step_expr = _accumulate_step_sizes(subtree.size, linear_axis, axis_tree.comm)

            if linear_axis not in utils.just_one(get_shape(step_expr)).nodes:
                step_expr = AxisVar(linear_axis) * step_expr

            layout_expr_acc_ = layout_expr_acc + step_expr + start
            layouts[path_acc_] = layout_expr_acc_

        start += compute_axis_tree_component_size(axis_tree, path_acc, component.label)

        if axis_tree.node_map[path_acc_]:
            sublayouts = _prepare_layouts(axis_tree, path_acc_, layout_expr_acc_, to_tabulate, tabulated, parent_axes_)
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
@functools.singledispatch
def _accumulate_step_sizes(obj: Any, axis, comm):
    """TODO

    This is suitable for ragged expressions where step sizes need to be accumulated
    into an offset array.

    """
    raise TypeError


@_accumulate_step_sizes.register(numbers.Number)
def _(num: numbers.Number, /, axis: Axis, comm) -> numbers.Number:
    return num


@_accumulate_step_sizes.register(op3_expr.Mul)
def _(mul: op3_expr.Mul, /, axis: Axis, comm) -> op3_expr.Mul:
    return _accumulate_step_sizes(mul.a, axis, comm) * _accumulate_step_sizes(mul.b, axis, comm)


@_accumulate_step_sizes.register(op3_expr.Add)
def _(add: op3_expr.Add, /, axis: Axis, comm) -> op3_expr.Add:
    return _accumulate_step_sizes(add.a, axis, comm) + _accumulate_step_sizes(add.b, axis, comm)


@_accumulate_step_sizes.register(op3_expr.Sub)
def _(sub: op3_expr.Sub, /, axis: Axis, comm) -> op3_expr.Sub:
    return _accumulate_step_sizes(sub.a, axis, comm) - _accumulate_step_sizes(sub.b, axis, comm)


@_accumulate_step_sizes.register(op3_expr.Comparison)
def _(cond: op3_expr.Comparison, /, axis: Axis, comm) -> op3_expr.Comparison:
    return type(cond)(*(_accumulate_step_sizes(op, axis, comm) for op in cond.operands))


@_accumulate_step_sizes.register(op3_expr.Conditional)
def _(cond: op3_expr.Conditional, /, axis: Axis, comm) -> op3_expr.Conditional:
    return op3_expr.Conditional(*(_accumulate_step_sizes(op, axis, comm) for op in cond.operands))


@_accumulate_step_sizes.register(LinearDatBufferExpression)
def _(dat_expr: LinearDatBufferExpression, /, axis: Axis, comm) -> LinearDatBufferExpression:
    return _accumulate_dat_expr(dat_expr, axis, comm)


@scoped_cache()
def _accumulate_dat_expr(size_expr: LinearDatBufferExpression, linear_axis: Axis, comm):
    # If the current axis does not form part of the step expression then the
    # layout function is actually just 'size_expr * AxisVar(axis)'.
    if linear_axis not in utils.just_one(get_shape(size_expr)):
        return size_expr

    # We do an accumulate (exscan) over a single axis. This means that things
    # always start from zero and so we can add the result to the accumulated
    # layout functions.

    # do the moral equivalent of
    #
    #   for i
    #     for j  # (the current axis)
    #       offset[i, j] = offset[i, j-1] + size[i, j]

    # by definition the current axis is in size_expr but other axes may be needed from 'linear_axis'
    offset_axes_subtree = merge_axis_trees((utils.just_one(get_shape(size_expr)), full_shape(linear_axis.as_tree())[0]))
    size_expr_loop_tree, size_expr_loop_var_replace_map = get_loop_tree(size_expr)

    offset_axes = size_expr_loop_tree.add_subtree(size_expr_loop_tree.leaf_path, offset_axes_subtree)

    # remove current axis as we need to scan over it
    loc = utils.just_one(path for path, axis_ in offset_axes.node_map.items() if axis_ == linear_axis)
    outer_loop_tree = offset_axes.drop_node(loc)
    assert linear_axis not in outer_loop_tree.nodes

    offset_dat = Dat.zeros(offset_axes.regionless, dtype=IntType)

    size_expr_alt0 = replace(size_expr, size_expr_loop_var_replace_map)

    if not outer_loop_tree.is_empty:
        ix = outer_loop_tree.index()

        axis_to_loop_var_replace_map = {
            AxisVar(ax): LoopIndexVar(ix, ax)
            for ax in ix.iterset.nodes
        }

        size_expr_alt = replace(size_expr_alt0, axis_to_loop_var_replace_map)

        assignee = offset_dat[ix].concretize()
        scan_axis = replace_exprs(linear_axis, axis_to_loop_var_replace_map)
        loop_(
            ix, exscan(assignee, size_expr_alt, "+", scan_axis, assignee.user_comm), eager=True
        )

    else:
        # import pyop3
        # pyop3.extras.debug.maybe_breakpoint("b")
        exscan(offset_dat.concretize(), size_expr, "+", linear_axis, offset_dat.user_comm, eager=True)

    offset_expr = offset_dat.concretize()

    # more subst needed - replace the axes with loop indices...
    if not size_expr_loop_var_replace_map:
        return offset_expr
    else:
        invmap = utils.invert_mapping(size_expr_loop_var_replace_map)
        retval = replace(offset_expr, invmap)
        return retval


# This gets the sizes right for a particular dat, then we merge them above
@scoped_cache()
def _tabulate_regions(offset_axes, step, comm):
    # Regions are always tabulated using all available free indices (i.e. all
    # parent axes) because they get interleaved.

    # TODO: explain this algorithm using
    #
    #     {A: [{x: 2}, {y: 1}]}
    #     └──➤ {B: [{u: 2}, {v: 1}]}
    #
    #     [ 00, 01, 10, 11 ||  02, 12 || 20, 21 ||  22  ]
    #            "xu"           "xv"      "yu"     "yv"
    # from test_nested_mismatching_regions. Focus on how this is special because
    # we have the requisite region information in this case.

    # Construct the permutation from the natural ordering to the actual one.
    # Using the case above as an example this means generating the array
    #
    #     [0, 1, 3, 4, 2, 5, 6, 7, 8]
    #
    # This is done by looping over each region set in turn and writing the
    # offsets into a contiguous array. In this case this means writing
    # [0, 1, 3, 4] for region set 'xu', then [2, 5] for 'xv' into the next
    # available entries and so on.
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

    # We now have the necessary permutation but to compute offsets we actually
    # need to know the size of each entry. This is done by evaluating the 'step'
    # expression.
    # Note that unlike for the ragged case the offset computations here include
    # all of the available axes (there is no axis-wise 'exscan' here). This is
    # because the axes above this have not yet been tabulated so accumulation
    # is not a concern.
    step_dat = Dat.zeros(offset_axes.regionless, dtype=IntType)
    step_dat.assign(step, eager=True)

    # But the steps here are in the wrong order since they do not account for
    # the region interleaving. We therefore need to:
    #
    # 1. Reorder the steps into 'region' order
    reordered_steps = step_dat.data_ro[locs]
    # 2. Accumulate these steps to give us offsets
    reordered_offsets = utils.steps(reordered_steps)
    # 3. Undo the reordering
    offsets = reordered_offsets[utils.invert(locs)]

    return Dat(offset_axes.regionless, data=offsets)
