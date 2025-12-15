from __future__ import annotations

import numbers
import typing

from immutabledict import immutabledict as idict

from pyop3 import utils
from pyop3.buffer import ArrayBuffer
from pyop3.dtypes import IntType
from pyop3.expr import Dat, AxisVar, LoopIndexVar, ScalarBufferExpression
from pyop3.expr.base import loopified_shape  # TODO: move into visitors
from pyop3.insn import Loop
from pyop3.tree import AbstractAxisTree, AxisTree, UNIT_AXIS_TREE
from pyop3.tree.labelled_tree import as_path

if typing.TYPE_CHECKING:
    from pyop3.types import *


# NOTE: This is a very generic operation and I probably do something very similar elsewhere
def compute_axis_tree_size(axis_tree: AxisTree):
    if axis_tree.is_empty:
        return 0
    else:
        return _axis_tree_size_rec(axis_tree, idict())


def _axis_tree_size_rec(axis_tree: AxisTree, path):
    axis = axis_tree.node_map[path]

    if axis is None:
        return 1

    # The size of an axis tree is simply the product of the sizes of the
    # different nested subaxes. This remains the case even for ragged
    # inner axes.
    tree_size = 0
    for component in axis.components:
        tree_size += compute_axis_tree_component_size(axis_tree, path, component.label)
    return tree_size


def compute_axis_tree_component_size(axis_tree: AbstractAxisTree, path: PathT, component_label: ComponentLabelT):
    from pyop3 import Scalar
    from pyop3.expr.visitors import replace_terminals, replace

    path = as_path(path)

    axis = axis_tree.node_map[path]
    component = axis.matching_component(component_label)

    linear_axis = axis.linearize(component_label)

    path_ = path | {axis.label: component_label}
    if axis_tree.node_map[path_]:
        subtree_size = _axis_tree_size_rec(axis_tree, path_)
    else:
        subtree_size = 1

    # don't want to have <num> * <array> because (for example) the size of 3 * [1, 2, 1] is 4!
    # Therefore the right thing to do is to sum the internal bits.
    if not isinstance(subtree_size, numbers.Integral | Scalar | ScalarBufferExpression):
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
            return component.size * subtree_size
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

        Loop(i,
            component_size.iassign(subtree_size_expr)
        )()


        if component_size_axes is UNIT_AXIS_TREE:
            # ick way to make sure that if we have sizes wrapped up into Scalars that this
            # gets passed up
            mysize = utils.just_one(component_size.buffer.buffer._data)
            if not isinstance(subtree_size, numbers.Integral):
                sbuf = ArrayBuffer.from_scalar(mysize, constant=True)
                mysize = ScalarBufferExpression(sbuf)
            return mysize

        else:
            loop_to_axis_var_replace_map_ = utils.invert_mapping(axis_to_loop_var_replace_map)
            XXX = replace(component_size, loop_to_axis_var_replace_map_)

            axis_to_loop_var_replace_map = utils.invert_mapping(outer_loop_to_axis_var_replace_map)
            return replace(XXX, axis_to_loop_var_replace_map)
    else:
        return component.size * subtree_size


