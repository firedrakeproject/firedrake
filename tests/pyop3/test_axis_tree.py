import collections
import re

import numpy as np
from immutabledict import immutabledict as idict

import pyop3 as op3


def check_subtree_size(axis_tree, path, pattern, size_fn):
    if not isinstance(path, collections.abc.Mapping):
        path = {axis_label: None for axis_label in path}
    path = idict(path)

    subtree = axis_tree.subtree(path)

    assert re.fullmatch(op3.utils.regexify(pattern), str(subtree.size))

    # Before iterating drop the subtree and linearise
    iterset = axis_tree.drop_subtree(path, allow_empty_subtree=True).linearize(path)

    for path_, ix in iterset.iter(eager=True):
        assert path_ == path
        assert op3.evaluate(subtree.size, ix) == size_fn(*ix.values())


def test_ragged_axis_tree_subtree_sizes():
    """Test subtree sizes.

    In this test the lowest axis depends on the value of the top axis but
    not the middle one.

    """
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(2, "B")
    axis3 = op3.Axis(
        op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)),
        "C",
    )
    axes = op3.AxisTree.from_iterable((axis1, axis2, axis3))
    assert axes.size == 8

    check_subtree_size(axes, ["A"], "(2 * array_#[i_{A}])", lambda i: 2 * [1, 2, 1][i])
    check_subtree_size(axes, ["A", "B"], "array_#[i_{A}]", lambda i, j: [1, 2, 1][i])
    check_subtree_size(axes, ["A", "B", "C"], "0", lambda i, j, k: 0)
