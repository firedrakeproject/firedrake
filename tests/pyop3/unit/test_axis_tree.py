import numpy as np

import pyop3 as op3


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

    assert axes.subtree({"A": None}).size == "TODO"
    assert axes.subtree({"A": None, "B": None}).size == [1, 2, 1]
