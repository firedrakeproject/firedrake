from immutabledict import immutabledict as idict
import numpy as np

import pyop3 as op3


# this nearly works, and should be the right thing to do.
def check_layout(axis_tree, path, expected_offset):
    if not isinstance(path, collections.abc.Mapping):
        path = {axis_label: None for axis_label in path}
    path = idict(path)

    breakpoint()

    for ix in axis_tree.iter(eager=True):
        assert axis_tree.layout[path] == expected_offset(ix)


def test_ragged_with_nonstandard_axis_ordering():
    """Test that ragged axes are tabulated correctly.

    In this test there are 3 axes, where the innermost axis depends on the
    index of the outermost axis.

    """
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(2, "B")
    axis3 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "C")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2, axis3))

    layouts = axis_tree.layouts
    breakpoint()

    assert layouts[idict()] == 0
    assert layouts[idict({"A": None})] == [0, 2, 6]

    assert layouts[idict()] == [[0, 1], [2, 4], [6, 7]]
    check_layout(
        axis_tree,
        ["A", "B"],
        lambda i, j, k: [[0], [1, 2], [3]][i][k]*2 + j
    )
    # assert layouts[idict({"A": None, "B": None, "C": None})] == ???



def test_non_nested_matching_regions():
    axis1 = op3.Axis([
        op3.AxisComponent(1), op3.AxisComponent(1)
    ])
    axis21 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ])
    ])
    axis22 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ])
    ])
    axis_tree = op3.AxisTree.from_nest({axis1: [axis21, axis22]})

    l = axis_tree.layouts
    breakpoint()


def test_adjacent_mismatching_regions():
    """

    Should be as if the regions are not even there

    """
    axis = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ]),
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "X"),
            op3.AxisComponentRegion(1, "Y"),
        ]),
    ])
    axis_tree = axis.as_tree()

    assert axis_tree.size == 6

    l = axis_tree.layouts
    breakpoint()


def test_non_nested_mismatching_regions():
    """

    Should be as if the regions are not even there

    """
    axis1 = op3.Axis([
        op3.AxisComponent(1), op3.AxisComponent(1)
    ])
    axis21 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ])
    ])
    axis22 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "X"),
            op3.AxisComponentRegion(1, "Y"),
        ])
    ])
    axis_tree = op3.AxisTree.from_nest({axis1: [axis21, axis22]})

    l = axis_tree.layouts
    breakpoint()


def test_nested_regions():
    """Test that nested regions are partitioned correctly."""
    axis1 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ])
    ])
    axis2 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "X"),
            op3.AxisComponentRegion(1, "Y"),
        ])
    ])
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2))

    path1, path2, path3 = axis_tree.node_map.keys()

    assert axis_tree.layouts[path1] == 0
    assert axis_tree.layouts[path2] == op3.NAN

    leaf_layout = axis_tree.layouts[path3]
    # equivalent to [ 00, 01, 10, 11 || 02, 12 || 20, 21 || 22 ]
    assert (leaf_layout.buffer.buffer._data == [0, 1, 4, 2, 3, 5, 6, 7, 8]).all()


def test_ragged_nested_regions():
    """Test that nested regions are partitioned correctly.

    In this test the size of the inner region is dependent upon the outer
    axis.

    """
    axis1 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "A"),
            op3.AxisComponentRegion(1, "B"),
        ])
    ])
    axis2 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(
                op3.Dat(axis1, data=np.asarray([1, 2, 0], dtype=op3.IntType)),
                "X",
            ),
            op3.AxisComponentRegion(
                op3.Dat(axis1, data=np.asarray([2, 0, 1], dtype=op3.IntType)),
                "Y",
            ),
        ])
    ])
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2))

    assert axis_tree.size == 6

    path1, path2, path3 = axis_tree.node_map.keys()

    assert axis_tree.layouts[path1] == 0
    assert axis_tree.layouts[path2] == op3.NAN

    leaf_layout = axis_tree.layouts[path3]
    # equivalent to [ 00, 10, 11 || 01, 02 || <empty> || 20 ]
    #                    "AX"        "AY"      "BX"     "BY"
    assert (leaf_layout.buffer.buffer._data == [0, 3, 4, 1, 2, 5]).all()
