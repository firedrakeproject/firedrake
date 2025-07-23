from immutabledict import immutabledict as idict

import pyop3 as op3


def test_nested_regions():
    """

    [[0, 1, 4], [2, 3, 5], [6, 7, 8]]

    """
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

    l = axis_tree.layouts
    breakpoint()


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
