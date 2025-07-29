import collections
import re

import numpy as np
import pytest
from immutabledict import immutabledict as idict

import pyop3 as op3


def as_path(path):
    if not isinstance(path, collections.abc.Mapping):
        path = {axis_label: None for axis_label in path}
    return idict(path)



def check_layout(axis_tree, path, indices, offset_pattern, offset_fn):
    path = as_path(path)
    layout_expr = axis_tree.layouts[path]

    # Check the pattern
    assert re.fullmatch(op3.utils.regexify(offset_pattern), str(layout_expr))

    check_indices(axis_tree, path, indices)

    # Only loop over the subtree that we are investigating
    iterset = axis_tree.drop_subtree(path, allow_empty_subtree=True).linearize(path)

    # Check offsets
    for path_, ix in iterset.iter(eager=True):
        assert op3.evaluate(layout_expr, ix) == offset_fn(*ix.values())


def check_nan_layout(axis_tree, path, indices):
    path = as_path(path)

    layout_expr = axis_tree.layouts[path]
    assert layout_expr is op3.NAN

    check_indices(axis_tree, path, indices)


def check_indices(axis_tree, path, indices):
    # Only loop over the subtree that we are investigating
    iterset = axis_tree.drop_subtree(path, allow_empty_subtree=True).linearize(path)

    # Check indices and offsets
    indices_iter = iter(indices)
    for path_, ix in iterset.iter(eager=True):
        assert path_ == path
        assert tuple(ix.values()) == next(indices_iter)
    # Make sure all indices are consumed
    assert not set(indices_iter)


def test_1d_affine_layout():
    axis_tree = op3.Axis(5, "A").as_tree()

    assert axis_tree.size == 5

    check_layout(axis_tree, ["A"], [(i,) for i in range(5)], "i_{A}", lambda i: i)


def test_2d_affine_layout():
    axis_tree = op3.AxisTree.from_iterable((op3.Axis(3, "A"), op3.Axis(2, "B")))

    assert axis_tree.size == 6

    check_layout(
        axis_tree,
        ["A"],
        [(i,) for i in range(3)],
        "(i_{A} * 2)",
        lambda i: 2*i,
    )
    check_layout(
        axis_tree,
        ["A", "B"],
        [(i, j) for i in range(3) for j in range(2)],
        "((i_{A} * 2) + i_{B})",
        lambda i, j: 2*i + j,
    )


def test_1d_multi_component_layout():
    axis_tree = op3.Axis(
        [op3.AxisComponent(3, "a"), op3.AxisComponent(2, "b")],
        "A"
    ).as_tree()

    assert axis_tree.size == 5

    check_layout(axis_tree, {"A": "a"}, [(i,) for i in range(3)], "i_{A}", lambda i: i)
    check_layout(axis_tree, {"A": "b"}, [(i,) for i in range(2)], "(i_{A} + 3)", lambda i: i + 3)


def test_ragged_basic():
    """Test that ragged axes are tabulated correctly."""
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "B")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2))

    assert axis_tree.size == 4

    check_layout(
        axis_tree,
        ["A"],
        [(0,), (1,), (2,)],
        "array_#[i_{A}]",
        lambda i: [0, 1, 3][i],
    )
    check_layout(
        axis_tree,
        ["A", "B"],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        "(array_#[i_{A}] + i_{B})",
        lambda i, j: [0, 1, 3][i] + j,
    )


def test_ragged_with_scalar_subaxis():
    """Test that ragged axes are tabulated correctly."""
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "B")
    axis3 = op3.Axis(2, "C")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2, axis3))

    assert axis_tree.size == 8

    check_layout(
        axis_tree,
        ["A"],
        [(0,), (1,), (2,)],
        "(array_#[i_{A}] * 2)",
        lambda i: 2*[0, 1, 3][i],
    )
    check_layout(
        axis_tree,
        ["A", "B"],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        "((array_#[i_{A}] * 2) + (i_{B} * 2))",
        lambda i, j: 2*[0, 1, 3][i] + 2*j,
    )
    check_layout(
        axis_tree,
        ["A", "B", "C"],
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 0, 1)],
        "(((array_#[i_{A}] * 2) + (i_{B} * 2)) + i_{C})",
        lambda i, j, k: 2*[0, 1, 3][i] + 2*j + k,
    )


def test_ragged_with_multiple_ragged_subaxes():
    """Test that ragged axes are tabulated correctly.

    In this test there are 3 axes where the size of the inner axis depends on
    the size of the next axis out.

    """
    axis1 = op3.Axis(2, "A")
    axis2 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2], dtype=op3.IntType)), "B")
    axis3 = op3.Axis(op3.Dat(axis2, data=np.asarray([1, 2], dtype=op3.IntType)), "C")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2, axis3))

    check_layout(axis_tree, ["A"], "array_#[i_{A}]", lambda i: [0, 1][i])
    check_layout(
        axis_tree,
        ["A", "B"],
        "(array_#[i_{A}] + array_#[(array_#[i_{A}] + i_{B})])",
        lambda i, j: [0, 1][i] + [[0], [0, 1]][i][j],
    )
    check_layout(
        axis_tree,
        ["A", "B", "C"],
        "((array_#[i_{A}] + array_#[(array_#[i_{A}] + i_{B})]) + i_{C})",
        lambda i, j, k: [0, 1][i] + [[0], [0, 1]][i][j] + k,
    )


def test_ragged_with_nonstandard_axis_ordering():
    """Test that ragged axes are tabulated correctly.

    In this test there are 3 axes, where the innermost axis depends on the
    index of the outermost axis.

    """
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(2, "B")
    axis3 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "C")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2, axis3))

    assert axis_tree.size == 8

    check_layout(axis_tree, ["A"], "(2 * array_#[i_{A}])", lambda i: 2*[0, 1, 3][i])
    check_layout(
        axis_tree,
        ["A", "B"],
        "((2 * array_#[i_{A}]) + (i_{B} * array_#[i_{A}]))",
        lambda i, j: 2*[0, 1, 3][i] + [1, 2, 1][i]*j,
    )
    check_layout(
        axis_tree,
        ["A", "B", "C"],
        "(((2 * array_#[i_{A}]) + (i_{B} * array_#[i_{A}])) + i_{C})",
        lambda i, j, k: 2*[0, 1, 3][i] + [1, 2, 1][i]*j + k,
    )


# NOTE: For this single case can just do an affine thing - I don't think it's worth fixing
# def test_regions_basic():
#     axis_tree = op3.Axis([
#         op3.AxisComponent([
#             op3.AxisComponentRegion(2, "A"),
#             op3.AxisComponentRegion(1, "B"),
#         ])
#     ]).as_tree()



def test_non_nested_matching_regions():
    axis1 = op3.Axis(
        [op3.AxisComponent(1, "a"), op3.AxisComponent(1, "b")],
        "A",
    )
    axis21 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "x"),
            op3.AxisComponentRegion(1, "y"),
        ])
    ], "B")
    axis22 = op3.Axis([
        op3.AxisComponent([
            op3.AxisComponentRegion(2, "x"),
            op3.AxisComponentRegion(1, "y"),
        ])
    ], "C")
    axis_tree = op3.AxisTree.from_nest({axis1: [axis21, axis22]})

    assert axis_tree.size == 6

    check_nan_layout(axis_tree, {"A": "a"}, [(0,)])
    check_nan_layout(axis_tree, {"A": "b"}, [(0,)])
    check_layout(
        axis_tree,
        {"A": "a", "B": None},
        [(0, 0), (0, 1), (0, 2)],
        "(i_{B} + array_#[((i_{A} * 3) + i_{B})])",
        lambda i, j: j + [[0, 0, 2]][i][j],
    )
    check_layout(
        axis_tree,
        {"A": "b", "C": None},
        [(0, 0), (0, 1), (0, 2)],
        "(i_{C} + array_#[((i_{A} * 3) + i_{C})])",
        lambda i, j: j + [[2, 2, 3]][i][j],
    )


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
