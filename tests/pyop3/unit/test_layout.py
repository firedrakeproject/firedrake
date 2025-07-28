import collections
import re

import numpy as np
import pytest
from immutabledict import immutabledict as idict

import pyop3 as op3


def check_layout(axis_tree, path, pattern, offset_fn):
    if not isinstance(path, collections.abc.Mapping):
        path = {axis_label: None for axis_label in path}
    path = idict(path)

    layout_expr = axis_tree.layouts[path]

    assert re.fullmatch(op3.utils.regexify(pattern), str(layout_expr))

    # Before iterating drop the subtree and linearise
    iterset = axis_tree.drop_subtree(path, allow_empty_subtree=True).linearize(path)

    for path_, ix in iterset.iter(eager=True):
        assert path_ == path
        assert op3.evaluate(layout_expr, ix) == offset_fn(*ix.values())


def test_1d_affine_layout():
    axes = op3.Axis(5, "A").as_tree()
    check_layout(axes, ["A"], "i_{A}", lambda i: i)


def test_2d_affine_layout():
    axes = op3.AxisTree.from_iterable((op3.Axis(3, "A"), op3.Axis(2, "B")))
    check_layout(axes, ["A"], "(i_{A} * 2)", lambda i: 2*i)
    check_layout(axes, ["A", "B"], "((i_{A} * 2) + i_{B})", lambda i, j: 2*i + j)


def test_1d_multi_component_layout():
    axes = op3.Axis(
        [op3.AxisComponent(3, "a"), op3.AxisComponent(2, "b")],
        "A"
    ).as_tree()
    check_layout(axes, {"A": "a"}, "i_{A}", lambda i: i)
    check_layout(axes, {"A": "b"}, "(i_{A} + 3)", lambda i: i + 3)


# def test_1d_zero_sized_layout():
#     axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 0}, "ax0"))
#
#     layout0 = axes.layouts[pmap({"ax0": "pt0"})]
#
#     assert as_str(layout0) == "var_0"
#     # check_invalid_indices(axes, [[], [0]])


# def test_multi_component_layout_with_zero_sized_subaxis():
#     axes = op3.AxisTree.from_nest(
#         {
#             op3.Axis({"pt0": 2, "pt1": 1}, "ax0"): {
#                 "pt0": op3.Axis({"pt0": 0}, "ax1"),
#                 "pt1": op3.Axis({"pt0": 3}, "ax1"),
#             }
#         }
#     )
#
#     assert axes.size == 3
#
#     layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0"})]
#     layout1 = axes.layouts[freeze({"ax0": "pt1", "ax1": "pt0"})]
#
#     assert as_str(layout0) == "var_0"
#     assert as_str(layout1) == "var_0*3 + var_1"
#
#     check_offsets(
#         axes,
#         [
#             ([[0, 0], {"ax0": "pt1", "ax1": "pt0"}], 0),
#             ([[0, 1], {"ax0": "pt1", "ax1": "pt0"}], 1),
#             ([[0, 2], {"ax0": "pt1", "ax1": "pt0"}], 2),
#         ],
#     )
#     # check_invalid_indices(
#     #     axes,
#     #     [
#     #         [],
#     #         [("pt0", 0), 0],
#     #         [("pt1", 0), 3],
#     #         [("pt1", 1), 0],
#     #     ],
#     # )


# def test_ragged_layout():
#     nnz_axis = op3.Axis({"pt0": 3}, "ax0")
#     nnz = op3.HierarchicalArray(nnz_axis, data=np.asarray([2, 1, 2]), dtype=op3.IntType)
#
#     axes = op3.AxisTree.from_nest({nnz_axis: op3.Axis({"pt0": nnz}, "ax1")}).freeze()
#
#     layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]
#     array0 = just_one(collect_multi_arrays(layout0))
#
#     assert as_str(layout0) == "array_0 + var_0"
#     assert np.allclose(array0.data_ro, [0, 2, 3])
#     check_offsets(
#         axes,
#         [
#             ([[0, 0]], 0),
#             ([[0, 1]], 1),
#             ([[1, 0]], 2),
#             ([[2, 0]], 3),
#             ([[2, 1]], 4),
#         ],
#     )
#     # check_invalid_indices(
#     #     axes,
#     #     [
#     #         [-1, 0],
#     #         [0, -1],
#     #         [0, 2],
#     #         [1, -1],
#     #         [1, 1],
#     #         [2, -1],
#     #         [2, 2],
#     #         [3, 0],
#     #     ],
#     # )


# def test_ragged_layout_with_two_outer_axes():
#     axis0 = op3.Axis({"pt0": 2}, "ax0")
#     axis1 = op3.Axis({"pt0": 2}, "ax1")
#     nnz_axes = op3.AxisTree.from_nest(
#         {axis0: axis1},
#     )
#     nnz_data = np.asarray([[2, 1], [1, 2]])
#     nnz = op3.HierarchicalArray(nnz_axes, data=nnz_data.flatten(), dtype=op3.IntType)
#
#     axes = op3.AxisTree.from_nest(
#         {axis0: {axis1: op3.Axis({"pt0": nnz}, "ax2")}},
#     )
#
#     layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
#     array0 = just_one(collect_multi_arrays(layout0))
#
#     assert as_str(layout0) == "array_0 + var_0"
#     assert np.allclose(array0.data_ro, np.asarray([[0, 2], [3, 4]]).flatten())
#     check_offsets(
#         axes,
#         [
#             ([[0, 0, 0]], 0),
#             ([[0, 0, 1]], 1),
#             ([[0, 1, 0]], 2),
#             ([[1, 0, 0]], 3),
#             ([[1, 1, 0]], 4),
#             ([[1, 1, 1]], 5),
#         ],
#     )
#     # check_invalid_indices(
#     #     axes,
#     #     [
#     #         [0, 0, 2],
#     #         [0, 1, 1],
#     #         [1, 0, 1],
#     #         [1, 1, 2],
#     #         [1, 2, 0],
#     #         [2, 0, 0],
#     #     ],
#     # )


# def test_independent_ragged_axes():
#     axis0 = op3.Axis({"pt0": 2}, "ax0")
#     axis1 = op3.Axis({"pt0": 2}, "ax1")
#
#     nnz_data0 = np.asarray([2, 1])
#     nnz0 = op3.HierarchicalArray(axis0, name="nnz0", data=nnz_data0, dtype=op3.IntType)
#     nnz_data1 = np.asarray([1, 0])
#     nnz1 = op3.HierarchicalArray(axis1, name="nnz1", data=nnz_data1, dtype=op3.IntType)
#
#     axis2 = op3.Axis({"pt0": nnz0, "pt1": nnz1, "pt2": 2}, "ax2")
#     axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
#
#     assert axes.size == 16
#
#     layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
#     layout1 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt1"})]
#     layout2 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt2"})]
#
#     # NOTE: This is wrong, the start values here should index into some offset array,
#     # otherwise the layouts all start from zero.
#     assert as_str(layout0) == "array_0 + var_0"
#     assert as_str(layout1) == "array_0 + var_0"
#     assert as_str(layout2) == "array_0 + var_0"
#
#     array0 = single_valued(
#         just_one(collect_multi_arrays(l)) for l in [layout0, layout1, layout2]
#     )
#     assert (array0.data_ro == flatten([[0, 5], [9, 13]])).all()
#
#     check_offsets(
#         axes,
#         [
#             ([0, 0, ("pt0", 0)], 0),
#             ([0, 0, ("pt0", 1)], 1),
#             ([0, 0, ("pt1", 0)], 2),
#             ([0, 0, ("pt2", 0)], 3),
#             ([0, 0, ("pt2", 1)], 4),
#             ([0, 1, ("pt0", 0)], 5),
#             ([0, 1, ("pt0", 1)], 6),
#             ([0, 1, ("pt2", 0)], 7),
#             ([0, 1, ("pt2", 1)], 8),
#             ([1, 0, ("pt0", 0)], 9),
#             ([1, 0, ("pt1", 0)], 10),
#             ([1, 0, ("pt2", 0)], 11),
#             ([1, 0, ("pt2", 1)], 12),
#             ([1, 1, ("pt0", 0)], 13),
#             ([1, 1, ("pt2", 0)], 14),
#             ([1, 1, ("pt2", 1)], 15),
#         ],
#     )
#     # check_invalid_indices(
#     #     axes,
#     #     [
#     #         [0, 0, 2],
#     #         [0, 1, 1],
#     #         [1, 0, 1],
#     #         [1, 1, 2],
#     #         [1, 2, 0],
#     #         [2, 0, 0],
#     #     ],
#     # )


# def test_tabulate_nested_ragged_indexed_layouts():
#     axis0 = op3.Axis(3)
#     axis1 = op3.Axis(2)
#     axis2 = op3.Axis(2)
#     nnz_data = np.asarray([[1, 0], [3, 2], [1, 1]], dtype=op3.IntType).flatten()
#     nnz_axes = op3.AxisTree.from_iterable([axis0, axis1])
#     nnz = op3.HierarchicalArray(nnz_axes, data=nnz_data)
#     axes = op3.AxisTree.from_iterable([axis0, axis1, op3.Axis(nnz), axis2])
#     # axes = op3.AxisTree.from_iterable([axis0, op3.Axis(nnz), op3.Axis(2)])
#     # axes = op3.AxisTree.from_iterable([axis0, op3.Axis(nnz)])
#
#     p = axis0.index()
#     indexed_axes = just_one(axes[p].context_map.values())
#
#     layout = indexed_axes.layouts[indexed_axes.path(*indexed_axes.leaf)]
#     array0 = just_one(collect_multi_arrays(layout))
#     expected = np.asarray(steps(nnz_data, drop_last=True), dtype=op3.IntType) * 2
#     assert (array0.data_ro == expected).all()



def test_ragged_basic():
    """Test that ragged axes are tabulated correctly."""
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "B")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2))

    check_layout(axis_tree, ["A"], "array_#[i_{A}]", lambda i: [0, 1, 3][i])
    check_layout(axis_tree, ["A", "B"], "(array_#[i_{A}] + i_{B})", lambda i, j: [0, 1, 3][i] + j)


def test_ragged_with_scalar_subaxis():
    """Test that ragged axes are tabulated correctly."""
    axis1 = op3.Axis(3, "A")
    axis2 = op3.Axis(op3.Dat(axis1, data=np.asarray([1, 2, 1], dtype=op3.IntType)), "B")
    axis3 = op3.Axis(2, "C")
    axis_tree = op3.AxisTree.from_iterable((axis1, axis2, axis3))

    check_layout(axis_tree, ["A"], "(array_#[i_{A}] * 2)", lambda i: 2*[0, 1, 3][i])
    check_layout(axis_tree, ["A", "B"], "((array_#[i_{A}] * 2) + (i_{B} * 2))", lambda i, j: 2*[0, 1, 3][i] + 2*j)
    check_layout(axis_tree, ["A", "B", "C"], "(((array_#[i_{A}] * 2) + (i_{B} * 2)) + i_{C})", lambda i, j, k: 2*[0, 1, 3][i] + 2*j + k)


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
