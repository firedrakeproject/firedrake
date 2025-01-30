import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import freeze, pmap

import pyop3 as op3
from pyop3.axtree import merge_trees
from pyop3.dtypes import IntType
from pyop3.utils import UniqueNameGenerator, flatten, just_one, single_valued, steps


class RenameMapper(pym.mapper.IdentityMapper):
    """Mapper that renames variables in layout expressions.

    This enables one to obtain a consistent string representation of
    an expression.

    """

    def __init__(self):
        self.name_generator = None

    def __call__(self, expr):
        # reset the counter each time the mapper is used
        self.name_generator = UniqueNameGenerator()
        return super().__call__(expr)

    def map_axis_variable(self, expr):
        return pym.var(self.name_generator("var"))

    def map_array(self, expr):
        return pym.var(self.name_generator("array"))


class OrderedCollector(pym.mapper.CombineMapper):
    def combine(self, values):
        return sum(values, ())

    def map_constant(self, _):
        return ()

    map_variable = map_constant
    map_wildcard = map_constant
    map_dot_wildcard = map_constant
    map_star_wildcard = map_constant
    map_function_symbol = map_constant

    def map_array(self, expr):
        return (expr.array,)


_rename_mapper = RenameMapper()
_ordered_collector = OrderedCollector()


def as_str(layout):
    return str(_rename_mapper(layout))


def collect_multi_arrays(layout):
    return _ordered_collector(layout)


def check_offsets(axes, offset_args_and_offsets):
    for args, offset in offset_args_and_offsets:
        assert axes.offset(*args) == offset


def check_invalid_indices(axes, indicess):
    for indices, path in indicess:
        with pytest.raises(IndexError):
            axes.offset(indices, path)


@pytest.mark.parametrize("numbering", [None, [2, 3, 0, 4, 1]])
def test_1d_affine_layout(numbering):
    # the numbering should not change the final layout
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 5}, "ax0", numbering=numbering))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]

    assert as_str(layout0) == "var_0"
    check_offsets(
        axes,
        [
            ([0], 0),
            ([1], 1),
            ([2], 2),
            ([3], 3),
            ([4], 4),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         ({"ax0": 5}, {"ax0": "pt0"}),
    #     ])


def test_2d_affine_layout():
    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": 3}, "ax0"): op3.Axis({"pt0": 2}, "ax1")},
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]

    assert as_str(layout0) == "var_0*2 + var_1"
    check_offsets(
        axes,
        [
            ([[0, 0]], 0),
            ([[0, 1]], 1),
            ([[1, 0]], 2),
            ([[1, 1]], 3),
            ([[2, 0]], 4),
            ([[2, 1]], 5),
        ],
    )
    # check_invalid_indices(axes, [[3, 0], [0, 2], [1, 2], [2, 2]])


def test_1d_multi_component_layout():
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 3, "pt1": 2}, "ax0"))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]
    layout1 = axes.layouts[pmap({"ax0": "pt1"})]

    assert as_str(layout0) == "var_0"
    assert as_str(layout1) == "var_0 + 3"
    check_offsets(
        axes,
        [
            ([0, {"ax0": "pt0"}], 0),
            ([1, {"ax0": "pt0"}], 1),
            ([2, {"ax0": "pt0"}], 2),
            ([0, {"ax0": "pt1"}], 3),
            ([1, {"ax0": "pt1"}], 4),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [],
    #         [("pt0", -1)],
    #         [("pt0", 3)],
    #         [("pt1", -1)],
    #         [("pt1", 2)],
    #         [("pt0", 0), 0],
    #     ],
    # )


def test_1d_multi_component_permuted_layout():
    axes = op3.AxisTree.from_nest(
        op3.Axis(
            {"pt0": 3, "pt1": 2},
            "ax0",
            numbering=[4, 0, 3, 2, 1],
        )
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]
    layout1 = axes.layouts[pmap({"ax0": "pt1"})]

    assert as_str(layout0) == "array_0"
    assert as_str(layout1) == "array_0"
    assert np.allclose(layout0.array.data_ro, [1, 3, 4])
    assert np.allclose(layout1.array.data_ro, [0, 2])
    check_offsets(
        axes,
        [
            ([0, {"ax0": "pt0"}], 1),
            ([1, {"ax0": "pt0"}], 3),
            ([2, {"ax0": "pt0"}], 4),
            ([0, {"ax0": "pt1"}], 0),
            ([1, {"ax0": "pt1"}], 2),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [("pt0", -1)],
    #         [("pt0", 3)],
    #         [("pt1", -1)],
    #         [("pt1", 2)],
    #     ],
    # )


def test_1d_zero_sized_layout():
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 0}, "ax0"))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]

    assert as_str(layout0) == "var_0"
    # check_invalid_indices(axes, [[], [0]])


def test_multi_component_layout_with_zero_sized_subaxis():
    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": 2, "pt1": 1}, "ax0"): {
                "pt0": op3.Axis({"pt0": 0}, "ax1"),
                "pt1": op3.Axis({"pt0": 3}, "ax1"),
            }
        }
    )

    assert axes.size == 3

    layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0"})]
    layout1 = axes.layouts[freeze({"ax0": "pt1", "ax1": "pt0"})]

    assert as_str(layout0) == "var_0"
    assert as_str(layout1) == "var_0*3 + var_1"

    check_offsets(
        axes,
        [
            ([[0, 0], {"ax0": "pt1", "ax1": "pt0"}], 0),
            ([[0, 1], {"ax0": "pt1", "ax1": "pt0"}], 1),
            ([[0, 2], {"ax0": "pt1", "ax1": "pt0"}], 2),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [],
    #         [("pt0", 0), 0],
    #         [("pt1", 0), 3],
    #         [("pt1", 1), 0],
    #     ],
    # )


def test_permuted_multi_component_layout_with_zero_sized_subaxis():
    axis0 = op3.Axis({"pt0": 3, "pt1": 2}, "ax0", numbering=[3, 1, 4, 2, 0])
    axis1 = op3.Axis({"pt0": 0}, "ax1")
    axis2 = op3.Axis({"pt0": 3}, "ax1")
    axes = op3.AxisTree.from_nest({axis0: {"pt0": axis1, "pt1": axis2}})

    assert axes.size == 6

    layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0"})]
    layout1 = axes.layouts[freeze({"ax0": "pt1", "ax1": "pt0"})]

    assert as_str(layout0) == "array_0 + var_0"
    assert as_str(layout1) == "array_0 + var_0"

    array0 = just_one(collect_multi_arrays(layout0))
    array1 = just_one(collect_multi_arrays(layout1))
    assert (array0.data_ro == [3, 6, 6]).all()
    assert (array1.data_ro == [0, 3]).all()

    check_offsets(
        axes,
        [
            ([[0, 0], {"ax0": "pt1", "ax1": "pt0"}], 0),
            ([[0, 1], {"ax0": "pt1", "ax1": "pt0"}], 1),
            ([[0, 2], {"ax0": "pt1", "ax1": "pt0"}], 2),
            ([[1, 0], {"ax0": "pt1", "ax1": "pt0"}], 3),
            ([[1, 1], {"ax0": "pt1", "ax1": "pt0"}], 4),
            ([[1, 2], {"ax0": "pt1", "ax1": "pt0"}], 5),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [("pt0", 0), 0],
    #         [("pt1", 0)],
    #         [("pt1", 2), 0],
    #         [("pt1", 0), 3],
    #         [("pt1", 0), 0, 0],
    #     ],
    # )


def test_ragged_layout():
    nnz_axis = op3.Axis({"pt0": 3}, "ax0")
    nnz = op3.HierarchicalArray(nnz_axis, data=np.asarray([2, 1, 2]), dtype=op3.IntType)

    axes = op3.AxisTree.from_nest({nnz_axis: op3.Axis({"pt0": nnz}, "ax1")}).freeze()

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]
    array0 = just_one(collect_multi_arrays(layout0))

    assert as_str(layout0) == "array_0 + var_0"
    assert np.allclose(array0.data_ro, [0, 2, 3])
    check_offsets(
        axes,
        [
            ([[0, 0]], 0),
            ([[0, 1]], 1),
            ([[1, 0]], 2),
            ([[2, 0]], 3),
            ([[2, 1]], 4),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [-1, 0],
    #         [0, -1],
    #         [0, 2],
    #         [1, -1],
    #         [1, 1],
    #         [2, -1],
    #         [2, 2],
    #         [3, 0],
    #     ],
    # )


def test_ragged_layout_with_two_outer_axes():
    axis0 = op3.Axis({"pt0": 2}, "ax0")
    axis1 = op3.Axis({"pt0": 2}, "ax1")
    nnz_axes = op3.AxisTree.from_nest(
        {axis0: axis1},
    )
    nnz_data = np.asarray([[2, 1], [1, 2]])
    nnz = op3.HierarchicalArray(nnz_axes, data=nnz_data.flatten(), dtype=op3.IntType)

    axes = op3.AxisTree.from_nest(
        {axis0: {axis1: op3.Axis({"pt0": nnz}, "ax2")}},
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
    array0 = just_one(collect_multi_arrays(layout0))

    assert as_str(layout0) == "array_0 + var_0"
    assert np.allclose(array0.data_ro, np.asarray([[0, 2], [3, 4]]).flatten())
    check_offsets(
        axes,
        [
            ([[0, 0, 0]], 0),
            ([[0, 0, 1]], 1),
            ([[0, 1, 0]], 2),
            ([[1, 0, 0]], 3),
            ([[1, 1, 0]], 4),
            ([[1, 1, 1]], 5),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [0, 0, 2],
    #         [0, 1, 1],
    #         [1, 0, 1],
    #         [1, 1, 2],
    #         [1, 2, 0],
    #         [2, 0, 0],
    #     ],
    # )


@pytest.mark.xfail(reason="Adjacent ragged components do not yet work")
def test_independent_ragged_axes():
    axis0 = op3.Axis({"pt0": 2}, "ax0")
    axis1 = op3.Axis({"pt0": 2}, "ax1")

    nnz_data0 = np.asarray([2, 1])
    nnz0 = op3.HierarchicalArray(axis0, name="nnz0", data=nnz_data0, dtype=op3.IntType)
    nnz_data1 = np.asarray([1, 0])
    nnz1 = op3.HierarchicalArray(axis1, name="nnz1", data=nnz_data1, dtype=op3.IntType)

    axis2 = op3.Axis({"pt0": nnz0, "pt1": nnz1, "pt2": 2}, "ax2")
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})

    assert axes.size == 16

    layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
    layout1 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt1"})]
    layout2 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt2"})]

    # NOTE: This is wrong, the start values here should index into some offset array,
    # otherwise the layouts all start from zero.
    assert as_str(layout0) == "array_0 + var_0"
    assert as_str(layout1) == "array_0 + var_0"
    assert as_str(layout2) == "array_0 + var_0"

    array0 = single_valued(
        just_one(collect_multi_arrays(l)) for l in [layout0, layout1, layout2]
    )
    assert (array0.data_ro == flatten([[0, 5], [9, 13]])).all()

    check_offsets(
        axes,
        [
            ([0, 0, ("pt0", 0)], 0),
            ([0, 0, ("pt0", 1)], 1),
            ([0, 0, ("pt1", 0)], 2),
            ([0, 0, ("pt2", 0)], 3),
            ([0, 0, ("pt2", 1)], 4),
            ([0, 1, ("pt0", 0)], 5),
            ([0, 1, ("pt0", 1)], 6),
            ([0, 1, ("pt2", 0)], 7),
            ([0, 1, ("pt2", 1)], 8),
            ([1, 0, ("pt0", 0)], 9),
            ([1, 0, ("pt1", 0)], 10),
            ([1, 0, ("pt2", 0)], 11),
            ([1, 0, ("pt2", 1)], 12),
            ([1, 1, ("pt0", 0)], 13),
            ([1, 1, ("pt2", 0)], 14),
            ([1, 1, ("pt2", 1)], 15),
        ],
    )
    # check_invalid_indices(
    #     axes,
    #     [
    #         [0, 0, 2],
    #         [0, 1, 1],
    #         [1, 0, 1],
    #         [1, 1, 2],
    #         [1, 2, 0],
    #         [2, 0, 0],
    #     ],
    # )


def test_tabulate_nested_ragged_indexed_layouts():
    axis0 = op3.Axis(3)
    axis1 = op3.Axis(2)
    axis2 = op3.Axis(2)
    nnz_data = np.asarray([[1, 0], [3, 2], [1, 1]], dtype=op3.IntType).flatten()
    nnz_axes = op3.AxisTree.from_iterable([axis0, axis1])
    nnz = op3.HierarchicalArray(nnz_axes, data=nnz_data)
    axes = op3.AxisTree.from_iterable([axis0, axis1, op3.Axis(nnz), axis2])
    # axes = op3.AxisTree.from_iterable([axis0, op3.Axis(nnz), op3.Axis(2)])
    # axes = op3.AxisTree.from_iterable([axis0, op3.Axis(nnz)])

    p = axis0.index()
    indexed_axes = just_one(axes[p].context_map.values())

    layout = indexed_axes.layouts[indexed_axes.path(*indexed_axes.leaf)]
    array0 = just_one(collect_multi_arrays(layout))
    expected = np.asarray(steps(nnz_data, drop_last=True), dtype=op3.IntType) * 2
    assert (array0.data_ro == expected).all()


def test_tabulate_nested_ragged_independent_axes():
    """Test that inner ragged axes needn't be indexed by all the outer axes."""
    axis0 = op3.Axis({"a": 3}, "A")
    axis1 = op3.Axis({"b": 2}, "B")

    # only depends on axis0
    nnz2 = op3.HierarchicalArray(axis0, data=np.asarray([1, 2, 1], dtype=IntType))
    axis2 = op3.Axis({"c": nnz2}, "C")

    # only depends on axis1
    nnz3 = op3.HierarchicalArray(axis1, data=np.asarray([1, 2], dtype=IntType))
    axis3 = op3.Axis({"d": nnz3}, "D")

    axes = op3.AxisTree.from_iterable([axis0, axis1, axis2, axis3])

    path0 = pmap({"A": "a"})
    path01 = path0 | {"B": "b"}
    path012 = path01 | {"C": "c"}
    path0123 = path012 | {"D": "d"}

    assert as_str(axes.layouts[path0]) == "array_0"
    assert as_str(axes.layouts[path01]) == "array_0 + array_1"
    assert as_str(axes.layouts[path012]) == "array_0 + array_1 + array_2"
    assert as_str(axes.layouts[path0123]) == "array_0 + array_1 + array_2 + var_0"

    array0, array1, array2 = collect_multi_arrays(axes.layouts[path0123])

    assert {ax.label for ax in array0.axes.nodes} == {"A"}
    assert (array0.data_ro == [0, 3, 9]).all()

    assert {ax.label for ax in array1.axes.nodes} == {"A", "B"}
    assert (array1.data_ro == [0, 1, 0, 2, 0, 1]).all()

    assert {ax.label for ax in array2.axes.nodes} == {"A", "B", "C"}
    assert (array2.data_ro == [0, 0, 0, 1, 0, 2, 0, 0]).all()


class TestMergeTrees:
    @pytest.fixture
    def axis_a_xy(self):
        return op3.Axis({"x": 2, "y": 2}, "a")

    @pytest.fixture
    def axis_b_x(self):
        return op3.Axis({"x": 2}, "b")

    @pytest.fixture
    def axis_c_x(self):
        return op3.Axis({"x": 2}, "c")

    def test_merge_same_tree(self, axis_b_x):
        axes = op3.AxisTree(axis_b_x)
        assert merge_trees(axes, axes) == axes

    def test_merge_distinct_axes(self, axis_b_x, axis_c_x):
        axes1 = op3.AxisTree(axis_b_x)
        axes2 = op3.AxisTree(axis_c_x)

        expected = op3.AxisTree.from_iterable([axis_b_x, axis_c_x])
        assert merge_trees(axes1, axes2) == expected
