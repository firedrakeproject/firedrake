import loopy as lp
import numpy as np
import pytest
from pyrsistent import freeze, pmap

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


@pytest.fixture
def vector_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.INC])


# TODO make a function not a fixture
@pytest.fixture
def vector2_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.INC])


@pytest.fixture
def vec2_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vec2_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.INC])


@pytest.fixture
def vec6_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (6,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(code, [op3.READ, op3.INC])


@pytest.fixture
def vec12_inc_kernel():
    code = lp.make_kernel(
        ["{ [i]: 0 <= i < 6 }", "{ [j]: 0 <= j < 2 }"],
        "y[j] = y[j] + x[i, j]",
        [
            lp.GlobalArg("x", op3.ScalarType, (6, 2), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(code, [op3.READ, op3.INC])


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("indexed", [None, "slice", "subset"])
def test_inc_from_tabulated_map(
    scalar_inc_kernel, vector_inc_kernel, vector2_inc_kernel, nested, indexed
):
    m, n = 4, 3
    map_data = np.asarray([[1, 2, 0], [2, 0, 1], [3, 2, 3], [2, 0, 1]])

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes = op3.AxisTree.from_nest({axis: op3.Axis({"pt0": n}, "ax1")})
    map_dat = op3.Dat(
        map_axes,
        name="map0",
        data=map_data.flatten(),
        dtype=op3.IntType,
    )

    if indexed == "slice":
        map_dat = map_dat[:, 1:3]
        kernel = vector2_inc_kernel
    elif indexed == "subset":
        subset_ = op3.Dat(
            op3.Axis({"pt0": 2}, "ax1"),
            name="subset",
            data=np.asarray([1, 2]),
            dtype=op3.IntType,
        )
        map_dat = map_dat[:, subset_]
        kernel = vector2_inc_kernel
    else:
        kernel = vector_inc_kernel

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat),
            ],
        },
        "map0",
    )

    if nested:
        # op3.do_loop(
        loop = op3.loop(
            p := axis.index(),
            op3.loop(q := map0(p).index(), scalar_inc_kernel(dat0[q], dat1[p])),
        )
        loop()
    else:
        op3.do_loop(p := axis.index(), kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        if indexed == "slice":
            for j in range(1, 3):
                expected[i] += dat0.data_ro[map_data[i, j]]
        elif indexed == "subset":
            for j in [1, 2]:
                expected[i] += dat0.data_ro[map_data[i, j]]
        else:
            for j in range(n):
                expected[i] += dat0.data_ro[map_data[i, j]]
    assert np.allclose(dat1.data_ro, expected)


def test_inc_from_multi_component_temporary(vector_inc_kernel):
    m, n = 3, 4
    arity = 2
    map_data = np.asarray([[1, 2], [0, 1], [3, 2]])

    axis0 = op3.Axis({"pt0": m, "pt1": n}, "ax0")
    axis1 = axis0["pt0"].root

    dat0 = op3.MultiArray(
        axis0, name="dat0", data=np.arange(axis0.size), dtype=op3.ScalarType
    )
    dat1 = op3.MultiArray(axis1, name="dat1", dtype=dat0.dtype)

    # poor man's identity map
    map_axes0 = op3.AxisTree.from_nest({axis1: op3.Axis(1)})
    map_dat0 = op3.Dat(
        map_axes0,
        name="map0",
        data=np.arange(map_axes0.size),
        dtype=op3.IntType,
    )

    map_axes1 = op3.AxisTree.from_nest({axis1: op3.Axis(arity)})
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
                op3.TabulatedMapComponent("ax0", "pt1", map_dat1),
            ],
        },
        "map0",
    )

    op3.do_loop(p := axis1.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        expected[i] += dat0.data_ro[i]  # identity
        for j in range(arity):
            # add offset of m to reads since we are indexing the second
            # component (stored contiguously)
            expected[i] += dat0.data_ro[map_data[i, j] + m]
    assert np.allclose(dat1.data, expected)


def test_inc_with_multiple_maps(vector_inc_kernel):
    m = 5
    arity0, arity1 = 2, 1
    map_data0 = np.asarray([[1, 2], [0, 2], [0, 1], [3, 4], [2, 1]])
    map_data1 = np.asarray([[1], [1], [3], [0], [2]])

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0, "ax1")})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1, "ax1")})

    map_dat0 = op3.Dat(
        map_axes0,
        name="map0",
        data=map_data0.flatten(),
        dtype=op3.IntType,
    )
    map_dat1 = op3.Dat(
        map_axes1,
        name="map1",
        data=map_data1.flatten(),
        dtype=op3.IntType,
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        # FIXME
        # "map0",
        "ax1",
    )

    op3.do_loop(p := axis.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j0 in range(arity0):
            expected[i] += dat0.data_ro[map_data0[i, j0]]
        for j1 in range(arity1):
            expected[i] += dat0.data_ro[map_data1[i, j1]]
    assert np.allclose(dat1.data, expected)


@pytest.mark.parametrize("nested", [True, False])
def test_inc_with_map_composition(scalar_inc_kernel, vec6_inc_kernel, nested):
    m = 5
    arity0, arity1 = 2, 3
    map_data0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]])
    map_data1 = np.asarray(
        [[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]],
    )

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(m), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0)})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1)})

    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
            ],
        },
        "map0",
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        "map1",
    )

    if nested:
        op3.do_loop(
            p := axis.index(),
            op3.loop(
                q := map0(p).index(),
                op3.loop(r := map1(q).index(), scalar_inc_kernel(dat0[r], dat1[p])),
            ),
        )
    else:
        op3.do_loop(p := axis.index(), vec6_inc_kernel(dat0[map1(map0(p))], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in range(arity0):
            for k in range(arity1):
                expected[i] += dat0.data_ro[map_data1[map_data0[i, j], k]]
    assert np.allclose(dat1.data_ro, expected)


@pytest.mark.parametrize("nested", [True, False])
def test_vector_inc_with_map_composition(vec2_inc_kernel, vec12_inc_kernel, nested):
    m, n = 5, 2
    arity0, arity1 = 2, 3
    map_data0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]])
    map_data1 = np.asarray([[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]])

    axis = op3.Axis({"pt0": m}, "ax0")

    dat_axes = op3.AxisTree.from_nest({axis: op3.Axis({"pt0": n}, "ax1")})
    dat0 = op3.Dat(
        dat_axes, name="dat0", data=np.arange(dat_axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(dat_axes, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0)})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1)})

    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
            ],
        },
        "map0",
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        "map1",
    )

    if nested:
        op3.do_loop(
            p := axis.index(),
            op3.loop(
                q := map0(p).index(),
                op3.loop(r := map1(q).index(), vec2_inc_kernel(dat0[r, :], dat1[p, :])),
            ),
        )
    else:
        op3.do_loop(
            p := axis.index(), vec12_inc_kernel(dat0[map1(map0(p)), :], dat1[p, :])
        )

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in range(arity0):
            for k in range(arity1):
                idx = map_data1[map_data0[i, j], k]
                for d in range(n):
                    expected[i * n + d] += dat0.data_ro[idx * n + d]
    assert np.allclose(dat1.data_ro, expected)


def test_partial_map_connectivity(vector2_inc_kernel):
    axis = op3.Axis({"pt0": 3}, "ax0")
    dat0 = op3.Dat(axis, data=np.arange(3, dtype=op3.ScalarType))
    dat1 = op3.Dat(axis, dtype=dat0.dtype)

    map_axes = op3.AxisTree.from_nest({axis: op3.Axis(2)})
    map_data = [[0, 1], [2, 0], [2, 2]]
    map_array = np.asarray(flatten(map_data), dtype=op3.IntType)
    map_dat = op3.Dat(map_axes, data=map_array)

    # Some elements of map_ are not present in axis, so should be ignored
    map_ = op3.Map(
        {
            freeze({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat),
                op3.TabulatedMapComponent("not_ax0", "not_pt0", map_dat),
            ]
        },
    )

    op3.do_loop(p := axis.index(), vector2_inc_kernel(dat0[map_(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(3):
        for j in range(2):
            expected[i] += dat0.data_ro[map_data[i][j]]
    assert np.allclose(dat1.data_ro, expected)


def test_inc_with_variable_arity_map(scalar_inc_kernel):
    m = 3
    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size, dtype=op3.ScalarType)
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    nnz_data = np.asarray([3, 2, 1], dtype=op3.IntType)
    nnz = op3.Dat(axis, name="nnz", data=nnz_data, max_value=3)

    map_axes = op3.AxisTree.from_nest({axis: op3.Axis(nnz)})
    map_data = [[2, 1, 0], [2, 1], [2]]
    map_array = np.asarray(flatten(map_data), dtype=op3.IntType)
    map_dat = op3.Dat(map_axes, name="map0", data=map_array)
    map0 = op3.Map(
        {freeze({"ax0": "pt0"}): [op3.TabulatedMapComponent("ax0", "pt0", map_dat)]},
        name="map0",
    )

    op3.do_loop(
        p := axis.index(),
        op3.loop(q := map0(p).index(), scalar_inc_kernel(dat0[q], dat1[p])),
    )

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in map_data[i]:
            expected[i] += dat0.data_ro[j]
    assert np.allclose(dat1.data_ro, expected)


@pytest.mark.parametrize("method", ["codegen", "python"])
def test_loop_over_multiple_ragged_maps(factory, method):
    m = 5
    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size, dtype=op3.IntType)
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    # map0
    nnz0_data = np.asarray([3, 2, 1, 0, 3], dtype=op3.IntType)
    nnz0 = op3.Dat(axis, name="nnz0", data=nnz0_data)

    map0_axes = op3.AxisTree.from_nest({axis: op3.Axis(nnz0)})
    map0_data = [[2, 4, 0], [3, 3], [1], [], [4, 2, 1]]
    map0_array = np.asarray(op3.utils.flatten(map0_data), dtype=op3.IntType)
    map0_dat = op3.Dat(map0_axes, name="map0", data=map0_array)
    map0 = op3.Map(
        {freeze({"ax0": "pt0"}): [op3.TabulatedMapComponent("ax0", "pt0", map0_dat)]},
        name="map0",
    )

    # map1
    nnz1_data = np.asarray([2, 0, 3, 1, 2], dtype=op3.IntType)
    nnz1 = op3.Dat(axis, name="nnz1", data=nnz1_data)

    map1_axes = op3.AxisTree.from_nest({axis: op3.Axis(nnz1)})
    map1_data = [[4, 0], [], [1, 0, 0], [3], [2, 3]]
    map1_array = np.asarray(op3.utils.flatten(map1_data), dtype=op3.IntType)
    map1_dat = op3.Dat(map1_axes, name="map1", data=map1_array)
    map1 = op3.Map(
        {freeze({"ax0": "pt0"}): [op3.TabulatedMapComponent("ax0", "pt0", map1_dat)]},
        name="map1",
    )

    inc = factory.inc_kernel(1, op3.IntType)

    if method == "codegen":
        op3.do_loop(
            p := axis.index(),
            op3.loop(
                q := map1(map0(p)).index(),
                inc(dat0[q], dat1[p]),
            ),
        )
    else:
        assert method == "python"
        for p in axis.iter():
            for q in map1(map0(p.index)).iter({p}):
                prev_val = dat1.get_value(p.target_exprs, p.target_path)
                inc = dat0.get_value(q.target_exprs, q.target_path)
                dat1.set_value(p.target_exprs, prev_val + inc, p.target_path)

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in map0_data[i]:
            for k in map1_data[j]:
                expected[i] += dat0.data_ro[k]
    assert (dat1.data_ro == expected).all()


@pytest.mark.parametrize("method", ["codegen", "python"])
def test_loop_over_multiple_multi_component_ragged_maps(factory, method):
    m, n = 5, 6
    axis = op3.Axis({"pt0": m, "pt1": n}, "ax0")
    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size, dtype=op3.IntType)
    )
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    # pt0 -> pt0
    nnz00_data = np.asarray([3, 2, 1, 0, 3], dtype=op3.IntType)
    nnz00 = op3.Dat(axis["pt0"], name="nnz00", data=nnz00_data)
    map0_axes0 = op3.AxisTree.from_nest({axis["pt0"].root: op3.Axis(nnz00)})
    map0_data0 = [[2, 4, 0], [3, 3], [1], [], [4, 2, 1]]
    map0_array0 = np.asarray(op3.utils.flatten(map0_data0), dtype=op3.IntType)
    map0_dat0 = op3.Dat(map0_axes0, name="map00", data=map0_array0)

    # pt0 -> pt1
    nnz01_data = np.asarray([1, 2, 1, 0, 4], dtype=op3.IntType)
    nnz01 = op3.Dat(axis["pt0"], name="nnz01", data=nnz01_data)
    map0_axes1 = op3.AxisTree.from_nest({axis["pt0"].root: op3.Axis(nnz01)})
    map0_data1 = [[2], [1, 0], [2], [], [1, 4, 2, 1]]
    map0_array1 = np.asarray(op3.utils.flatten(map0_data1), dtype=op3.IntType)
    map0_dat1 = op3.Dat(map0_axes1, name="map01", data=map0_array1)

    # pt1 -> pt1 (pt1 -> pt0 not implemented)
    nnz1_data = np.asarray([2, 2, 1, 3, 0, 2], dtype=op3.IntType)
    nnz1 = op3.Dat(axis["pt1"], name="nnz1", data=nnz1_data)
    map1_axes = op3.AxisTree.from_nest({axis["pt1"].root: op3.Axis(nnz1)})
    map1_data = [[2, 5], [0, 1], [3], [5, 5, 5], [], [2, 1]]
    map1_array = np.asarray(op3.utils.flatten(map1_data), dtype=op3.IntType)
    map1_dat = op3.Dat(map1_axes, name="map1", data=map1_array)

    map_ = op3.Map(
        {
            freeze({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map0_dat0),
                op3.TabulatedMapComponent("ax0", "pt1", map0_dat1),
            ],
            freeze({"ax0": "pt1"}): [
                op3.TabulatedMapComponent("ax0", "pt1", map1_dat),
            ],
        },
        name="map_",
    )

    inc = factory.inc_kernel(1, op3.IntType)

    if method == "codegen":
        op3.do_loop(
            p := axis["pt0"].index(),
            op3.loop(
                q := map_(map_(p)).index(),
                inc(dat0[q], dat1[p]),
            ),
        )
    else:
        assert method == "python"
        for p in axis["pt0"].iter():
            for q in map_(map_(p.index)).iter({p}):
                prev_val = dat1.get_value(p.target_exprs, p.target_path)
                inc = dat0.get_value(q.target_exprs, q.target_path)
                dat1.set_value(p.target_exprs, prev_val + inc, p.target_path)

    # To see what is going on we can determine the expected result in two
    # ways: one pythonically and one equivalent to the generated code.
    # We leave both here for reference as they aid in understanding what
    # the code is doing.
    expected_pythonic = np.zeros_like(dat1.data_ro)
    for i in range(m):
        # pt0 -> pt0 -> pt0
        for j in map0_data0[i]:
            for k in map0_data0[j]:
                expected_pythonic[i] += dat0.data_ro[k]
        # pt0 -> pt0 -> pt1
        for j in map0_data0[i]:
            for k in map0_data1[j]:
                # add m since we are targeting pt1
                expected_pythonic[i] += dat0.data_ro[k + m]
        # pt0 -> pt1 -> pt1
        for j in map0_data1[i]:
            for k in map1_data[j]:
                # add m since we are targeting pt1
                expected_pythonic[i] += dat0.data_ro[k + m]

    expected_codegen = np.zeros_like(dat1.data_ro)
    for i in range(m):
        # pt0 -> pt0 -> pt0
        for j in range(nnz00_data[i]):
            map_idx = map0_data0[i][j]
            for k in range(nnz00_data[map_idx]):
                ptr = map0_data0[map_idx][k]
                expected_codegen[i] += dat0.data_ro[ptr]
        # pt0 -> pt0 -> pt1
        for j in range(nnz00_data[i]):
            map_idx = map0_data0[i][j]
            for k in range(nnz01_data[map_idx]):
                # add m since we are targeting pt1
                ptr = map0_data1[map_idx][k] + m
                expected_codegen[i] += dat0.data_ro[ptr]
        # pt0 -> pt1 -> pt1
        for j in range(nnz01_data[i]):
            map_idx = map0_data1[i][j]
            for k in range(nnz1_data[map_idx]):
                # add m since we are targeting pt1
                ptr = map1_data[map_idx][k] + m
                expected_codegen[i] += dat0.data_ro[ptr]

    assert (expected_pythonic == expected_codegen).all()
    assert (dat1.data_ro == expected_pythonic).all()


def test_map_composition(vec2_inc_kernel):
    arity0, arity1 = 3, 2

    iterset = op3.Axis({"pt0": 2}, "ax0")
    dat_axis0 = op3.Axis(10)
    dat_axis1 = op3.Axis(arity1)
    dat0 = op3.Dat(
        dat_axis0, name="dat0", data=np.arange(dat_axis0.size, dtype=op3.ScalarType)
    )
    dat1 = op3.Dat(dat_axis1, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({iterset: op3.Axis(arity0)})
    map_data0 = np.asarray([[2, 4, 0], [6, 7, 1]])
    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent(
                    dat_axis0.label, dat_axis0.component.label, map_dat0, label="a"
                ),
            ],
        },
    )

    # The labelling for intermediate maps is quite opaque, we use the ID of the
    # ContextFreeCalledMap nodes in the index tree. This is so we do not hit any
    # conflicts when we compose the same map multiple times. I am unsure how to
    # expose this to the user nicely, and this is a use case I do not imagine
    # anyone actually wanting, so I am unpicking the right label from the
    # intermediate indexed object.
    p = iterset.index()
    indexed_dat0 = dat0[map0(p)]
    cf_indexed_dat0 = indexed_dat0.with_context(
        {p.id: ({"ax0": "pt0"}, {"ax0": "pt0"})}
    )
    called_map_node = op3.utils.just_one(cf_indexed_dat0.axes.nodes)

    # this map targets the entries in map0 so it can only contain 0s, 1s and 2s
    map_axes1 = op3.AxisTree.from_nest({iterset: op3.Axis(arity1)})
    map_data1 = np.asarray([[0, 2], [2, 1]])
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent(
                    called_map_node.label, called_map_node.component.label, map_dat1
                ),
            ],
        },
    )

    op3.do_loop(p, vec2_inc_kernel(indexed_dat0[map1(p)], dat1))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(iterset.size):
        temp = np.zeros(arity0, dtype=dat0.dtype)
        for j0 in range(arity0):
            temp[j0] = dat0.data_ro[map_data0[i, j0]]
        for j1 in range(arity1):
            expected[j1] += temp[map_data1[i, j1]]
    assert np.allclose(dat1.data_ro, expected)


@pytest.mark.parametrize("method", ["codegen", "python"])
def test_recursive_multi_component_maps(method):
    m, n = 5, 6
    arity0_0, arity0_1, arity1 = 3, 2, 1

    axis = op3.Axis(
        {"pt0": m, "pt1": n},
        "ax0",
    )
    axis0 = axis["pt0"].root
    axis1 = axis["pt1"].root

    # maps from pt0 so the array has size (m, arity0_0)
    map_axes0_0 = op3.AxisTree.from_nest({axis0: op3.Axis(arity0_0)})
    # maps to pt0 so the maximum possible index is m - 1
    map_data0_0 = np.asarray(
        [[2, 4, 0], [2, 3, 1], [0, 2, 3], [1, 3, 4], [3, 1, 0]],
    )
    assert np.prod(map_data0_0.shape) == map_axes0_0.size
    map_dat0_0 = op3.Dat(
        map_axes0_0, name="map0_0", data=map_data0_0.flatten(), dtype=op3.IntType
    )

    # maps from pt0 so the array has size (m, arity0_1)
    map_axes0_1 = op3.AxisTree.from_nest({axis0: op3.Axis(arity0_1)})
    # maps to pt1 so the maximum possible index is n - 1
    map_data0_1 = np.asarray([[4, 5], [2, 1], [0, 3], [5, 0], [3, 2]])
    assert np.prod(map_data0_1.shape) == map_axes0_1.size
    map_dat0_1 = op3.Dat(
        map_axes0_1, name="map0_1", data=map_data0_1.flatten(), dtype=op3.IntType
    )

    # maps from pt1 so the array has size (n, arity1)
    map_axes1 = op3.AxisTree.from_nest({axis1: op3.Axis(arity1)})
    # maps to pt1 so the maximum possible index is n - 1
    map_data1 = np.asarray([[4], [5], [2], [3], [0], [1]])
    assert np.prod(map_data1.shape) == map_axes1.size
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    # map from pt0 -> {pt0, pt1} and from pt1 -> {pt1}
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0_0),
                op3.TabulatedMapComponent("ax0", "pt1", map_dat0_1),
            ],
            pmap({"ax0": "pt1"}): [
                op3.TabulatedMapComponent("ax0", "pt1", map_dat1),
            ],
        },
        "map0",
    )
    map1 = map0.copy(name="map1")

    dat0 = op3.Dat(
        axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axis["pt0"], name="dat1", dtype=dat0.dtype)

    # the temporary from the maps will look like:
    # Axis([3, 2], label=map0)
    # ├──➤ Axis([3, 2], label=map1)
    # │    ├──➤ None
    # │    └──➤ None
    # └──➤ Axis(1, label=map1)
    #      └──➤ None
    # which has 17 entries
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 17 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (17,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="sum_kernel",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    sum_kernel = op3.Function(lpy_kernel, [op3.READ, op3.INC])

    if method == "codegen":
        op3.do_loop(p := axis["pt0"].index(), sum_kernel(dat0[map1(map0(p))], dat1[p]))
    else:
        assert method == "python"
        for p in axis["pt0"].iter():
            for q in map1(map0(p.index)).iter({p}):
                prev_val = dat1.get_value(p.target_exprs, p.target_path)
                inc = dat0.get_value(q.target_exprs, q.target_path)
                dat1.set_value(p.target_exprs, prev_val + inc, p.target_path)

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        # cpt0, cpt0 (9 entries)
        packed00 = dat0.data_ro[:5][map_data0_0[map_data0_0[i]]]
        # cpt0, cpt1 (6 entries)
        packed01 = dat0.data_ro[5:][map_data0_1[map_data0_0[i]]]
        # cpt1, cpt1 (2 entries)
        packed11 = dat0.data_ro[5:][map_data1[map_data0_1[i]]]

        # in the local kernel we sum all the entries together
        expected[i] = np.sum(packed00) + np.sum(packed01) + np.sum(packed11)
    assert np.allclose(dat1.data_ro, expected)


def test_sum_with_consecutive_maps():
    size = 5
    m, n = 10, 4
    arity0 = 3
    arity1 = 2

    iterset = op3.Axis({"pt0": size}, "ax0")
    dat_axes0 = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": m}, "ax1"): op3.Axis({"pt0": n}, "ax2")},
    )

    dat0 = op3.Dat(
        dat_axes0, name="dat0", data=np.arange(dat_axes0.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(iterset, name="dat1", dtype=dat0.dtype)

    # map0 maps from the iterset to ax1
    map_axes0 = op3.AxisTree.from_nest({iterset: op3.Axis(arity0)})
    map_data0 = np.asarray(
        [[2, 9, 0], [6, 7, 1], [5, 3, 8], [9, 3, 2], [2, 4, 6]],
    )
    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax1", "pt0", map_dat0),
            ],
        },
        "map0",
    )

    # map1 maps from the iterset to ax2
    map_axes1 = op3.AxisTree.from_nest({iterset: op3.Axis(arity1)})
    map_data1 = np.asarray([[0, 2], [2, 1], [3, 1], [0, 0], [1, 2]])
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax2", "pt0", map_dat1),
            ],
        },
        "map1",
    )

    lpy_kernel = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {arity0} }}", f"{{ [j]: 0 <= j < {arity1} }}"],
        "y[0] = y[0] + x[i, j]",
        [
            lp.GlobalArg(
                "x", op3.ScalarType, (arity0, arity1), is_input=True, is_output=False
            ),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="sum",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    sum_kernel = op3.Function(lpy_kernel, [op3.READ, op3.WRITE])

    op3.do_loop(p := iterset.index(), sum_kernel(dat0[map0(p), map1(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(iterset.size):
        for j in range(arity0):
            for k in range(arity1):
                expected[i] += dat0.data_ro[map_data0[i, j] * n + map_data1[i, k]]
    assert np.allclose(dat1.data_ro, expected)
