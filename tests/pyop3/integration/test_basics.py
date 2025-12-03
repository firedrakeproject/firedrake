import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.fixture
def vector_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (3,), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="vector_copy",
        lang_version=(2018, 2),
    )
    return op3.Function(code, [op3.READ, op3.WRITE])


def test_scalar_copy(factory):
    m = 10
    axis = op3.Axis(m)
    dat0 = op3.HierarchicalArray(
        axis, name="dat0", data=np.arange(axis.size, dtype=op3.ScalarType)
    )
    dat1 = op3.HierarchicalArray(
        axis,
        name="dat1",
        dtype=dat0.dtype,
    )

    kernel = factory.copy_kernel(1)
    # op3.do_loop(p := axis.index(), kernel(dat0[p], dat1[p]))
    loop = op3.loop(p := axis.index(), kernel(dat0[p], dat1[p]))
    loop()
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy(vector_copy_kernel):
    m, n = 10, 3

    axes = op3.AxisTree.from_nest({op3.Axis(m): op3.Axis(n)})
    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    op3.do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_vector_copy(vector_copy_kernel):
    m, n, a, b = 4, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": m, "pt1": n}): [op3.Axis(a), op3.Axis(b)]}
    )
    dat0 = op3.HierarchicalArray(
        axes,
        name="dat0",
        data=np.arange(axes.size),
        dtype=op3.ScalarType,
    )
    dat1 = op3.HierarchicalArray(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    op3.do_loop(
        p := axes.root["pt1"].index(),
        vector_copy_kernel(dat0[p, :], dat1[p, :]),
    )

    assert (dat1.data[: m * a] == 0).all()
    assert (dat1.data[m * a :] == dat0.data[m * a :]).all()


def test_copy_multi_component_temporary(vector_copy_kernel):
    m = 4
    n0, n1 = 2, 1

    axes = op3.AxisTree.from_nest(
        {op3.Axis(m): op3.Axis({"pt0": n0, "pt1": n1}, "ax1")}
    )
    dat0 = op3.HierarchicalArray(
        axes,
        name="dat0",
        data=np.arange(axes.size, dtype=op3.ScalarType),
    )
    dat1 = op3.HierarchicalArray(axes, name="dat1", dtype=dat0.dtype)

    # An explicit slice object is required because typical slice notation ":" is
    # ambiguous when there are multiple components that might be getting sliced.
    slice_ = op3.Slice(
        "ax1", [op3.AffineSliceComponent("pt0"), op3.AffineSliceComponent("pt1")]
    )

    op3.do_loop(
        p := axes.root.index(), vector_copy_kernel(dat0[p, slice_], dat1[p, slice_])
    )
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_scalar_copy_with_two_outer_loops(factory):
    m, n, a, b = 8, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m, "pt1": n}): [
                op3.Axis(a),
                op3.Axis(b),
            ]
        },
    )
    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(m * a + n * b), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(axes, name="dat1", dtype=dat0.dtype)

    kernel = factory.copy_kernel(1)
    op3.do_loop(p := axes["pt1", :].index(), kernel(dat0[p], dat1[p]))
    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])
