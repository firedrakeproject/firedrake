import numpy as np

import pyop3 as op3


def test_transpose(scalar_copy_kernel):
    n = 5
    # axis0 and axis1 must have different labels
    axis0 = op3.Axis(n, "ax0")
    axis1 = op3.Axis(n, "ax1")
    axes0 = op3.AxisTree.from_nest({axis0: axis1})
    axes1 = op3.AxisTree.from_nest({axis1: axis0})

    dat0 = op3.HierarchicalArray(
        axes0, name="dat0", data=np.arange(axes0.size), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(axes1, name="dat1", dtype=dat0.dtype)

    op3.do_loop(
        p := axis0.index(),
        op3.loop(q := axis1.index(), scalar_copy_kernel(dat0[p, q], dat1[q, p])),
    )
    assert np.allclose(
        dat1.data.reshape((n, n)),
        dat0.data.reshape((n, n)).T,
    )


def test_nested_multi_component_loops(scalar_copy_kernel):
    a, b, c, d = 2, 3, 4, 5
    axis0 = op3.Axis({"a": a, "b": b}, "ax0")
    axis1 = op3.Axis({"c": c, "d": d}, "ax1")
    axis1_dup = axis1.copy(id=axis1.unique_id())
    axes = op3.AxisTree.from_nest({axis0: [axis1, axis1_dup]})

    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(axes.size, dtype=op3.ScalarType)
    )
    dat1 = op3.HierarchicalArray(axes, name="dat1", dtype=dat0.dtype)

    # op3.do_loop(
    loop = op3.loop(
        p := axis0.index(),
        op3.loop(q := axis1.index(), scalar_copy_kernel(dat0[p, q], dat1[p, q])),
    )
    loop()
    assert np.allclose(dat1.data_ro, dat0.data_ro)
