import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.compile.loopy import LOOPY_LANG_VERSION, LOOPY_TARGET


def test_scalar_copy(factory):
    m = 10
    axis = op3.Axis(m)
    dat0 = op3.Dat(axis, data=np.arange(axis.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(1)
    op3.loop(p := axis.iter(), kernel(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy(factory):
    m, n = 10, 3

    axes = op3.AxisTree.from_nest({op3.Axis(m): op3.Axis(n)})
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)
    op3.loop(p := axes.root.iter(), kernel(dat0[p, :], dat1[p, :]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


@pytest.mark.parametrize("use_slice", [False, True])
def test_multi_component_vector_copy(factory, use_slice):
    m, n, a, b = 4, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": m, "pt1": n}): [op3.Axis(a), op3.Axis(b)]}
    )
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)

    if use_slice:
        op3.loop(
            p := axes["pt1"].root.iter(),
            kernel(dat0[p, :], dat1[p, :]),
            eager=True,
        )
    else:
        op3.loop(
            p := axes["pt1"].root.iter(),
            kernel(dat0[p], dat1[p]),
            eager=True,
        )

    assert (dat1.data_ro[: m * a] == 0).all()
    assert (dat1.data_ro[m * a :] == dat0.data_ro[m * a :]).all()


def test_copy_multi_component_inner(factory):
    m = 4
    n0, n1 = 2, 1

    axes = op3.AxisTree.from_nest(
        {op3.Axis(m): op3.Axis({"pt0": n0, "pt1": n1}, "ax1")}
    )
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)
    op3.loop(
        p := axes.root.iter(), kernel(dat0[p], dat1[p]), eager=True,
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro)


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
    dat0 = op3.Dat(axes, data=np.arange(m * a + n * b, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(1)
    op3.loop(p := axes["pt1", :].iter(), kernel(dat0[p], dat1[p]), eager=True)
    assert all(dat1.data_ro[: m * a] == 0)
    assert all(dat1.data_ro[m * a :] == dat0.data_ro[m * a :])


def test_loop_over_parametrised_length(factory):
    length = op3.Scalar(np.int32(0))
    iter_axes = op3.Axis([op3.AxisComponent(length, "pt0")], "ax0")

    dat_axes = op3.Axis([op3.AxisComponent(10, "pt0")], "ax0")
    dat = op3.Dat.empty(dat_axes, dtype=int)

    set_one = factory.set_kernel(1, dtype=int)
    for l in [0, 3, 7, 10]:
        dat.zero(eager=True)
        assert (dat.data_ro == 0).all()
        length.assign(l, eager=True)
        op3.loop(p := iter_axes.iter(), set_one(dat[p]), eager=True)
        assert (dat.data_ro[:l] == 1).all()
        assert (dat.data_ro[l:] == 0).all()
