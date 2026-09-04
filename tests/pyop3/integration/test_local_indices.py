# TODO arguably a bad file name/test layout
import numpy as np
import pytest

import pyop3 as op3


def test_copy_with_local_indices(scalar_copy_kernel):
    axis = op3.Axis(10)
    dat0 = op3.HierarchicalArray(axis, data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.HierarchicalArray(axis, dtype=dat0.dtype)

    op3.do_loop(
        p := axis.index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_copy_slice(scalar_copy_kernel):
    axis = op3.Axis(10)
    dat0 = op3.HierarchicalArray(
        axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(axis[:5], name="dat1", dtype=dat0.dtype)

    op3.do_loop(
        p := axis[::2].index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro[::2])


@pytest.mark.xfail(
    reason="Passing loop indices to the local kernel is not currently supported"
)
def test_pass_loop_index_as_argument(factory):
    m = 10
    axes = op3.Axis(m)
    dat = op3.HierarchicalArray(axes, dtype=op3.IntType)

    copy_kernel = factory.copy_kernel(1, dtype=dat.dtype)
    op3.do_loop(p := axes.index(), copy_kernel(p, dat[p]))
    assert (dat.data_ro == list(range(m))).all()


@pytest.mark.xfail(
    reason="Passing loop indices to the local kernel is not currently supported"
)
def test_pass_multi_component_loop_index_as_argument(factory):
    m, n = 10, 12
    axes = op3.Axis([m, n])
    dat = op3.HierarchicalArray(axes, dtype=op3.IntType)

    copy_kernel = factory.copy_kernel(1, dtype=dat.dtype)
    op3.do_loop(p := axes.index(), copy_kernel(p, dat[p]))

    expected = list(range(m)) + list(range(n))
    assert (dat.data_ro == expected).all()
