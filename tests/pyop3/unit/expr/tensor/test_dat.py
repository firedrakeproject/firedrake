import numpy as np
import pytest

import pyop3 as op3


@pytest.fixture
def dat():
    axes = op3.AxisTree.from_iterable([5, 3])
    return op3.Dat.zeros(axes)


def test_copy(dat):
    new_dat = dat.copy()
    dat.assign(1, eager=True)

    assert new_dat.axes == dat.axes
    assert np.allclose(new_dat.data_ro, 0)
    assert np.allclose(dat.data_ro, 1)


def test_eager_zero(dat):
    dat.assign(1, eager=True)
    assert np.allclose(dat.data_ro, 1)

    expr = dat.zero(eager=True)
    assert np.allclose(dat.data_ro, 0)
    assert expr is None, "Eager assignment returns 'None'"


def test_lazy_zero(dat):
    dat.assign(1, eager=True)
    assert np.allclose(dat.data_ro, 1)

    expr = dat.zero()
    assert np.allclose(dat.data_ro, 1)

    expr()
    assert np.allclose(dat.data_ro, 0)


def test_eager_assign(dat):
    expr = dat.assign(1, eager=True)
    assert np.allclose(dat.data_ro, 1)
    assert expr is None, "Eager assignment returns 'None'"


def test_lazy_assign(dat):
    expr = dat.assign(1)
    assert np.allclose(dat.data_ro, 0)

    expr()
    assert np.allclose(dat.data_ro, 1)


def test_assign_subset(dat):
    dat[::2, 1].assign(1, eager=True)
    assert np.allclose(dat.data_ro.reshape((5, 3))[::2, 1], 1)
    assert sum(dat.data_ro) == 3


def test_axpy(dat):
    dat2 = dat.copy()
    dat2.assign(2, eager=True)

    dat.axpy(3, dat2)
    assert np.allclose(dat.data_ro, 3*2)


def test_maxpy(dat):
    dat2 = dat.copy()
    dat3 = dat.copy()
    dat2.assign(2, eager=True)
    dat3.assign(3, eager=True)

    dat.maxpy((2, 3), (dat2, dat3))
    assert np.allclose(dat.data_ro, 2*2 + 3*3)
