import numbers

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
    assert dat.data_ro.sum() == 3


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


def test_dat_state_tracking(dat):
    d1 = dat
    d2 = op3.Dat.zeros_like(dat)

    assert d1.dat_version == 0
    assert d2.dat_version == 0

    # Access data_ro property
    d1.data_ro
    assert d1.dat_version == 0
    assert d2.dat_version == 0

    # Access data_rw property
    d1.data_rw
    assert d1.dat_version == 1
    assert d2.dat_version == 0

    # Access data_wo property
    d2.data_wo[...] += 1
    assert d1.dat_version == 1
    assert d2.dat_version == 1

    # Access zero property
    d1.zero(eager=True)
    assert d1.dat_version == 2
    assert d2.dat_version == 1

    # Copy d2 into d1
    d1.assign(d2, eager=True)
    assert d1.dat_version == 3
    assert d2.dat_version == 1

    # Context managers (modify d1 and d2)
    with d1.vec_wo as x:
        pass

    with d2.vec_rw as x:
        pass

    assert d1.dat_version == 4
    assert d2.dat_version == 2

    # parloop
    d3 = op3.Dat.zeros_like(d1)
    assert d3.dat_version == 0
    k = op3.Function.from_c_string(
        "write",
        "*v = 1;",
        [("v", d3.dtype, op3.WRITE)],
    )
    op3.loop(op3.Axis(1).iter(), k(d3), eager=True)
    assert d3.dat_version == 1


def test_accessing_data_with_halos_increments_dat_version(dat):
    assert dat.dat_version == 0
    dat.data_ro_with_halos
    assert dat.dat_version == 0
    dat.data_rw_with_halos
    assert dat.dat_version == 1


class TestDatLinalg:

    @pytest.fixture(params=[np.float64, np.complex128])
    def scalar_type(self, request):
        return request.param

    @pytest.fixture
    def axes(self):
        return op3.AxisTree.from_iterable([5, 3])

    @pytest.fixture
    def x(self, axes, scalar_type):
        return op3.Dat.zeros(axes, dtype=scalar_type)

    @pytest.fixture
    def y(self, axes, scalar_type):
        return op3.Dat(axes, data=np.arange(1, 16, dtype=scalar_type))

    @pytest.fixture
    def x2(self, scalar_type):
        return op3.Dat.from_sequence([1, 2, 3], dtype=scalar_type)

    @pytest.fixture
    def y2(self, scalar_type):
        return op3.Dat.from_sequence([6, 7], dtype=scalar_type)

    @pytest.fixture
    def z(self, x):
        return op3.Dat.zeros_like(x)

    def test_add(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(x+y, eager=True)
        assert (z.data_ro == 3 * y.data_ro).all()

    def test_sub(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(x-y, eager=True)
        assert (z.data_ro == y.data_ro).all()

    def test_mul(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(x*y, eager=True)
        assert (z.data_ro == 2 * y.data_ro * y.data_ro).all()

    def test_div(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(x/y, eager=True)
        assert (z.data_ro == 2.0).all()

    def test_add_shape_mismatch(self, x2, y2, z):
        with pytest.raises(ValueError):
            z.assign(x2+y2, eager=True)

    def test_sub_shape_mismatch(self, x2, y2, z):
        with pytest.raises(ValueError):
            z.assign(x2-y2, eager=True)

    def test_mul_shape_mismatch(self, x2, y2, z):
        with pytest.raises(ValueError):
            z.assign(x2*y2, eager=True)

    def test_div_shape_mismatch(self, x2, y2, z):
        with pytest.raises(ValueError):
            z.assign(x2/y2, eager=True)

    def test_add_scalar(self, x, y, z):
        x.data_wo[...] = y.data_ro + 1.0
        z.assign(y+1, eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_radd_scalar(self, x, y, z):
        x.data_wo[...] = y.data_ro + 1.0
        z.assign(1+y, eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_sub_scalar(self, x, y, z):
        x.data_wo[...] = y.data_ro - 1.0
        z.assign(y-1, eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_rsub_scalar(self, x, y, z):
        x.data_wo[...] = 1. - y.data_ro
        z.assign(1-y, eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_mul_scalar(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(y*2., eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_rmul_scalar(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(2.*y, eager=True)
        assert (z.data_ro == x.data_ro).all()

    def test_div_scalar(self, x, y, z):
        x.data_wo[...] = 2 * y.data_ro
        z.assign(x/2., eager=True)
        assert (z.data_ro == y.data_ro).all()

    def test_iadd(self, x, y):
        x.data_wo[...] = 2 * y.data_ro
        x += y
        assert (x.data == 3 * y.data).all()

    def test_isub(self, x, y):
        x.data_wo[...] = 2 * y.data_ro
        x -= y
        assert (x.data == y.data).all()

    def test_imul(self, x, y):
        x.data_wo[...] = 2 * y.data_ro
        x *= y
        assert (x.data == 2 * y.data * y.data).all()

    def test_iadd_shape_mismatch(self, x2, y2):
        with pytest.raises(ValueError):
            x2 += y2

    def test_isub_shape_mismatch(self, x2, y2):
        with pytest.raises(ValueError):
            x2 -= y2

    def test_imul_shape_mismatch(self, x2, y2):
        with pytest.raises(ValueError):
            x2 *= y2

    def test_iadd_scalar(self, x, y):
        x.data_wo[...] = y.data_ro + 1.
        y += 1.0
        assert (x.data == y.data).all()

    def test_isub_scalar(self, x, y):
        x.data_wo[...] = y.data_ro - 1.
        y -= 1.0
        assert (x.data == y.data).all()

    def test_imul_scalar(self, x, y):
        x.data_wo[...] = 2 * y.data_ro
        y *= 2.0
        assert (x.data == y.data).all()

    def test_idiv_scalar(self, x, y):
        x.data_wo[...] = 2 * y.data_ro
        x /= 2.0
        assert (x.data == y.data).all()

    def test_norm(self):
        d = op3.Dat.from_sequence([3, 4], dtype=float)
        assert abs(d.norm - 5) < 1e-12
        assert isinstance(d.norm, numbers.Real)

    def test_inner(self):
        d1 = op3.Dat.from_sequence([3, 4], dtype=float)
        d2 = op3.Dat.empty_like(d1)
        d2.data_wo[...] = [4, 5]

        i1 = d1.inner(d2)
        assert abs(i1 - 32) < 1e-12

        i2 = d2.inner(d1)
        assert abs(i2 - 32) < 1e-12
