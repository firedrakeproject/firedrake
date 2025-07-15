import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(4)
    return FunctionSpace(mesh, "CG", 1)


def test_cofunction_assign_cofunction_with_subset(V):
    f = Cofunction(V.dual())
    f.dat.data_wo[...] = 1.0
    assert np.allclose(f.dat.data_ro, 1.0)

    g = Cofunction(V.dual())
    g.dat.data_wo[...] = 2.0

    import pyop3.extras.debug
    pyop3.extras.debug.enable_conditional_breakpoints()

    f.assign(g, subset=[0, 1, 2])
    assert np.allclose(f.dat.data_ro[:3], 2.0)
    assert np.allclose(f.dat.data_ro[3:], 1.0)


def test_cofunction_assign_scaled_cofunction_with_subset(V):
    f = Cofunction(V.dual())
    f.dat.data[:] = 1.0
    assert np.allclose(f.dat.data_ro, 1.0)

    g = Cofunction(V.dual())
    g.dat.data[:] = 2.0

    f.assign(-3 * g, subset=[0, 1, 2])
    assert np.allclose(f.dat.data_ro[:3], -6.0)
    assert np.allclose(f.dat.data_ro[3:], 1.0)


def test_scalar_cofunction_zero(V):
    f = Cofunction(V.dual())

    f.dat.data[:] = 1

    g = f.zero()
    assert f is g
    assert np.allclose(f.dat.data_ro, 0.0)


def test_scalar_cofunction_zero_with_subset(V):
    f = Cofunction(V.dual())
    # create an arbitrary subset consisting of the first two nodes
    assert V.node_count > 2

    f.dat.data[:] = 1

    g = f.zero(subset=[0, 1])
    assert f is g
    assert np.allclose(f.dat.data_ro[:2], 0.0)
    assert np.allclose(f.dat.data_ro[2:], 1.0)
