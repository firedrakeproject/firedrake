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


def test_cofunction_riesz_representation_l2_dat_version(V):
    f = Cofunction(V.dual())
    version = f.dat.dat_version
    _ = f.riesz_representation(riesz_map="l2")
    assert f.dat.dat_version == version


def test_riesz_map_options_prefix(V):
    options = PETSc.Options()
    options['riesz_ksp_type'] = 'richardson'
    options['riesz_ksp_max_it'] = '1'
    options['riesz_pc_type'] = 'none'

    riesz = RieszMap(V, options_prefix='riesz')
    with pytest.raises(ConvergenceError):
        f = Cofunction(V.dual()).assign(1.)
        riesz(f)

    del options['riesz_ksp_type']
    del options['riesz_ksp_max_it']
    del options['riesz_pc_type']
