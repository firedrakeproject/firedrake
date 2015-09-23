from math import pi
import pytest
import numpy as np

from firedrake import *
# Must come after firedrake import (that loads MPI)
try:
    import gmshpy
except ImportError:
    gmshpy = None


def integrate_one(m):
    V = FunctionSpace(m, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("1"))
    return assemble(u * dx)


def test_unit_interval():
    assert abs(integrate_one(UnitIntervalMesh(3)) - 1) < 1e-3


def test_interval():
    assert abs(integrate_one(IntervalMesh(3, 5.0)) - 5.0) < 1e-3


def test_interval_three_arg():
    assert abs(integrate_one(IntervalMesh(10, -1, 1)) - 2.0) < 1e-3


def test_interval_negative_length():
    with pytest.raises(RuntimeError):
        IntervalMesh(10, 2, 1)


def test_periodic_unit_interval():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(3)) - 1) < 1e-3


def test_periodic_interval():
    assert abs(integrate_one(PeriodicIntervalMesh(3, 5.0)) - 5.0) < 1e-3


def test_unit_square():
    assert abs(integrate_one(UnitSquareMesh(3, 3)) - 1) < 1e-3


def test_rectangle():
    assert abs(integrate_one(RectangleMesh(3, 3, 10, 2)) - 20) < 1e-3


def test_unit_cube():
    assert abs(integrate_one(UnitCubeMesh(3, 3, 3)) - 1) < 1e-3


def test_box():
    assert abs(integrate_one(BoxMesh(3, 3, 3, 1, 2, 3)) - 6) < 1e-3


def test_unit_circle():
    pytest.importorskip('gmshpy')
    assert abs(integrate_one(UnitCircleMesh(4)) - pi * 0.5 ** 2) < 0.02


def test_unit_triangle():
    assert abs(integrate_one(UnitTriangleMesh()) - 0.5) < 1e-3


def test_unit_tetrahedron():
    assert abs(integrate_one(UnitTetrahedronMesh()) - 0.5 / 3) < 1e-3


@pytest.mark.parallel
def test_unit_interval_parallel():
    assert abs(integrate_one(UnitIntervalMesh(30)) - 1) < 1e-3


@pytest.mark.parallel
def test_interval_parallel():
    assert abs(integrate_one(IntervalMesh(30, 5.0)) - 5.0) < 1e-3


@pytest.mark.parallel
def test_periodic_unit_interval_parallel():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(30)) - 1) < 1e-3


@pytest.mark.parallel
def test_periodic_interval_parallel():
    assert abs(integrate_one(PeriodicIntervalMesh(30, 5.0)) - 5.0) < 1e-3


@pytest.mark.parallel
def test_unit_square_parallel():
    assert abs(integrate_one(UnitSquareMesh(5, 5)) - 1) < 1e-3


@pytest.mark.parallel
def test_unit_cube_parallel():
    assert abs(integrate_one(UnitCubeMesh(3, 3, 3)) - 1) < 1e-3


@pytest.mark.skipif("gmshpy is None", reason='gmshpy not available')
@pytest.mark.parallel
def test_unit_circle_parallel():
    assert abs(integrate_one(UnitCircleMesh(4)) - pi * 0.5 ** 2) < 0.02


def assert_num_exterior_facets_equals_zero(m):
    # Need to initialise the mesh so that exterior facets have been
    # built.
    m.init()
    assert m.exterior_facets.set.total_size == 0


def run_icosahedral_sphere_mesh_num_exterior_facets():
    m = UnitIcosahedralSphereMesh(0)
    assert_num_exterior_facets_equals_zero(m)


def test_icosahedral_sphere_mesh_num_exterior_facets():
    run_icosahedral_sphere_mesh_num_exterior_facets()


@pytest.mark.parallel(nprocs=2)
def test_icosahedral_sphere_mesh_num_exterior_facets_parallel():
    run_icosahedral_sphere_mesh_num_exterior_facets()


def run_cubed_sphere_mesh_num_exterior_facets():
    m = UnitCubedSphereMesh(0)
    assert_num_exterior_facets_equals_zero(m)


def test_cubed_sphere_mesh_num_exterior_facets():
    run_cubed_sphere_mesh_num_exterior_facets()


@pytest.mark.parallel(nprocs=2)
def test_cubed_sphere_mesh_num_exterior_facets_parallel():
    run_cubed_sphere_mesh_num_exterior_facets()


def test_bendy_icos():
    for d in range(1, 4):
        m = IcosahedralSphereMesh(5.0, refinement_level=1, degree=d)
        coords = m.coordinates.dat.data
        assert np.allclose(np.linalg.norm(coords, axis=1), 5.0)


def test_bendy_icos_unit():
    for d in range(1, 4):
        m = UnitIcosahedralSphereMesh(refinement_level=1, degree=d)
        coords = m.coordinates.dat.data
        assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)


test_bendy_icos_parallel = pytest.mark.parallel(nprocs=2)(test_bendy_icos)
test_bendy_icos_unit_parallel = pytest.mark.parallel(nprocs=2)(test_bendy_icos_unit)


def test_bendy_cube():
    for d in range(1, 4):
        m = CubedSphereMesh(5.0, refinement_level=1, degree=d)
        coords = m.coordinates.dat.data
        assert np.allclose(np.linalg.norm(coords, axis=1), 5.0)


def test_bendy_cube_unit():
    for d in range(1, 4):
        m = UnitCubedSphereMesh(refinement_level=1, degree=d)
        coords = m.coordinates.dat.data
        assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)


test_bendy_cube_parallel = pytest.mark.parallel(nprocs=2)(test_bendy_cube)
test_bendy_cube_unit_parallel = pytest.mark.parallel(nprocs=2)(test_bendy_cube_unit)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
