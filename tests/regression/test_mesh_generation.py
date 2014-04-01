from math import pi
import pytest

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


def test_periodic_unit_interval():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(3)) - 1) < 1e-3


def test_periodic_interval():
    assert abs(integrate_one(PeriodicIntervalMesh(3, 5.0)) - 5.0) < 1e-3


def test_unit_square():
    assert abs(integrate_one(UnitSquareMesh(3, 3)) - 1) < 1e-3


def test_unit_cube():
    assert abs(integrate_one(UnitCubeMesh(3, 3, 3)) - 1) < 1e-3


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


@pytest.mark.xfail(reason='Periodic intervals not implemented in parallel')
@pytest.mark.parallel
def test_periodic_unit_interval_parallel():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(30)) - 1) < 1e-3


@pytest.mark.xfail(reason='Periodic intervals not implemented in parallel')
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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
