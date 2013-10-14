from math import pi
import pytest

from firedrake import *


def integrate_one(m):
    V = FunctionSpace(m, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("1"))
    return assemble(u * dx)


@pytest.mark.parametrize(('volume', 'expected'), [
    (integrate_one(UnitIntervalMesh(3)), 1),
    (integrate_one(UnitSquareMesh(3, 3)), 1),
    (integrate_one(UnitCubeMesh(3, 3, 3)), 1),
    (integrate_one(UnitCircleMesh(15)), pi * 0.5 ** 2),
    (integrate_one(UnitTriangleMesh()), 0.5),
    (integrate_one(UnitTetrahedronMesh()), 0.5 / 3)])
def test_mesh_generation(volume, expected):
    assert abs(volume - expected) < 1e-3
