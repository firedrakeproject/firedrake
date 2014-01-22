import pytest

from firedrake import *


def integrate_one(intervals):
    m = UnitIntervalMesh(intervals)
    layers = intervals + 1
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    V = FunctionSpace(mesh, 'CG', 1)

    u = Function(V)

    u.interpolate(Expression("1"))

    return assemble(u * dx)


def test_unit_interval():
    assert abs(integrate_one(5) - 1) < 1e-12

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
