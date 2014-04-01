import pytest
import numpy as np
from firedrake import *


def integrate_one(intervals):
    m = UnitIntervalMesh(intervals)
    layers = intervals
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    V = FunctionSpace(mesh, 'CG', 1)

    u = Function(V)

    u.interpolate(Expression("1"))

    return assemble(u * dx)


def test_unit_interval():
    assert abs(integrate_one(5) - 1) < 1e-12


def test_interval_div_free():
    m = UnitIntervalMesh(50)
    mesh = ExtrudedMesh(m, 50)

    V = VectorFunctionSpace(mesh, 'CG', 3)

    u = Function(V)

    u.interpolate(Expression(('x[0]*x[0]*x[1]', '-x[0]*x[1]*x[1]')))

    # u is pointwise divergence free, so the integral should also be
    # div-free.
    assert np.allclose(assemble(div(u)*dx), 0)

    L2 = FunctionSpace(mesh, 'DG', 2)

    v = TestFunction(L2)

    f = assemble(div(u)*v*dx)

    # Check pointwise div-free
    assert np.allclose(f.dat.data, 0)


def test_periodic_interval_div_free():
    m = PeriodicUnitIntervalMesh(50)
    mesh = ExtrudedMesh(m, 50)

    V = VectorFunctionSpace(mesh, 'CG', 3)

    u = Function(V)

    u.interpolate(Expression(('sin(2*pi*x[0])',
                              '-2*pi*x[1]*cos(2*pi*x[0])')))

    # u is pointwise divergence free, so the integral should also be
    # div-free.
    assert np.allclose(assemble(div(u)*dx), 0)

    L2 = FunctionSpace(mesh, 'DG', 2)

    v = TestFunction(L2)

    f = assemble(div(u)*v*dx)

    # Check pointwise div-free
    assert np.allclose(f.dat.data, 0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
