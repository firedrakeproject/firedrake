import pytest
import numpy as np
from firedrake import *
from tests.common import *


def test_constant(cg1):
    f = interpolate(Constant(1.0), cg1)
    assert np.allclose(1.0, f.dat.data)


def test_function():
    m = UnitTriangleMesh()
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    f = interpolate(Expression("x[0]*x[0]"), V1)
    g = interpolate(f, V2)

    # g shall be equivalent to:
    h = interpolate(Expression("x[0]"), V2)

    assert np.allclose(g.dat.data, h.dat.data)


def test_coordinates(cg2):
    f = interpolate(Expression("x[0]*x[0]"), cg2)

    x = SpatialCoordinate(cg2.mesh())
    g = interpolate(x[0]*x[0], cg2)

    assert np.allclose(f.dat.data, g.dat.data)


def test_piola():
    m = UnitTriangleMesh()
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(Expression(("x[0]", "0.0")), U)
    g = interpolate(f[0], V)

    # g shall be equivalent to:
    h = project(f[0], V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_vector():
    m = UnitTriangleMesh()
    U = FunctionSpace(m, 'RT', 1)
    V = VectorFunctionSpace(m, 'P', 2)

    f = project(Expression(("x[0]", "0.0")), U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_constant_expression():
    m = UnitTriangleMesh()
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(Expression(("x[0]", "x[1]")), U)
    g = interpolate(div(f), V)

    assert np.allclose(2.0, g.dat.data)


def test_compound_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 2)
    V = FunctionSpace(m, 'P', 2)

    f = project(Expression(("x[0]", "x[1]")), U)
    g = interpolate(Constant(1.5)*div(f) + sin(x[0] * np.pi), V)

    # g shall be equivalent to:
    h = interpolate(Expression("3.0 + sin(pi * x[0])"), V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_cell_orientation():
    m = UnitCubedSphereMesh(2)
    m.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
    x = m.coordinates
    U = FunctionSpace(m, 'RTCF', 1)
    V = VectorFunctionSpace(m, 'DQ', 1)

    f = project(as_tensor([x[1], -x[0], 0.0]), U)
    g = interpolate(f, V)

    # g shall be close to:
    h = project(f, V)

    assert abs(g.dat.data - h.dat.data).max() < 1e-2


def test_mixed():
    m = UnitTriangleMesh()
    x = m.coordinates
    V1 = FunctionSpace(m, 'BDFM', 2)
    V2 = VectorFunctionSpace(m, 'P', 2)
    f = Function(V1 * V2)
    f.sub(0).project(as_tensor([x[1], -x[0]]))
    f.sub(1).interpolate(as_tensor([x[0], x[1]]))

    V = FunctionSpace(m, 'P', 1)
    g = interpolate(dot(grad(f[0]), grad(f[3])), V)

    assert np.allclose(1.0, g.dat.data)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
