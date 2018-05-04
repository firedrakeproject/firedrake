import pytest
import numpy as np
from firedrake import *


def test_first_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    n = FacetNormal(mesh)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX = TestFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed = assemble(derivative(J, X)).dat.data
    actual = assemble(u * u * div(dX) * dx).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)

    J = inner(grad(u), grad(u)) * dx
    computed = assemble(derivative(J, X)).dat.data
    dJdX = -2*inner(dot(grad(dX), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(dX) * dx
    actual = assemble(dJdX).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)

    f = x * y * sin(x) * cos(y)
    J = f * dx
    computed = assemble(derivative(J, X)).dat.data
    dJdX = div(f*dX) * dx
    actual = assemble(dJdX).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)

    J = f * ds
    computed = assemble(derivative(J, X)).dat.data
    dJdX = inner(grad(f), dX) * ds \
        + f * (div(dX) - inner(dot(grad(dX), n), n)) * ds
    actual = assemble(dJdX).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)


def test_mixed_derivatives():
    mesh = UnitSquareMesh(6, 6)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    v = TrialFunction(V)
    dX = TestFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed1 = assemble(derivative(derivative(J, X), u)).M.values
    computed2 = assemble(derivative(derivative(J, u), X)).M.values
    actual = assemble(2 * u * v * div(dX) * dx).M.values
    assert np.allclose(computed1, actual, rtol=1e-14)
    assert np.allclose(computed2.T, actual, rtol=1e-14)

    J = inner(grad(u), grad(u)) * dx
    computed1 = assemble(derivative(derivative(J, X), u)).M.values
    computed2 = assemble(derivative(derivative(J, u), X)).M.values
    actual = assemble(2*inner(grad(u), grad(v)) * div(dX) * dx
                      - 2*inner(dot(grad(dX), grad(u)), grad(v)) * dx
                      - 2*inner(grad(u), dot(grad(dX), grad(v))) * dx).M.values
    assert np.allclose(computed1, actual, rtol=1e-14)
    assert np.allclose(computed2.T, actual, rtol=1e-14)


def test_integral_scaling_edge_case():
    mesh = UnitSquareMesh(6, 6)
    X = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    J = u * u * dx
    with pytest.raises(ValueError):
        assemble(Constant(2.0) * derivative(J, X))
    with pytest.raises(ValueError):
        assemble(derivative(Constant(2.0) * derivative(J, X), X))
    with pytest.raises(ValueError):
        assemble(Constant(2.0) * derivative(derivative(J, X), X))


def test_second_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    X = SpatialCoordinate(mesh)
    x, y = X
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX1 = TestFunction(mesh.coordinates.function_space())
    dX2 = TrialFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed = assemble(derivative(derivative(J, X, dX1), X, dX2)).M.values
    actual = assemble(u * u * div(dX1) * div(dX2) * dx - u * u * tr(grad(dX1)*grad(dX2)) * dx).M.values
    assert np.allclose(computed, actual, rtol=1e-14)


def test_coordinate_handling():
    mesh = UnitSquareMesh(60, 60)
    V = VectorFunctionSpace(mesh, "CG", 1)
    X = SpatialCoordinate(mesh)
    dX = TestFunction(mesh.coordinates.function_space())

    # The shape derivative of an expression and a coefficient is different.  If
    # f is some expression, then the shape derivative of J = \int f dx is given
    # by dJ[V] = \int \nabla f \cdot V + f div(V) dx.  If however the
    # functional is given by J = \int u dx and u moves with the domain, as is
    # the case for finite element functions, then the shape derivative reads
    # dJ[V] = \int u div(V) dx. This can get confusing, because
    # mesh.coordinates is a special finite element function: it changes values
    # as the domain is moved. It is treated seperately and here we check that this
    # does the right thing.

    J1 = inner(mesh.coordinates, mesh.coordinates) * dx
    J2 = inner(X, X) * dx

    computed1 = assemble(derivative(J1, mesh.coordinates)).dat.data
    computed2 = assemble(derivative(J2, mesh.coordinates)).dat.data
    computed3 = assemble(derivative(J2, mesh.coordinates, dX)).dat.data
    computed4 = assemble(derivative(J1, X)).dat.data
    computed5 = assemble(derivative(J2, X)).dat.data
    computed6 = assemble(derivative(J2, X, dX)).dat.data
    actual = assemble(div(inner(X, X) * dX) * dx).dat.data

    assert np.allclose(actual, computed1, rtol=1e-14)
    assert np.allclose(actual, computed2, rtol=1e-14)
    assert np.allclose(actual, computed3, rtol=1e-14)
    assert np.allclose(actual, computed4, rtol=1e-14)

    # if we interpolate into some fe function first, the result should be different,
    # as the values of u are not updated with the deformation but stay the same

    u = Function(V)
    u.interpolate(X)
    J3 = inner(u, u) * dx
    computed5 = assemble(derivative(J3, X)).dat.data
    computed6 = assemble(derivative(J3, mesh.coordinates)).dat.data
    assert not np.allclose(computed1, computed5, rtol=1e-14)
    assert not np.allclose(computed1, computed6, rtol=1e-14)
    actual_moving = assemble(inner(u, u) * div(dX) * dx).dat.data
    assert np.allclose(actual_moving, computed5, rtol=1e-14)


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
