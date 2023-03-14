import pytest
import numpy as np
from firedrake import *


@pytest.mark.skipif(utils.complex_mode, reason="Don't expect coordinate derivatives to work in complex")
def test_first_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    n = FacetNormal(mesh)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX = TestFunction(mesh.coordinates.function_space())

    def test_first(J, dJ):
        actual = assemble(dJ).dat.data
        computed = assemble(derivative(J, X)).dat.data
        assert np.allclose(computed, actual, rtol=1e-14)

    Ja = u * u * dx
    dJa = u * u * div(dX) * dx
    test_first(Ja, dJa)

    Jb = inner(grad(u), grad(u)) * dx
    dJb = -2*inner(dot(grad(dX), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(dX) * dx
    test_first(Jb, dJb)

    f = x * y * sin(x) * cos(y)
    Jc = f * dx
    dJc = div(f*dX) * dx
    test_first(Jc, dJc)

    Jd = f * ds
    dJd = inner(grad(f), dX) * ds \
        + f * (div(dX) - inner(dot(grad(dX), n), n)) * ds
    test_first(Jd, dJd)

    sd = SubDomainData(x < 0.5)
    Je = u * u * dx(subdomain_data=sd)
    dJe = u * u * div(dX) * dx(subdomain_data=sd)
    test_first(Je, dJe)

    J = Ja + Jb + Jc + Jd + Je
    dJ = dJa + dJb + dJc + dJd + dJe
    test_first(J, dJ)


@pytest.mark.skipif(utils.complex_mode, reason="Don't expect coordinate derivatives to work in complex")
def test_mixed_derivatives():
    mesh = UnitSquareMesh(6, 6)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    v = TrialFunction(V)
    dX = TestFunction(mesh.coordinates.function_space())

    def test_mixed(J, dJ_manual):
        computed1 = assemble(derivative(derivative(J, X), u)).M.values
        computed2 = assemble(derivative(derivative(J, u), X)).M.values
        actuala = assemble(dJ_manual).M.values
        assert np.allclose(computed1, actuala, rtol=1e-14)
        assert np.allclose(computed2.T, actuala, rtol=1e-14)

    Ja = u * u * dx
    dJa = 2 * u * v * div(dX) * dx
    test_mixed(Ja, dJa)

    Jb = inner(grad(u), grad(u)) * dx
    dJb = 2*inner(grad(u), grad(v)) * div(dX) * dx \
        - 2*inner(dot(nabla_grad(dX), grad(u)), grad(v)) * dx \
        - 2*inner(grad(u), dot(nabla_grad(dX), grad(v))) * dx
    test_mixed(Jb, dJb)

    J = Ja+Jb
    dJ = dJa + dJb
    test_mixed(J, dJ)


@pytest.mark.skipif(utils.complex_mode, reason="Don't expect coordinate derivatives to work in complex")
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


@pytest.mark.skipif(utils.complex_mode, reason="Don't expect coordinate derivatives to work in complex")
def test_second_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    X = SpatialCoordinate(mesh)
    x, y = X
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX1 = TestFunction(mesh.coordinates.function_space())
    dX2 = TrialFunction(mesh.coordinates.function_space())

    def test_second(J, ddJ):
        computed = assemble(derivative(derivative(J, X, dX1), X, dX2)).M.values
        actual = assemble(ddJ).M.values
        assert np.allclose(computed, actual, rtol=1e-14)

    Ja = u * u * dx
    ddJa = u * u * div(dX1) * div(dX2) * dx - u * u * tr(grad(dX1)*grad(dX2)) * dx
    test_second(Ja, ddJa)

    Jb = inner(grad(u), grad(u)) * dx
    ddJb = 2*inner(dot(dot(nabla_grad(dX2), nabla_grad(dX1)), grad(u)), grad(u)) * dx \
        + 2*inner(dot(nabla_grad(dX1), dot(nabla_grad(dX2), grad(u))), grad(u)) * dx \
        + 2*inner(dot(nabla_grad(dX1), grad(u)), dot(nabla_grad(dX2), grad(u))) * dx \
        - 2*inner(dot(nabla_grad(dX2), grad(u)), grad(u)) * div(dX1) * dx \
        - inner(grad(u), grad(u)) * tr(nabla_grad(dX1)*nabla_grad(dX2)) * dx \
        - 2*inner(dot(nabla_grad(dX1), grad(u)), grad(u)) * div(dX2) * dx \
        + inner(grad(u), grad(u)) * div(dX1) * div(dX2) * dx
    test_second(Jb, ddJb)

    test_second(Ja+Jb, ddJa + ddJb)


@pytest.mark.skipif(utils.complex_mode, reason="Don't expect coordinate derivatives to work in complex")
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
