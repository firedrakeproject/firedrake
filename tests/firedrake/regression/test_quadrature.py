from math import sqrt

from firedrake import *
import pytest
import numpy as np


@pytest.fixture(params=[False, True], ids=["triangle", "quadrilateral"])
def mesh(request):
    return UnitSquareMesh(5, 5, quadrilateral=request.param)


def test_hand_specified_quadrature(mesh):
    V = FunctionSpace(mesh, 'CG', 2)
    v = TestFunction(V)

    a = conj(v) * dx

    a_q0 = assemble(a, form_compiler_parameters={'quadrature_degree': 0})
    a_q2 = assemble(a, form_compiler_parameters={'quadrature_degree': 2})

    assert not np.allclose(a_q0.dat.data, a_q2.dat.data)


def test_hand_specified_max_quadrature():
    mesh = UnitIntervalMesh(1)

    x, = SpatialCoordinate(mesh)
    a = (x**4)*dx

    # These should be the same because we only need degree=4 for exact integration.
    x4 = assemble(a)
    x4_maxquad5 = assemble(a, form_compiler_parameters={"max_quadrature_degree": 5})

    assert np.isclose(x4, x4_maxquad5)

    # These should be the same because degree=2 will limit the quadrature
    x4_quad2 = assemble(a, form_compiler_parameters={"quadrature_degree": 2})
    x4_maxquad2 = assemble(a, form_compiler_parameters={"max_quadrature_degree": 2})

    assert np.isclose(x4_quad2, x4_maxquad2)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("mat_type", ["matfree", "aij"])
@pytest.mark.parametrize("family", ["Quadrature", "Boundary Quadrature"])
def test_quadrature_element(mesh, family, mat_type, diagonal):
    V = FunctionSpace(mesh, family, 2)
    v = TestFunction(V)
    u = TrialFunction(V)
    if family == "Boundary Quadrature":
        a = inner(u, v) * ds + inner(u('+'), v('+')) * dS
    else:
        a = inner(u, v) * dx

    assemble(a, mat_type=mat_type, diagonal=diagonal)


@pytest.fixture(params=[FunctionSpace, VectorFunctionSpace, TensorFunctionSpace],
                ids=lambda space: space.__name__)
def boundary_quadrature_data(mesh, request):
    V = request.param(mesh, "Boundary Quadrature", 2)
    x = SpatialCoordinate(mesh)
    if V.value_shape == ():
        expr = x[0]**2 + x[1]
    elif len(V.value_shape) == 1:
        expr = as_vector([x[0]**2, x[1]])
    else:
        expr = as_tensor([[x[0]**2, x[1]], [x[0]*x[1], 1 + x[0]]])
    return V, expr


@pytest.mark.parallel([1, 2])
def test_boundary_quadrature_interpolation(boundary_quadrature_data):
    V, expr = boundary_quadrature_data
    f = Function(V).interpolate(expr)
    error = f - expr

    # The functionals are point evaluations, so interpolation is exact there
    assert sqrt(abs(assemble(inner(error, error) * ds))) < 1e-12
    trace = error('+')
    assert sqrt(abs(assemble(inner(trace, trace) * dS))) < 1e-12


@pytest.mark.parallel([1, 2])
def test_boundary_quadrature_interpolation_in_form(boundary_quadrature_data):
    V, expr = boundary_quadrature_data
    error = interpolate(expr, V) - expr
    assert sqrt(abs(assemble(inner(error, error) * ds))) < 1e-12
    trace = error('+')
    assert sqrt(abs(assemble(inner(trace, trace) * dS))) < 1e-12
