import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_properties(mesh):
    P = FunctionSpace(mesh, "DG", 0)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    g = Function(V)

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    f = lambda x, y: x*y
    pe = point_expr(f, function_space=P)
    pe2 = pe(u, g)

    _check_extop_attributes_(pe2, (u, g), P, (0, 0), ())

    assert pe2.operator_data == f
    assert pe2.expr == f


def test_pointwise_expr_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: x*y, function_space=P)
    p2 = p(u, v)
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == u*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    # Not evaluate on the same space hence the lack of precision
    try:
        assert abs(assemble_a1 - assemble_a2) < 1.0e-3
    except:
        raise ValueError('\n p2.ufl_operands:', p2.ufl_operands, '\n Value p2 operands: ', *tuple(e.dat.data_ro for e in p2.ufl_operands), '\n\t u: ', u, u.dat.data_ro, '\n\t v:', v, v.dat.data_ro, '\n\n assemble_a1: ', assemble_a1, ' assemble_a2:', assemble_a2, '\n\t assemble(p2.expr(u,v)): ', assemble(p2.expr(u, v)*dx), '\n\t intepol P: ', assemble(Function(P).interpolate(p2.expr(u, v))*dx))

    u2 = Function(V)
    g = Function(V).interpolate(cos(x))
    v = TestFunction(V)

    f = Function(V).interpolate(cos(x)*sin(y))
    p = point_expr(lambda x: x**2+1, function_space=V)
    p2 = p(g)

    F = (dot(grad(p2*u), grad(v)) + u*v)*dx - f*v*dx
    solve(F == 0, u)

    F2 = (dot(grad((g**2+1)*u2), grad(v)) + u2*v)*dx - f*v*dx
    solve(F2 == 0, u2)

    a1 = assemble(u*dx)
    a2 = assemble(u2*dx)
    err = (a1-a2)**2
    assert err < 1.0e-9


def test_compute_derivatives(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: 0.5*x**2*y, function_space=P)
    p2 = p(u, v)
    dp2du = p2._ufl_expr_reconstruct_(u, v, derivatives=(1, 0))
    a2 = dp2du*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert dp2du.derivatives == (1, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == 0.5*u**2*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    # Not evaluate on the same space hence the lack of precision
    assert abs(assemble_a1 - assemble_a2) < 1.0e-3


def test_scalar_check_equality(mesh):

    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


def test_vector_check_equality(mesh):

    V1 = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(as_vector([cos(x), sin(y)]))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


def test_tensor_check_equality(mesh):

    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = TensorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    phi = Function(V0).interpolate(as_vector([cos(x), sin(y)]))
    f = grad(phi)

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09
